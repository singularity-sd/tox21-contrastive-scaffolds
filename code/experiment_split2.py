"""Contrastive fingerprint pretraining and supervised Tox21 prediction."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parent
RAW=ROOT/"data"/"tox21.csv.gz"
OUT=ROOT
TASKS=["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]
SEEDS=[17,29,43]
FP_BITS=1024
PRETRAIN_EPOCHS=80
MAX_EPOCHS,PATIENCE,BATCH=70,10,256
DEVICE=torch.device("cpu")
torch.set_num_threads(4)
RDLogger.DisableLog("rdApp.*")

def fingerprint(mol):
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=FP_BITS)
    arr=np.zeros((FP_BITS,),dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

def score_multi(y,probs):
    aps=[]; aucs=[]; per=[]
    for j,name in enumerate(TASKS):
        mask=np.isfinite(y[:,j]); yy=y[mask,j].astype(int); pp=probs[mask,j]
        if len(np.unique(yy))<2:
            ap=auc=float("nan")
        else:
            ap=float(average_precision_score(yy,pp)); auc=float(roc_auc_score(yy,pp))
            aps.append(ap); aucs.append(auc)
        per.append({"task":name,"n_labeled":int(mask.sum()),"n_positive":int(np.sum(yy==1)),"n_negative":int(np.sum(yy==0)),"average_precision":ap,"roc_auc":auc})
    return (float(np.mean(aps)) if aps else float("nan"),float(np.mean(aucs)) if aucs else float("nan"),per)

class FingerprintNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(FP_BITS,512),nn.ReLU(),nn.Dropout(.15),nn.Linear(512,256),nn.ReLU())
        self.projector=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64))
        self.head=nn.Linear(256,len(TASKS))
    def forward(self,x): return self.head(self.encoder(x))

def augment(x,drop_rate,rng):
    keep=(torch.rand(x.shape,device=x.device,generator=rng)>=drop_rate).float()
    return x*keep

def pretrain_contrastive(x_train,seed,drop_rate=.10,temperature=.20):
    torch.manual_seed(seed); model=FingerprintNet().to(DEVICE)
    opt=torch.optim.AdamW(list(model.encoder.parameters())+list(model.projector.parameters()),lr=1e-3,weight_decay=1e-4)
    x=torch.as_tensor(x_train,dtype=torch.float32,device=DEVICE)
    rng=torch.Generator(device=DEVICE).manual_seed(seed+2026)
    start=time.perf_counter()
    for epoch in range(PRETRAIN_EPOCHS):
        order=torch.randperm(len(x),generator=rng,device=DEVICE)
        model.train()
        for ix in order.split(BATCH):
            if len(ix)<2: continue
            xb=x[ix]
            v1=augment(xb,drop_rate,rng); v2=augment(xb,drop_rate,rng)
            z=F.normalize(torch.cat([model.projector(model.encoder(v1)),model.projector(model.encoder(v2))],dim=0),dim=1)
            sim=(z@z.T)/temperature
            sim.fill_diagonal_(-1e9)
            b=len(ix); targets=torch.cat([torch.arange(b,2*b,device=DEVICE),torch.arange(0,b,device=DEVICE)])
            loss=F.cross_entropy(sim,targets)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.encoder.state_dict(),time.perf_counter()-start

def macro_val_ap(y,prob):
    ap,_,_=score_multi(y,prob)
    return ap

def fit_supervised(xtr,ytr,xv,yv,seed,encoder_state=None):
    torch.manual_seed(seed)
    model=FingerprintNet().to(DEVICE)
    if encoder_state is not None: model.encoder.load_state_dict(encoder_state)
    x=torch.as_tensor(xtr,dtype=torch.float32,device=DEVICE)
    y=torch.as_tensor(np.nan_to_num(ytr,nan=0.0),dtype=torch.float32,device=DEVICE)
    mask=torch.as_tensor(np.isfinite(ytr),dtype=torch.float32,device=DEVICE)
    positives=np.nansum(ytr==1,axis=0); negatives=np.nansum(ytr==0,axis=0)
    pw=torch.as_tensor(np.clip(negatives/np.maximum(positives,1),1.0,25.0),dtype=torch.float32,device=DEVICE)
    xv_t=torch.as_tensor(xv,dtype=torch.float32,device=DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    best=-1.; best_state=None; best_epoch=0; wait=0; start=time.perf_counter()
    rng=torch.Generator(device=DEVICE).manual_seed(seed+7)
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        for ix in torch.randperm(len(x),generator=rng,device=DEVICE).split(BATCH):
            xb,yb,mb=x[ix],y[ix],mask[ix]
            logits=model(xb)
            loss_mat=F.binary_cross_entropy_with_logits(logits,yb,reduction="none",pos_weight=pw)
            loss=(loss_mat*mb).sum()/mb.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vp=torch.sigmoid(model(xv_t)).cpu().numpy()
        val_ap=macro_val_ap(yv,vp)
        if val_ap>best+1e-5:
            best,best_epoch,wait=val_ap,epoch,0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=PATIENCE: break
    seconds=time.perf_counter()-start
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): test_model=model
    return test_model,best,best_epoch,seconds

def predict_model(model,x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x,dtype=torch.float32,device=DEVICE))).cpu().numpy()

def split_scaffolds(groups,y):
    idx=np.arange(len(groups))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=2028).split(idx,groups=groups))
    local_train,local_val=next(GroupShuffleSplit(n_splits=1,test_size=.12,random_state=2056).split(trainval,groups=groups[trainval]))
    return np.sort(trainval[local_train]),np.sort(trainval[local_val]),np.sort(test)

def main():
    df=pd.read_csv(RAW)
    fps=[]; ys=[]; groups=[]; smiles=[]; mol_ids=[]; bad=[]
    for ix,row in df.iterrows():
        smi=str(row["smiles"])
        mol=Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append({"row":int(ix),"mol_id":row.get("mol_id",""),"smiles":smi,"reason":"RDKit could not parse/sanitize this SMILES"})
            continue
        can=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        scaffold=MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles=Chem.MolToSmiles(scaffold,canonical=True,isomericSmiles=False) if scaffold.GetNumAtoms() else "ACYCLIC:"+can
        fps.append(fingerprint(mol)); ys.append(row[TASKS].to_numpy(dtype=float)); groups.append(scaffold_smiles); smiles.append(can); mol_ids.append(row.get("mol_id",""))
    X=np.asarray(fps,dtype=np.float32); Y=np.asarray(ys,dtype=np.float32); groups=np.asarray(groups); smiles=np.asarray(smiles)
    if X.shape[1]!=FP_BITS or Y.shape[1]!=12: raise RuntimeError("Unexpected Tox21 data dimensions")
    if len(set(smiles))!=len(smiles): raise RuntimeError("Duplicate canonical molecules need grouping before splitting")
    train,val,test=split_scaffolds(groups,Y)
    if (set(groups[train])&set(groups[val])) or (set(groups[train])&set(groups[test])) or (set(groups[val])&set(groups[test])):
        raise RuntimeError("Scaffold leakage across train, validation, and test")
    pd.DataFrame(bad).to_csv(OUT/"excluded_invalid.csv",index=False)
    split_rows=[]
    for name,ix in [("train",train),("validation",val),("test",test)]:
        split_rows.append({"split":name,"compounds":len(ix),"scaffolds":len(set(groups[ix])),"unique_canonical_smiles":len(set(smiles[ix]))})
    pd.DataFrame(split_rows).to_csv(OUT/"split2_audit_split.csv",index=False)
    counts=[]
    for name,ix in [("train",train),("validation",val),("test",test)]:
        for j,t in enumerate(TASKS):
            known=np.isfinite(Y[ix,j]); labels=Y[ix[known],j]
            counts.append({"split":name,"task":t,"n_labeled":len(labels),"n_positive":int(np.sum(labels==1)),"n_negative":int(np.sum(labels==0))})
    pd.DataFrame(counts).to_csv(OUT/"split2_support_per_task.csv",index=False)
    run_rows=[]; pred_rows=[]; encoder_durations=[]; all_warnings=[]
    # A linear fingerprint baseline, fit separately per task to respect missing labels.
    start=time.perf_counter(); logit_probs=np.full((len(test),len(TASKS)),np.nan)
    for j,t in enumerate(TASKS):
        mask=np.isfinite(Y[train,j]); model=LogisticRegression(C=1.0,class_weight="balanced",max_iter=500,solver="liblinear",random_state=2026)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X[train][mask],Y[train,j][mask].astype(int))
        all_warnings.extend([{"category":type(w.message).__name__,"message":str(w.message)} for w in caught])
        logit_probs[:,j]=model.predict_proba(X[test])[:,1]
    logit_seconds=time.perf_counter()-start
    ap,auc,per=score_multi(Y[test],logit_probs)
    run_rows.append({"model":"Logistic_fingerprint","seed":2026,"fit_seconds":logit_seconds,"total_training_seconds":logit_seconds,"macro_average_precision":ap,"macro_roc_auc":auc,"best_epoch":np.nan})
    for row in per: row["model"]="Logistic_fingerprint"
    pd.DataFrame(per).to_csv(OUT/"split2_logistic_per_task.csv",index=False)
    for j,i in enumerate(test):
        pred_rows.append({"model":"Logistic_fingerprint","seed":2026,"row":int(i),"mol_id":mol_ids[i],"scaffold":groups[i],"true_labels":json.dumps([None if not np.isfinite(v) else int(v) for v in Y[i]]),"probabilities":json.dumps(logit_probs[j].tolist())})
    xtr,ytr,xv,yv=X[train],Y[train],X[val],Y[val]
    for seed in SEEDS:
        enc_state,pre_seconds=pretrain_contrastive(xtr,seed,drop_rate=.10,temperature=.20)
        encoder_durations.append({"seed":seed,"pretrain_seconds":pre_seconds})
        for name,state in [("Contrastive_MLP",enc_state),("Supervised_MLP",None)]:
            model,val_ap,best_epoch,fit_seconds=fit_supervised(xtr,ytr,xv,yv,seed,state)
            probs=predict_model(model,X[test])
            ap,auc,per=score_multi(Y[test],probs)
            run_rows.append({"model":name,"seed":seed,"fit_seconds":fit_seconds,"pretrain_seconds":pre_seconds if name=="Contrastive_MLP" else 0.0,"total_training_seconds":fit_seconds+(pre_seconds if name=="Contrastive_MLP" else 0.0),"best_epoch":best_epoch,"validation_macro_average_precision":val_ap,"macro_average_precision":ap,"macro_roc_auc":auc})
            for row in per: row["model"]=name; row["seed"]=seed
            pd.DataFrame(per).to_csv(OUT/f"split2_per_task_{name}_seed{seed}.csv",index=False)
            for j,i in enumerate(test):
                pred_rows.append({"model":name,"seed":seed,"row":int(i),"mol_id":mol_ids[i],"scaffold":groups[i],"true_labels":json.dumps([None if not np.isfinite(v) else int(v) for v in Y[i]]),"probabilities":json.dumps(probs[j].tolist())})
            print(json.dumps({"model":name,"seed":seed,"validation_macro_ap":val_ap,"test_macro_ap":ap,"test_macro_auc":auc,"epochs":best_epoch}),flush=True)
    runs=pd.DataFrame(run_rows); runs.to_csv(OUT/"split2_metrics_run.csv",index=False)
    pred_df=pd.DataFrame(pred_rows); pred_df.to_csv(OUT/"split2_test_predictions.csv",index=False)
    synth=runs.groupby("model").agg(mean_macro_ap=("macro_average_precision","mean"),sd_macro_ap=("macro_average_precision","std"),mean_macro_auc=("macro_roc_auc","mean"),mean_fit_seconds=("fit_seconds","mean"),mean_total_training_seconds=("total_training_seconds","mean")).reset_index()
    synth.to_csv(OUT/"split2_metrics_summary.csv",index=False)
    # Three-seed probability-average comparison with scaffold-cluster bootstrap.
    probs_by_model={}
    for name in ["Contrastive_MLP","Supervised_MLP"]:
        sub=pred_df[pred_df.model==name].copy()
        each=[]
        for seed in SEEDS:
            each.append(np.stack(sub[sub.seed==seed].sort_values("row").probabilities.map(json.loads).to_list()))
        probs_by_model[name]=np.mean(each,axis=0)
    logit_rows=pred_df[pred_df.model=="Logistic_fingerprint"].sort_values("row")
    probs_by_model["Logistic_fingerprint"]=np.stack(logit_rows.probabilities.map(json.loads).to_list())
    ytest=Y[test]
    ensemble_rows=[]; ensemble_per=[]
    for name,pred in probs_by_model.items():
        ap,auc,per=score_multi(ytest,pred)
        ensemble_rows.append({"model":name,"macro_average_precision":ap,"macro_roc_auc":auc})
        for r in per: r["model"]=name; ensemble_per.append(r)
    pd.DataFrame(ensemble_rows).to_csv(OUT/"split2_metrics_ensemble.csv",index=False)
    pd.DataFrame(ensemble_per).to_csv(OUT/"split2_metrics_per_task_ensemble.csv",index=False)
    # Cluster bootstrap by test scaffold group. Resampling preserves all compounds on a sampled scaffold.
    uniq=np.unique(groups[test]); id_by={g:np.flatnonzero(groups[test]==g) for g in uniq}
    rng=np.random.default_rng(20261107); comparisons=[]
    for a,b in [("Contrastive_MLP","Supervised_MLP"),("Contrastive_MLP","Logistic_fingerprint")]:
        obs_a=score_multi(ytest,probs_by_model[a])[0]; obs_b=score_multi(ytest,probs_by_model[b])[0]
        diffs=[]
        for _ in range(1000):
            chosen=rng.choice(uniq,size=len(uniq),replace=True)
            ix=np.concatenate([id_by[g] for g in chosen])
            aa=score_multi(ytest[ix],probs_by_model[a][ix])[0]; bb=score_multi(ytest[ix],probs_by_model[b][ix])[0]
            diffs.append(aa-bb)
        comparisons.append({"model_a":a,"model_b":b,"macro_ap_difference":float(obs_a-obs_b),"bootstrap_95_low":float(np.quantile(diffs,.025)),"bootstrap_95_high":float(np.quantile(diffs,.975)),"test_scaffolds":len(uniq),"replicates":1000})
    pd.DataFrame(comparisons).to_csv(OUT/"split2_bootstrap_scaffold.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    order=["Contrastive_MLP","Supervised_MLP","Logistic_fingerprint"]
    axs[0].boxplot([runs[runs.model==m].macro_average_precision.dropna().to_numpy() for m in order],tick_labels=["Contrastive MLP","Supervised MLP","Logistic"],showmeans=True)
    axs[0].set_ylabel("Test macro average precision"); axs[0].set_title("Scaffold-held-out split"); axs[0].tick_params(axis="x",labelrotation=12)
    axs[1].bar(["Contrastive\nMLP","Supervised\nMLP","Logistic"],[synth.set_index("model").loc[m].mean_total_training_seconds for m in order],color=["#5b88b4","#e0a13c","#7a9c70"])
    axs[1].set_ylabel("Mean fit seconds"); axs[1].set_title("Training time"); 
    fig.tight_layout(); fig.savefig(OUT/"split2_fig_results.png",dpi=220); plt.close(fig)
    pd.DataFrame([{"category":"data_quality","message":f"Excluded {len(bad)} rows with SMILES that RDKit could not parse or sanitize."}]).to_csv(OUT/"split2_warnings.csv",index=False)
    # Write a compact complete machine-readable record.
    result={"forecast_id":"T3v3-2026-0102","dataset":"Tox21","n_raw":len(df),"n_valid":len(X),"n_invalid_excluded":len(bad),"n_scaffolds":len(set(groups)),"n_tasks":len(TASKS),"tasks":TASKS,"split":"Grouped by Bemis-Murcko scaffold; train/test seed 2028; validation seed 2056; all scaffold groups kept together; invalid SMILES excluded before splitting","split_counts":split_rows,"support_per_task":counts,"seed_results":runs.to_dict(orient="records"),"summary":synth.replace({np.nan:None}).to_dict(orient="records"),"ensemble_results":ensemble_rows,"bootstrap":comparisons,"contrastive_pretraining":{"train_only":True,"augmentation":"independent 10% ECFP bit dropout views","temperature":.20,"epochs":PRETRAIN_EPOCHS},"warnings":all_warnings,"scope":"One in-vitro multi-assay dataset; scaffold split tests chemical-series transfer, not human or clinical toxicity."}
    (OUT/"split2_results.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    print(synth.to_string(index=False),flush=True); print(pd.DataFrame(ensemble_rows).to_string(index=False),flush=True); print(pd.DataFrame(comparisons).to_string(index=False),flush=True)

if __name__=="__main__": main()



