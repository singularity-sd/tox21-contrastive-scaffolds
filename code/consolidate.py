from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
runs=[]; ens=[]; boots=[]; audits=[]; supports=[]
for split,prefix,audit_file,support_file,idx_seed in [
    ("Split 1","","audit_split.csv","support_per_task.csv","2026/2054"),
    ("Split 2","split2_","split2_audit_split.csv","split2_support_per_task.csv","2028/2056")]:
    m=pd.read_csv(P/f"{prefix}metrics_run.csv"); m["split"]=split; runs.append(m)
    e=pd.read_csv(P/f"{prefix}metrics_ensemble.csv"); e["split"]=split; ens.append(e)
    b=pd.read_csv(P/f"{prefix}bootstrap_scaffold.csv"); b["split"]=split; boots.append(b)
    a=pd.read_csv(P/audit_file); a["split"]=split; a["seed_test_validation"]=idx_seed; audits.append(a)
    s=pd.read_csv(P/support_file); s["split"]=split; supports.append(s)
runs=pd.concat(runs,ignore_index=True); ensembles=pd.concat(ens,ignore_index=True); bootstrap=pd.concat(boots,ignore_index=True); audit=pd.concat(audits,ignore_index=True)
runs.to_csv(P/"metrics_run_complete.csv",index=False); ensembles.to_csv(P/"metrics_ensemble_complete.csv",index=False); bootstrap.to_csv(P/"bootstrap_scaffold_complete.csv",index=False); audit.to_csv(P/"audit_partition_counts.csv",index=False)
per=[]
for split,prefix in [("Split 1",""),("Split 2","split2_")]:
    d=pd.read_csv(P/f"{prefix}metrics_per_task_ensemble.csv")
    a=d[d.model=="Contrastive_MLP"].set_index("task")
    b=d[d.model=="Supervised_MLP"].set_index("task")
    for task in a.index:
        per.append({"split":split,"task":task,"contrastive_ap":float(a.loc[task,"average_precision"]),"supervised_ap":float(b.loc[task,"average_precision"]),"ap_difference":float(a.loc[task,"average_precision"]-b.loc[task,"average_precision"]),"contrastive_auc":float(a.loc[task,"roc_auc"]),"supervised_auc":float(b.loc[task,"roc_auc"])})
per=pd.DataFrame(per); per.to_csv(P/"metrics_per_task_comparison.csv",index=False)
summary=runs.groupby(["split","model"],as_index=False).agg(mean_macro_ap=("macro_average_precision","mean"),sd_macro_ap=("macro_average_precision","std"),mean_macro_auc=("macro_roc_auc","mean"),mean_total_training_seconds=("total_training_seconds","mean"))
summary.to_csv(P/"metrics_summary_full.csv",index=False)
result={"forecast_id":"T3v3-2026-0102","forecast_position":111,"forecast_evidence_count":46,"dataset":"Tox21","n_raw":7831,"n_valid":7823,"n_invalid_excluded":8,"n_scaffolds":4099,"n_tasks":12,"split_description":"Bemis-Murcko scaffold grouped train/validation/test splits; all assay labels masked when missing","split_seeds":{"Split 1":"test 2026, validation 2054","Split 2":"test 2028, validation 2056"},"split_audits":audit.to_dict(orient="records"),"scaffold_overlap_audit":pd.read_csv(P/"audit_scaffold_complete.csv").to_dict(orient="records"),"metrics_by_split":summary.replace({np.nan:None}).to_dict(orient="records"),"ensemble_metrics":ensembles.to_dict(orient="records"),"assay_differences":per.to_dict(orient="records"),"paired_scaffold_bootstrap":bootstrap.to_dict(orient="records"),"interpretation":"Contrastive pretraining improved mean macro average precision over a supervised MLP and logistic fingerprints in both scaffold splits. The point estimate was positive for all 12 assays in split 1; per-assay differences in both splits are saved in metrics_per_task_comparison.csv. Results concern only in-vitro Tox21 assay labels.","warnings":[{"category":"data_quality","message":"Eight source rows had SMILES that RDKit could not parse or sanitize and were excluded before splitting; see excluded_invalid.csv."}]}
(P/"results_full.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
# Dual-split test AP comparison with three-seed variation.
fig,axs=plt.subplots(1,2,figsize=(9.1,3.55),sharey=True)
order=["Contrastive_MLP","Supervised_MLP","Logistic_fingerprint"]
labels=["Contrastive MLP","Supervised MLP","Logistic"]
colors=["#5b88b4","#e0a13c","#7a9c70"]
for ax,split in zip(axs,["Split 1","Split 2"]):
    part=runs[runs.split==split]
    vals=[part[part.model==m].macro_average_precision.dropna().to_numpy() for m in order]
    ax.boxplot(vals,tick_labels=labels,showmeans=True)
    ax.set_title(split+" (held-out scaffolds)")
    ax.tick_params(axis="x",labelrotation=12)
    ax.grid(axis="y",alpha=.2)
axs[0].set_ylabel("Test macro average precision")
fig.tight_layout(); fig.savefig(P.parent/"fig_results.png",dpi=220); plt.close(fig)
print(summary.to_string(index=False))
print(ensembles.to_string(index=False))
print(bootstrap.to_string(index=False))
print("Positive assay deltas by split:",per.groupby("split").ap_difference.apply(lambda x:int((x>0).sum())).to_dict())

