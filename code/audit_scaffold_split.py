from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
P=Path(__file__).resolve().parent
RDLogger.DisableLog("rdApp.*")
df=pd.read_csv(P/"data"/"tox21.csv.gz")
raw_rows=[]; canonical=[]; groups=[]
for i,row in df.iterrows():
    mol=Chem.MolFromSmiles(str(row["smiles"]))
    if mol is None: continue
    can=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    scaffold=MurckoScaffold.GetScaffoldForMol(mol)
    key=Chem.MolToSmiles(scaffold,canonical=True,isomericSmiles=False) if scaffold.GetNumAtoms() else "ACYCLIC:"+can
    raw_rows.append(i); canonical.append(can); groups.append(key)
groups=np.asarray(groups); raw_rows=np.asarray(raw_rows); canonical=np.asarray(canonical)
audit=[]
for name,test_seed,val_seed in [("Split 1",2026,2054),("Split 2",2028,2056)]:
    idx=np.arange(len(groups))
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=test_seed).split(idx,groups=groups))
    local_train,local_val=next(GroupShuffleSplit(n_splits=1,test_size=.12,random_state=val_seed).split(trainval,groups=groups[trainval]))
    train=np.sort(trainval[local_train]); val=np.sort(trainval[local_val]); test=np.sort(test)
    sets={k:set(groups[v]) for k,v in [("train",train),("validation",val),("test",test)]}
    overlaps={"train_validation":len(sets["train"]&sets["validation"]),"train_test":len(sets["train"]&sets["test"]),"validation_test":len(sets["validation"]&sets["test"])}
    if any(overlaps.values()): raise RuntimeError(f"Scaffold overlap found: {name} {overlaps}")
    for key,ix in [("train",train),("validation",val),("test",test)]:
        audit.append({"split":name,"partition":key,"compounds":len(ix),"scaffolds":len(sets[key]),"test_seed":test_seed,"validation_seed":val_seed,**{f"overlap_{k}":v for k,v in overlaps.items()}})
        pd.DataFrame({"raw_row":raw_rows[ix],"canonical_smiles":canonical[ix],"scaffold":groups[ix],"partition":key}).to_csv(P/f"membership_{name.lower().replace(' ','')}_{key}.csv",index=False)
pd.DataFrame(audit).to_csv(P/"audit_scaffold_complete.csv",index=False)
print(pd.DataFrame(audit).to_string(index=False))
