import json
from collections import Counter, defaultdict
from xact.dataset import load_dataset
s=load_dataset("data/dataset_cod_water.pkl")
rd=json.load(open("results/pdb_release_dates.json"))
for cut in ["2022-01-01","2023-01-01","2024-01-01","2025-01-01"]:
    tr=[x for x in s if rd[x["pdb"]]<cut]
    te=[x for x in s if rd[x["pdb"]]>=cut]
    by=defaultdict(list)
    for x in s: by[x["target"]].append(rd[x["pdb"]])
    med={t:sorted(v)[len(v)//2] for t,v in by.items()}
    trt={t for t,m in med.items() if m<cut}; tet={t for t,m in med.items() if m>=cut}
    str_tr=[x for x in tr if x["target"] in trt]; str_te=[x for x in te if x["target"] in tet]
    print(cut, "all",len(tr),len(te),"targets",len(set(x["target"] for x in tr)),len(set(x["target"] for x in te)),
          "strict",len(str_tr),len(str_te),"targets",len(set(x["target"] for x in str_tr)),len(set(x["target"] for x in str_te)),
          "test_targets",sorted(set(x["target"] for x in str_te)))
