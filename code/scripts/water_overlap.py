import json
from xact.dataset import load_dataset
s=load_dataset("data/dataset_cod_water.pkl")
rows=json.load(open("data/water_bridge_labels.json"))
both=water_only=direct_only=0
for i,x in enumerate(s):
    direct=set((x["labels"][:,0]>0.5).nonzero()[0].tolist())
    wb=set(rows[i].get("bridged_atoms",[]))
    both+=len(direct & wb); water_only+=len(wb-direct); direct_only+=len(direct-wb)
out={"both":both,"water_only":water_only,"direct_only":direct_only,"water_total":both+water_only,"water_only_fraction":water_only/max(1,both+water_only)}
json.dump(out,open("data/water_direct_overlap.json","w"),indent=2)
print(out)
