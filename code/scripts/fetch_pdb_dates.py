"""Fetch initial PDB release dates for TACT complexes."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SESSION = requests.Session()


def get_date(pdb):
    for attempt in range(4):
        try:
            r = SESSION.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}",
                            timeout=30)
            if r.status_code == 200:
                d = r.json()
                return pdb, d.get("rcsb_accession_info", {}).get(
                    "initial_release_date", "")[:10]
        except Exception:
            pass
        time.sleep(0.4 * (attempt + 1))
    return pdb, None


def main():
    pdbs = sorted({l.split("\t")[1] for l in open("results/tact_inchikeys.tsv")
                   if len(l.split("\t")) > 1})
    out = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_date, p) for p in pdbs]
        for i, fut in enumerate(as_completed(futs), 1):
            p, d = fut.result()
            out[p] = d
            if i % 100 == 0:
                print(i, len(pdbs), flush=True)
    with open("results/pdb_release_dates.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("pdbs", len(pdbs), "dated", sum(v is not None for v in out.values()))


if __name__ == "__main__":
    main()
