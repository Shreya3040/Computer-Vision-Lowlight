"""Verify downloaded LOL files against the sha256 manifest recorded in splits.json, and that splits are leakage-free."""
import argparse, hashlib, json
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--reference", default="configs/splits_reference.json"); a = ap.parse_args()
ref = json.load(open(a.reference)); cur = json.load(open(f"{a.data}/splits.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
bad = [k for k, (lo, hi) in ref["sha256_16"].items() if [sha(f"{a.data}/LOL/{k.split('/')[0]}/low/{k.split('/')[1]}"), sha(f"{a.data}/LOL/{k.split('/')[0]}/high/{k.split('/')[1]}")] != [lo, hi]]
print("LOL files checked:", len(ref["sha256_16"]), "| mismatches:", len(bad))
for k in ("train", "val", "test", "excluded_overlap"): print(f"  split {k:17s} identical to reference: {cur[k] == ref[k]}")
assert not bad and all(cur[k] == ref[k] for k in ("train", "val", "test")), "DATA MISMATCH"; print("OK")
