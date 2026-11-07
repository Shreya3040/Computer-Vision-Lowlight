"""Build leakage-safe LOL splits.

Finding (see README / docs): LOL `our485` contains many near-duplicate captures of the same scene,
and 14/15 `eval15` scenes have sibling captures inside `our485`. A random split would leak.
We therefore cluster images into scene groups (zero-mean-normalised grayscale thumbnail correlation
>= THRESH on the ground-truth images, connected components) over our485 + eval15, then:
  - drop every our485 scene group that contains an eval15 image          (-> `excluded_overlap`)
  - split the remaining groups train / val by GROUP with a fixed seed     (no scene in both)
  - test = eval15 (never used for training or checkpoint selection)
Writes data/splits.json plus a sha256 manifest for provenance.
Usage: python scripts/prepare_data.py --lol data/LOL --out data/splits.json
"""
import argparse, hashlib, json, os
import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg

THRESH, VAL_FRAC, SEED = 0.95, 0.15, 0


def thumb(p):
    x = cv2.resize(cv2.imread(p, 0), (48, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    x -= x.mean()
    return x / (np.linalg.norm(x) + 1e-8)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lol", default="data/LOL")
    ap.add_argument("--out", default="data/splits.json")
    a = ap.parse_args()
    tr = sorted(os.listdir(f"{a.lol}/our485/low")); te = sorted(os.listdir(f"{a.lol}/eval15/low"))
    T = np.stack([thumb(f"{a.lol}/our485/high/{n}") for n in tr] + [thumb(f"{a.lol}/eval15/high/{n}") for n in te])
    n_tr = len(tr)
    ncomp, lab = cg.connected_components(sp.csr_matrix((T @ T.T) >= THRESH), directed=False)
    test_groups = set(lab[n_tr:].tolist())
    keep_idx = [i for i in range(n_tr) if lab[i] not in test_groups]
    excl_idx = [i for i in range(n_tr) if lab[i] in test_groups]
    groups = sorted({int(lab[i]) for i in keep_idx})
    rng = np.random.default_rng(SEED); rng.shuffle(groups)
    n_val_g = max(1, int(round(VAL_FRAC * len(groups)))); val_g = set(groups[:n_val_g])
    train = [tr[i] for i in keep_idx if int(lab[i]) not in val_g]
    val = [tr[i] for i in keep_idx if int(lab[i]) in val_g]
    out = {
        "protocol": {"threshold": THRESH, "val_frac_of_groups": VAL_FRAC, "seed": SEED, "split_unit": "scene group"},
        "train": train, "val": val, "excluded_overlap": [tr[i] for i in excl_idx], "test": te,
        "counts": {"our485": n_tr, "train": len(train), "val": len(val), "excluded_overlap": len(excl_idx), "test": len(te)},
        "sha256_16": {f"our485/{n}": [sha(f"{a.lol}/our485/low/{n}"), sha(f"{a.lol}/our485/high/{n}")] for n in tr},
    }
    out["sha256_16"].update({f"eval15/{n}": [sha(f"{a.lol}/eval15/low/{n}"), sha(f"{a.lol}/eval15/high/{n}")] for n in te})
    # self-check: no cross-split near-duplicates
    idx = {n: i for i, n in enumerate(tr)}
    def cross(A, B):
        if not A or not B: return 0.0
        return float((np.stack([T[idx[x]] for x in A]) @ np.stack([T[idx[x]] for x in B]).T).max())
    out["max_thumb_corr"] = {"train_vs_val": cross(train, val), "train_vs_test": float((np.stack([T[idx[x]] for x in train]) @ T[n_tr:].T).max()),
                              "val_vs_test": float((np.stack([T[idx[x]] for x in val]) @ T[n_tr:].T).max())}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("counts", "max_thumb_corr")}, indent=1))


if __name__ == "__main__":
    main()
