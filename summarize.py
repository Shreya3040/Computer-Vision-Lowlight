"""Merge per_image.csv files of sharded evaluation runs and write summary.csv / summary.md.
Usage: python scripts/summarize.py results/baseline_v0/unseen_DICM results/baseline_v0/unseen_DICM_b ... --out results/baseline_v0/unseen_DICM_all"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True); a = ap.parse_args()
df = pd.concat([pd.read_csv(f"{d}/per_image.csv") for d in a.dirs]); os.makedirs(a.out, exist_ok=True)
df.to_csv(f"{a.out}/per_image.csv", index=False)
s = df.drop(columns=["image", "darkness"]).groupby("method").agg(["mean", "std"]).round(3)
s.columns = [f"{c}_{t}" if t == "std" else c for c, t in s.columns]; s.to_csv(f"{a.out}/summary.csv"); open(f"{a.out}/summary.md", "w").write(s.to_markdown())
print(f"{df.image.nunique()} images merged ->", a.out)
