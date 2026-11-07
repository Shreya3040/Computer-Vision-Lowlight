"""Comparative + ablation evaluation.

Paired set (PSNR/SSIM/LPIPS):  python evaluate.py --low_dir data/LOL/eval15/low --high_dir data/LOL/eval15/high
Unseen/unpaired (no-ref):      python evaluate.py --low_dir data/DICM
Add --ablation to run leave-one-out ablations of the hybrid pipeline.
Writes results/*.csv and results/summary.md plus side-by-side images.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from lowlight import LowLightEnhancer
from lowlight.data import list_images, read_rgb
from lowlight.metrics import evaluate_pair
from lowlight.pipeline import ablation_configs
from lowlight.utils import to_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True)
    ap.add_argument("--high_dir")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--max_side", type=int, default=720, help="resize for speed (0 = full res)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip first N files (to shard long runs)")
    ap.add_argument("--split_json", help="splits.json from scripts/prepare_data.py")
    ap.add_argument("--split", default="test", help="key in splits.json (train|val|test)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    enh = LowLightEnhancer()
    gts = {p.name: p for p in list_images(a.high_dir)} if a.high_dir else {}
    files = list_images(a.low_dir)
    if a.split_json:
        import json
        keep = set(json.load(open(a.split_json))[a.split])
        files = [f for f in files if f.name in keep]
    files = files[a.offset: a.offset + a.limit if a.limit else None]

    variants = {"classical": ("classical", None), "dl_only": ("dl", None), "hybrid": ("hybrid", None)}
    if a.ablation:
        for k, c in ablation_configs().items():
            variants[f"abl_{k}"] = ("hybrid", c)
    rows = []
    for f in files:
        img = read_rgb(f)
        ref = read_rgb(gts[f.name]) if f.name in gts else None
        if a.max_side:
            s = a.max_side / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                ref = cv2.resize(ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA) if ref is not None else None
        panels = [img]
        rows.append({"image": f.name, "method": "input", "latency_ms": 0.0, "darkness": "", **evaluate_pair(img, img, ref)})
        for name, (mode, cfg) in variants.items():
            try:
                r = enh.enhance(img, mode, cfg)
            except RuntimeError as e:
                print("skip", name, e); continue
            m = evaluate_pair(img, r.output, ref)
            rows.append({"image": f.name, "method": name, "latency_ms": r.latency_ms, "darkness": r.stats.darkness_level, **m})
            if not name.startswith("abl_"):
                panels.append(r.output)
        if ref is not None:
            panels.append(ref)
        cv2.imwrite(str(out / f"{f.stem}_compare.jpg"), cv2.cvtColor(to_u8(np.concatenate(panels, 1)), cv2.COLOR_RGB2BGR))
        print("done", f.name)

    import json, platform, torch, hashlib, time
    wh = {k: hashlib.sha256(open(p, "rb").read()).hexdigest()[:12] for k, p in
          {"rgb": "weights/zerodce_rgb.pt", "hybrid": "weights/zerodce_hybrid.pt"}.items() if os.path.exists(p)}
    (out / "run_meta.json").write_text(json.dumps({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "n_images": len(files), "device": enh.device or "cpu",
        "torch": torch.__version__, "python": platform.python_version(), "weights_sha": wh,
        "lpips": "unavailable (AlexNet backbone host blocked)", "niqe": "pyiqa (lower=better)"}, indent=1))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    summ = df.drop(columns=["image", "darkness"]).groupby("method").agg(["mean", "std"]).round(3)
    summ.columns = [f"{a}_{b}" if b == "std" else a for a, b in summ.columns]
    summ.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(summ.to_markdown())
    print(summ.to_string())


if __name__ == "__main__":
    main()
