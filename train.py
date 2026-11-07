"""Train / fine-tune Zero-DCE++ (RGB or hybrid 4-channel).

Unpaired (non-reference losses only):   python train.py --low_dir data/low
Paired fine-tune (LOL / SICE):          python train.py --low_dir data/LOL/our485/low --high_dir data/LOL/our485/high
Synthetic degradation of normal images: python train.py --low_dir data/normal --synthetic
Add --hybrid to train the 4-channel model that takes the classical illumination map as prior.
Resume / fine-tune from existing weights with --init weights/zerodce_rgb.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowlight.data import LowLightDataset
from lowlight.zerodce import DCENet, ZeroDCELoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True)
    ap.add_argument("--high_dir")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--init")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 4 if a.hybrid else 3
    out = a.out or f"weights/zerodce_{'hybrid' if a.hybrid else 'rgb'}.pt"
    net = DCENet(in_ch=in_ch).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["state_dict"])
    ds = LowLightDataset(a.low_dir, a.high_dir, a.crop, with_prior=a.hybrid, synthetic=a.synthetic)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=len(ds) >= a.bs)
    crit, opt = ZeroDCELoss().to(dev), torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)

    for ep in range(a.epochs):
        tot = 0.0
        for b in dl:
            low = b["low"].to(dev)
            x = torch.cat([low, b["prior"].to(dev)], 1) if a.hybrid else low
            enh, A = net(x)
            loss, parts = crit(enh, low, A, b["gt"].to(dev) if "gt" in b else None)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            opt.step(); tot += loss.item()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f} " + " ".join(f"{k}={v.item():.4f}" for k, v in parts.items()))
        torch.save({"state_dict": net.state_dict(), "in_ch": in_ch, "nf": 32, "n_iter": 8}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
