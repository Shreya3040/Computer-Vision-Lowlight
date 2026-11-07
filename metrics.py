"""Evaluation metrics: full-reference (PSNR/SSIM/LPIPS/dE), no-reference (NIQE via pyiqa when
installed + our own exposure / colour-cast / noise / edge measures) and latency."""
import time
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .analysis import estimate_noise_sigma
from .utils import luminance, to_u8

_lpips, _niqe = {}, {}


def psnr(a, b):
    return float(peak_signal_noise_ratio(b, a, data_range=1.0))


def ssim(a, b):
    return float(structural_similarity(b, a, data_range=1.0, channel_axis=2))


def lpips_score(a, b):
    """LPIPS (AlexNet). Needs `pip install lpips` + pretrained weights; NaN if unavailable."""
    try:
        import torch, lpips
        if "m" not in _lpips:
            _lpips["m"] = lpips.LPIPS(net="alex", verbose=False).eval()
        t = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            return float(_lpips["m"](t(a), t(b)))
    except Exception:
        return float("nan")


def niqe_score(img):
    """NIQE via pyiqa (lower = better). NaN if pyiqa is not installed."""
    try:
        import torch, pyiqa
        if "m" not in _niqe:
            _niqe["m"] = pyiqa.create_metric("niqe")
        return float(_niqe["m"](torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()))
    except Exception:
        return float("nan")


def delta_e(a, b):
    la = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2LAB)
    lb = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.sqrt(((la - lb) ** 2).sum(-1)).mean())


def colour_cast(img):
    """Distance of mean (a,b) from neutral in Lab. Large => strong global colour cast."""
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.hypot(lab[..., 1].mean(), lab[..., 2].mean()))


def overexposed_frac(img, thr=0.98):
    return float((img.max(-1) >= thr).mean())


def relative_noise(img):
    """Noise sigma relative to mean brightness (so brightening alone does not look 'noisier')."""
    return float(estimate_noise_sigma(img) / (luminance(img).mean() + 1e-3))


def edge_preservation(inp, out):
    """Pearson correlation between gradient magnitudes of input and output luminance."""
    def gm(x):
        g = luminance(x)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()
    a, b = gm(inp), gm(out)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else 0.0


def entropy(img):
    h = np.bincount(to_u8(luminance(img)).ravel(), minlength=256).astype(np.float64)
    p = h / h.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def evaluate_pair(inp, out, ref=None):
    m = {
        "mean_lum": float(luminance(out).mean()),
        "overexposed_%": 100 * overexposed_frac(out),
        "new_overexposed_%": 100 * max(0.0, overexposed_frac(out) - overexposed_frac(inp)),
        "colour_cast": colour_cast(out),
        "rel_noise": relative_noise(out),
        "noise_gain": relative_noise(out) / (relative_noise(inp) + 1e-6),
        "edge_corr": edge_preservation(inp, out),
        "entropy": entropy(out),
        "NIQE": niqe_score(out),
    }
    if ref is not None:
        m.update({"PSNR": psnr(out, ref), "SSIM": ssim(out, ref), "LPIPS": lpips_score(out, ref),
                  "dE_vs_ref": delta_e(out, ref)})
    return m


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t) * 1000
