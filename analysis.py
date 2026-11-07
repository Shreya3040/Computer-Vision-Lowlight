"""Image analysis: feature vector + rule-based policy that drives every adaptive choice."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from scipy.stats import skew
from .utils import luminance


def estimate_noise_sigma(img):
    """Immerkaer fast noise-variance estimate on luminance (returns sigma in [0,1] units)."""
    g = luminance(img) * 255.0
    h, w = g.shape
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(g.astype(np.float32), -1, k)[1:-1, 1:-1]
    sigma = np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2))
    return float(sigma / 255.0)


@dataclass
class ImageStats:
    mean_lum: float
    median_lum: float
    dark_frac: float          # fraction of pixels with V < 0.1
    skewness: float
    noise_sigma: float
    local_contrast: float
    nonuniformity: float      # spread of a heavily blurred luminance map, relative to its mean
    bright_frac: float        # already-bright pixels (lamps, windows): overexposure risk
    # derived
    darkness_level: str = ""

    def as_dict(self):
        return asdict(self)


def analyze(img):
    lum = luminance(img)
    v = img.max(-1)
    small = cv2.resize(lum, (min(256, lum.shape[1]), min(256, lum.shape[0])), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), max(small.shape) / 16)
    mean = float(lum.mean())
    st = ImageStats(
        mean_lum=mean,
        median_lum=float(np.median(lum)),
        dark_frac=float((v < 0.1).mean()),
        skewness=float(skew(small.ravel())),
        noise_sigma=estimate_noise_sigma(img),
        local_contrast=float(cv2.GaussianBlur((small - blur) ** 2, (0, 0), 3).mean() ** 0.5),
        nonuniformity=float(blur.std() / (blur.mean() + 1e-6)),
        bright_frac=float((v > 0.85).mean()),
    )
    st.darkness_level = (
        "extreme" if st.mean_lum < 0.04 else "dark" if st.mean_lum < 0.12
        else "dim" if st.mean_lum < 0.28 else "normal"
    )
    return st


def choose_params(st: ImageStats):
    """Rule-based policy: image statistics -> enhancement parameters (replaces fixed constants)."""
    # Target brightness: lift darker images more, but never to full mid-gray when lamps are present.
    target = np.clip(0.42 - 0.2 * st.bright_frac * 10, 0.30, 0.45)
    gamma = float(np.clip(np.log(target) / np.log(max(st.median_lum, 0.01) + 0.05), 0.30, 1.0))
    noisy = float(np.clip((st.noise_sigma - 0.004) / 0.03, 0, 1))        # 0 clean .. 1 very noisy
    return {
        "gamma": gamma,
        "denoise_strength": 0.25 + 0.75 * noisy,                           # blend toward denoised input
        "denoise_eps": 1e-3 + 0.02 * noisy,                                # guided-filter eps
        "clahe_clip": float(np.clip(3.0 - 2.2 * noisy, 0.8, 3.0)),         # gentle for noisy images
        "clahe_tile": 8 if st.local_contrast > 0.05 else 4,
        "wb_strength": 0.6 if st.darkness_level in ("extreme", "dark") else 0.3,
        "sat_boost": 1.0 + 0.25 * (1 - noisy),
        "dce_iters": {"extreme": 8, "dark": 8, "dim": 5, "normal": 3}[st.darkness_level],
        "dce_scale": 1,                                                    # overridden by image size
        "blend_threshold": float(np.clip(0.25 + 0.4 * st.nonuniformity, 0.25, 0.6)),
        "noise_level": noisy,
    }
