"""Classical branch: illumination estimation, Retinex, denoising, CLAHE, colour correction."""
import cv2
import numpy as np
from .utils import guided_filter, guided_filter_rgb, luminance, to_u8, to_float


def illumination_estimate(img, r=15, eps=1e-2):
    """LIME-style: max-RGB initial map refined by a guided filter (edge-aware, handles
    non-uniform lighting far better than a Gaussian blur)."""
    init = img.max(-1)
    L = guided_filter(luminance(img), init, r=r, eps=eps)
    return np.clip(L, 1e-3, 1.0).astype(np.float32)


def denoise(img, strength, eps):
    """Edge-preserving denoise BEFORE amplification. Guided filter on luminance structure
    + bilateral filter on a small-radius copy; blended by `strength`."""
    gf = guided_filter_rgb(img, r=4, eps=eps * 0.1)
    bl = cv2.bilateralFilter(to_u8(img), d=5, sigmaColor=25 + 40 * strength, sigmaSpace=5)
    den = 0.5 * gf + 0.5 * to_float(bl)
    return (1 - strength) * img + strength * den


def retinex(img, L, gamma):
    """Single-scale Retinex with illumination adjustment: R = I / L ; out = R * L^gamma."""
    L3 = L[..., None]
    R = np.clip(img / (L3 + 1e-3), 0, 1.5)
    adj = np.power(L3, gamma)
    return np.clip(R * adj, 0, 1), R


def msrcr(img, sigmas=(15, 80, 250), gain=1.0):
    """Multi-scale Retinex with colour restoration (baseline / fallback)."""
    x = img * 255.0 + 1.0
    out = np.zeros_like(x)
    for s in sigmas:
        out += np.log(x) - np.log(cv2.GaussianBlur(x, (0, 0), s) + 1.0)
    out /= len(sigmas)
    crf = np.log(125.0 * x / (x.sum(-1, keepdims=True) + 3.0))
    out = out * crf * gain
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def clahe_lab(img, clip, tile):
    lab = cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    lab[..., 0] = cl.apply(lab[..., 0])
    return to_float(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def shades_of_gray(img, p=6, strength=0.5):
    """Shades-of-gray white balance, applied partially to avoid over-correcting scenes whose
    dominant colour is real (e.g. a red neon sign)."""
    est = np.power(np.mean(np.power(img, p), axis=(0, 1)) + 1e-8, 1 / p)
    gain = est.mean() / (est + 1e-6)
    gain = 1 + strength * (gain - 1)
    return np.clip(img * gain, 0, 1)


def saturation_adjust(img, boost):
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * boost, 0, 1)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)


def classical_enhance(img, params, predenoise=True):
    """Full classical pipeline. Returns final image and intermediates for visualisation."""
    work = denoise(img, params["denoise_strength"], params["denoise_eps"]) if predenoise else img
    L = illumination_estimate(work)
    ret, R = retinex(work, L, params["gamma"])
    out = clahe_lab(ret, params["clahe_clip"], params["clahe_tile"])
    out = shades_of_gray(out, strength=params["wb_strength"])
    out = saturation_adjust(out, params["sat_boost"])
    return out.astype(np.float32), {
        "illumination": L, "reflectance": np.clip(R, 0, 1), "denoised": work, "retinex": ret,
    }
