"""Hybrid fusion: illumination-driven blend map, quality-aware candidate scoring,
edge-aware detail transfer and highlight-safe tone mapping."""
import numpy as np
from .utils import guided_filter, luminance
from . import metrics


def blend_weight_map(L, threshold, softness=0.08, refine_guide=None):
    """Per-pixel weight for the deep-learning output. Dark regions (low illumination) -> 1,
    already-lit regions -> 0, so lamps/windows are not blown out. Guided-filter smoothed so the
    weight map follows scene edges (no seams or halos)."""
    w = 1.0 / (1.0 + np.exp(-(threshold - L) / softness))
    guide = refine_guide if refine_guide is not None else L
    return np.clip(guided_filter(guide.astype(np.float32), w.astype(np.float32), r=12, eps=1e-3), 0, 1)[..., None]


def quality_score(img, inp, noise_level):
    """Lower = better. Penalises poor exposure, new clipping, colour cast, noise amplification,
    and rewards contrast/entropy (so the selector does not just pick the darkest candidate)."""
    mean = luminance(img).mean()
    pen = (
        2.0 * abs(mean - 0.42)
        + 6.0 * max(0.0, metrics.overexposed_frac(img) - metrics.overexposed_frac(inp))
        + 0.015 * max(0.0, metrics.colour_cast(img) - metrics.colour_cast(inp))
        + (0.3 + 1.0 * noise_level) * min(2.0, metrics.relative_noise(img) / (metrics.relative_noise(inp) + 1e-6)) * 0.3
        - 0.05 * metrics.entropy(img)
        - 0.5 * min(1.0, luminance(img).std() / 0.25)
    )
    return float(pen)


def quality_select(cands, inp, noise_level, temperature=0.15):
    """Soft selection over candidates by quality score. Returns fused image, weights, scores."""
    scores = np.array([quality_score(c, inp, noise_level) for c in cands])
    w = np.exp(-(scores - scores.min()) / temperature)
    w /= w.sum()
    return sum(wi * c for wi, c in zip(w, cands)), w, scores


def detail_transfer(out, clean_in, noise_level, max_gain=6.0):
    """Re-inject fine detail (guided-filter detail layer of the *denoised* input) scaled by the
    local amplification, damped by the noise level to avoid noise amplification."""
    li, lo = luminance(clean_in), luminance(out)
    detail = li - guided_filter(li, li, r=6, eps=1e-3)
    gain = np.clip(lo / (li + 1e-3), 1.0, max_gain)
    amp = 0.6 * (1.0 - noise_level)
    return np.clip(out + (amp * detail * gain)[..., None] * 0.5, 0, 1)


def soft_tonemap(img, knee=0.88):
    """Hue-preserving soft shoulder on the max channel: avoids hard clipping near lamps."""
    m = img.max(-1, keepdims=True)
    comp = knee + (1 - knee) * np.tanh((m - knee) / (1 - knee))
    scale = np.where(m > knee, comp / (m + 1e-6), 1.0)
    return np.clip(img * scale, 0, 1)
