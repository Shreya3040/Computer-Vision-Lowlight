"""End-to-end pipeline: classical | dl | hybrid | auto, with ablation switches."""
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np
from . import analysis, classical, fusion
from .metrics import Timer
from .utils import luminance, to_float

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Fixed parameters used when adaptive=False (ablation "no adaptive parameters").
FIXED_PARAMS = {"gamma": 0.45, "denoise_strength": 0.5, "denoise_eps": 0.01, "clahe_clip": 2.0,
                "clahe_tile": 8, "wb_strength": 0.5, "sat_boost": 1.1, "dce_iters": 8, "dce_scale": 1,
                "blend_threshold": 0.4, "noise_level": 0.3}


@dataclass
class PipelineConfig:
    adaptive: bool = True          # image-statistics-driven parameters
    predenoise: bool = True        # edge-preserving denoise before amplification
    illum_prior: bool = True       # feed classical illumination map to the network (4th channel)
    blend_map: bool = True         # illumination-driven spatial blend of DL and classical
    quality_select: bool = True    # no-reference quality-aware candidate weighting
    detail_transfer: bool = True   # edge-aware detail re-injection
    tonemap: bool = True           # highlight-safe soft shoulder
    max_side: int = 0              # optionally downscale very large inputs (0 = off)


@dataclass
class Result:
    output: np.ndarray
    mode: str
    stats: analysis.ImageStats
    params: dict
    intermediates: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    info: list = field(default_factory=list)


class LowLightEnhancer:
    def __init__(self, rgb_weights=None, hybrid_weights=None, device=None, config=None):
        self.config = config or PipelineConfig()
        self.device, self.models = device, {}
        self._load("rgb", rgb_weights or WEIGHTS_DIR / "zerodce_rgb.pt")
        self._load("hybrid", hybrid_weights or WEIGHTS_DIR / "zerodce_hybrid.pt")

    # ------------------------------------------------------------------ deep learning
    def _load(self, key, path):
        path = Path(path)
        if not path.exists():
            return
        import torch
        from .zerodce import DCENet
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=self.device)
        net = DCENet(in_ch=ck["in_ch"], nf=ck.get("nf", 32), n_iter=ck.get("n_iter", 8)).to(self.device).eval()
        net.load_state_dict(ck["state_dict"])
        self.models[key] = net

    def _run_dl(self, rgb, prior=None, iters=8, scale=1):
        import torch
        net = self.models["hybrid" if prior is not None else "rgb"]
        x = rgb if prior is None else np.concatenate([rgb, prior[..., None]], -1)
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out, _ = net(t, scale=scale, iters=iters)
        return np.clip(out[0].cpu().numpy().transpose(1, 2, 0), 0, 1)

    # ------------------------------------------------------------------ public API
    def enhance(self, img, mode="hybrid", config=None):
        """img: HxWx3 RGB uint8 or float [0,1]. mode in {classical, dl, hybrid, auto}."""
        cfg = config or self.config
        img = to_float(img) if img.dtype == np.uint8 else img.astype(np.float32)
        if cfg.max_side:
            from .utils import resize_max_side
            img = resize_max_side(img, cfg.max_side)
        st = analysis.analyze(img)
        params = analysis.choose_params(st) if cfg.adaptive else dict(FIXED_PARAMS)
        params["dce_scale"] = 1 if max(img.shape[:2]) <= 1280 else 2 if max(img.shape[:2]) <= 2560 else 4
        res = Result(output=img, mode=mode, stats=st, params=params)

        if mode == "auto":
            mode = "hybrid" if self.models else "classical"
            res.info.append(f"auto -> {mode}")
            res.mode = f"auto({mode})"
        if mode in ("dl", "hybrid") and not self.models:
            if mode == "dl":
                raise RuntimeError("No Zero-DCE++ weights found in weights/. Run train.py first.")
            res.info.append("WARNING: no DL weights found, hybrid fell back to classical only.")
            mode = "classical"

        with Timer() as t:
            if mode == "classical":
                res.output, res.intermediates = classical.classical_enhance(img, params, predenoise=True)
            elif mode == "dl":
                key_rgb = "rgb" if "rgb" in self.models else None
                if key_rgb is None:
                    raise RuntimeError("DL-only mode needs weights/zerodce_rgb.pt")
                res.output = self._run_dl(img, None, 8, params["dce_scale"])
            else:
                res.output, res.intermediates = self._hybrid(img, params, cfg, res.info)
        res.latency_ms = t.ms
        return res

    # ------------------------------------------------------------------ hybrid
    def _hybrid(self, img, p, cfg, info):
        clean = classical.denoise(img, p["denoise_strength"], p["denoise_eps"]) if cfg.predenoise else img
        C, inter = classical.classical_enhance(img, p, predenoise=cfg.predenoise)
        L = inter["illumination"]

        use_prior = cfg.illum_prior and "hybrid" in self.models
        if cfg.illum_prior and not use_prior:
            info.append("hybrid weights missing -> illumination prior disabled (using RGB model)")
        if use_prior:
            D = self._run_dl(clean, L, p["dce_iters"], p["dce_scale"])
        elif "rgb" in self.models:
            D = self._run_dl(clean, None, p["dce_iters"], p["dce_scale"])
        else:
            D = self._run_dl(clean, L, p["dce_iters"], p["dce_scale"])

        if cfg.blend_map:
            w = fusion.blend_weight_map(L, p["blend_threshold"], refine_guide=luminance(clean))
            B = w * D + (1 - w) * C
        else:
            w, B = None, 0.5 * (C + D)

        if cfg.quality_select:
            out, qw, scores = fusion.quality_select([C, D, B], clean, p["noise_level"])
            info.append(f"quality weights C/D/B = {np.round(qw, 2).tolist()}")
        else:
            out = B
        if cfg.detail_transfer:
            out = fusion.detail_transfer(out, clean, p["noise_level"])
        if cfg.tonemap:
            out = fusion.soft_tonemap(out)

        inter.update({"classical_out": C, "dl_out": D, "blend_out": B, "denoised_input": clean})
        if w is not None:
            inter["blend_weight_map"] = w[..., 0]
        return np.clip(out, 0, 1).astype(np.float32), inter


def ablation_configs(base=None):
    """Leave-one-out ablation configs for the hybrid pipeline."""
    base = base or PipelineConfig()
    names = ["adaptive", "predenoise", "illum_prior", "blend_map", "quality_select", "detail_transfer", "tonemap"]
    return {"full": base, **{f"no_{n}": replace(base, **{n: False}) for n in names}}
