# Adaptive Low-Light Restoration (classical + Zero-DCE++ hybrid)

```
lowlight/analysis.py    image statistics + rule-based policy -> all adaptive parameters
lowlight/classical.py   LIME-style guided illumination, Retinex, guided/bilateral denoise, CLAHE, shades-of-gray WB, saturation
lowlight/zerodce.py     Zero-DCE++ (PyTorch), optional 4th channel = classical illumination prior, non-reference + supervised losses
lowlight/fusion.py      illumination-driven blend map, quality-aware candidate selection, detail transfer, soft tone-map
lowlight/pipeline.py    modes: classical | dl | hybrid | auto, plus ablation switches (PipelineConfig)
lowlight/metrics.py     PSNR, SSIM, LPIPS, NIQE (optional), dE, colour cast, overexposure, noise gain, edge correlation, latency
train.py  evaluate.py  app.py
```

## Quick start
```bash
pip install -r requirements.txt
python app.py                                   # Gradio UI: compare modes, metrics, intermediate maps
python evaluate.py --low_dir data/LOL/eval15/low --high_dir data/LOL/eval15/high --ablation
python evaluate.py --low_dir data/DICM          # unseen / no-reference (also LIME, MEF, NPE, ExDark)
```

## IMPORTANT: the shipped weights are demo-only
`weights/*.pt` were trained for ~200 steps on 6 synthetically darkened sample photos, only so the app runs end to end.
They are **not** representative of model quality. Do real training before reporting any numbers:
```bash
# 1) unpaired, non-reference losses (Zero-DCE style)
python train.py --low_dir data/SICE_or_LOL/low --epochs 100 --bs 8 --crop 256
python train.py --low_dir data/SICE_or_LOL/low --epochs 100 --bs 8 --crop 256 --hybrid       # 4-ch with illumination prior
# 2) supervised fine-tune on paired LOL / LOL-v2 / SICE (adds L1 + SSIM terms)
python train.py --low_dir data/LOL/our485/low --high_dir data/LOL/our485/high --init weights/zerodce_rgb.pt --lr 5e-5
# 3) no low-light data? degrade normal-light photos synthetically (Poisson + read noise, vignette, colour cast)
python train.py --low_dir data/normal_photos --synthetic
```
Train on LOL only and test on DICM / LIME / MEF / NPE / ExDark to demonstrate generalisation to unseen images.

## How the adaptivity works
`analysis.analyze` computes mean/median luminance, dark fraction, skew, Immerkaer noise sigma, local contrast,
illumination non-uniformity and bright-pixel fraction. `choose_params` maps these to: Retinex gamma, denoise strength,
guided-filter eps, CLAHE clip/tile (gentle on noisy images), white-balance strength, saturation, number of curve
iterations for Zero-DCE++, blend threshold and detector scale for large images. `PipelineConfig(adaptive=False)` uses fixed constants (ablation).

Hybrid flow: denoise first -> classical branch -> Zero-DCE++ on the denoised image (+ illumination prior channel) ->
per-pixel blend (DL in dark regions, classical in lit regions) -> quality-aware soft selection over {classical, DL, blend}
-> detail transfer (damped by noise level) -> soft tone-map shoulder.

## Ablation (leave-one-out)
`evaluate.py --ablation` runs: full, no_adaptive, no_predenoise, no_illum_prior, no_blend_map, no_quality_select,
no_detail_transfer, no_tonemap. Smoke-test observation on ONE synthetic image (demo weights, not a result):
removing pre-denoising or adaptive parameters raised noise gain from 0.21 to 0.96 / 0.72 and cut SSIM sharply, while
detail transfer and tone-map had no measurable effect there (the image was noisy, so detail transfer is damped to ~0 by design, and no pixel reached the highlight knee).

## Failure analysis (to extend with your own examples from evaluate.py)
| Failure | Cause | Mitigation in this repo | Remaining risk |
|---|---|---|---|
| Noise amplification in extremely dark inputs | curves/Retinex multiply sensor noise | pre-denoise scaled by noise sigma, gentle CLAHE, damped detail transfer | very high ISO noise still leaves blotchy chroma noise or smoothed texture |
| Halos / blown highlights around lamps, windows | global brightening of already-lit pixels | blend map keeps classical output in lit regions; soft tone-map | DL-only mode can still clip (seen in street test) |
| Colour shift (cyan sky, tinted shadows) | white balance estimated from a scene with real dominant colour; sensor chroma noise | partial (not full) shades-of-gray; colour-cast term in quality selector | mixed colour temperatures remain ambiguous |
| Hallucinated texture / flat regions | network trained on little data | quality selector can fall back to classical; train on real paired data | untested on faces/text at extreme darkness |
| Motion blur, severe under-exposure with no signal | information is not in the image | none (enhancement cannot recover it) | out of scope |
