# Audit and frozen baseline v0 (before any improvement)

Code: original LUMEN-X package + **demo weights** (about 200 steps on 6 synthetic images; weights sha {'rgb': '9b03537bad73', 'hybrid': '7041c932f8b8'}).
Hardware: cpu, 1 CPU thread, torch 2.15.0+cu132. All numbers below are produced by `evaluate.py` (`results/baseline_v0/*/per_image.csv`).
LPIPS: **unavailable** in this environment (AlexNet backbone host blocked). NIQE: pyiqa, lower is better (GT images of eval15 average 4.63).

## Data protocol (leakage-safe)
LOL `our485` holds many near-duplicate captures of the same scene, and 14 of the 15 `eval15` scenes have sibling captures inside `our485`
(for example eval15/22 vs our485/21). A random or official split would therefore leak test scenes into training.
`scripts/prepare_data.py` clusters scenes (thumbnail correlation >= 0.95), drops every `our485` scene linked to `eval15`, and splits the rest by scene group.
Counts: {'our485': 485, 'train': 395, 'val': 66, 'excluded_overlap': 24, 'test': 15}. Max cross-split thumbnail correlation (threshold 0.95): train_vs_val 0.85, train_vs_test 0.76, val_vs_test 0.64.
Unseen (no ground truth): DICM (64), LIME (10), ExDark 100-image deterministic subset of its test split. Sources are third-party GitHub mirrors (see `scripts/download_data.sh`).

## LOL eval15 (n=15, full 600x400)
| method | PSNR | SSIM | dE_vs_ref | NIQE | new_overexposed_% | noise_gain | latency_ms |
|---|---|---|---|---|---|---|---|
| input | 7.77 ± 2.62 | 0.19 ± 0.10 | 47.20 ± 9.42 | 6.75 ± 1.11 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0 |
| classical | 14.52 ± 1.46 | 0.40 ± 0.08 | 20.59 ± 3.67 | 8.67 ± 1.13 | 0.50 ± 0.72 | 1.56 ± 0.26 | 56 |
| dl_only | 16.90 ± 3.50 | 0.56 ± 0.11 | 22.62 ± 6.54 | 8.00 ± 1.14 | 0.67 ± 2.29 | 0.81 ± 0.14 | 802 |
| hybrid | 15.78 ± 2.49 | 0.45 ± 0.11 | 20.23 ± 4.93 | 8.47 ± 1.11 | 0.00 ± 0.00 | 1.33 ± 0.21 | 957 |

## Unseen sets (no-reference; images resized to max side 720)
### DICM (n=64)
| method | NIQE | mean_lum | new_overexposed_% | noise_gain | edge_corr | latency_ms |
|---|---|---|---|---|---|---|
| input | 4.45 ± 1.38 | 0.33 ± 0.19 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0 |
| classical | 3.98 ± 1.27 | 0.42 ± 0.10 | 0.58 ± 0.98 | 2.60 ± 1.22 | 0.79 ± 0.15 | 76 |
| dl_only | 4.34 ± 1.56 | 0.70 ± 0.24 | 31.53 ± 22.61 | 0.83 ± 0.45 | 0.53 ± 0.13 | 1092 |
| hybrid | 3.92 ± 1.28 | 0.44 ± 0.11 | 0.00 ± 0.00 | 2.19 ± 1.10 | 0.80 ± 0.13 | 1377 |
### LIME (n=10)
| method | NIQE | mean_lum | new_overexposed_% | noise_gain | edge_corr | latency_ms |
|---|---|---|---|---|---|---|
| input | 4.41 ± 1.57 | 0.14 ± 0.07 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0 |
| classical | 4.43 ± 2.68 | 0.38 ± 0.04 | 0.85 ± 0.87 | 1.65 ± 0.56 | 0.70 ± 0.17 | 64 |
| dl_only | 4.41 ± 2.45 | 0.56 ± 0.14 | 10.11 ± 12.45 | 0.87 ± 0.30 | 0.61 ± 0.08 | 888 |
| hybrid | 4.03 ± 1.86 | 0.40 ± 0.06 | 0.00 ± 0.00 | 1.22 ± 0.42 | 0.72 ± 0.16 | 1104 |
### ExDark100 (n=100)
| method | NIQE | mean_lum | new_overexposed_% | noise_gain | edge_corr | latency_ms |
|---|---|---|---|---|---|---|
| input | 4.71 ± 1.72 | 0.14 ± 0.09 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0 |
| classical | 3.41 ± 1.18 | 0.35 ± 0.06 | 0.55 ± 0.79 | 1.64 ± 0.57 | 0.64 ± 0.20 | 77 |
| dl_only | 3.97 ± 1.74 | 0.50 ± 0.18 | 8.06 ± 10.56 | 0.78 ± 0.29 | 0.55 ± 0.12 | 1085 |
| hybrid | 3.49 ± 1.29 | 0.38 ± 0.07 | 0.00 ± 0.00 | 1.15 ± 0.39 | 0.67 ± 0.19 | 1377 |

## Audit findings that drive the plan
1. Hybrid is not the best mode on LOL PSNR/SSIM (DL-only is), but it has the lowest colour distance and no new clipping on every set.
2. Demo DL-only clips heavily on unseen images (see `new_overexposed_%`); hybrid hides this via blending and tone-mapping.
3. Every method scores a worse NIQE than the dark input on LOL (all > 6.75 vs GT 4.63): enhancement currently amplifies noise/artifacts. NIQE also tends to favour dark, smooth images, so it is read together with the other metrics.
4. Cyan cast: the input already has B/R = 1.15; global white balance is a no-op (gains about 1.0), Retinex and the uniform saturation boost amplify the sky chroma, and DL adds a cast on neutral brick (b* -0.4 -> -11.3).
5. The quality selector has Spearman 0.00 with PSNR on eval15 and picks the PSNR-best candidate in 4/15 images.
6. Adaptivity is nearly dormant on LOL: dce_iters is 8 everywhere and the absolute noise estimate underestimates noise in very dark images.
