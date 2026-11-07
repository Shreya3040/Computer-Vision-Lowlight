**LUMEN-X**

Adaptive Low-Light Image Restoration with a Classical + Zero-DCE++ Hybrid Pipeline

*LUMEN-X Hackathon Technical Solution Report (Revised Edition)*

Team / institution: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ (to be completed by the team)

**Abstract**

Images captured in low light suffer from low brightness, amplified sensor noise, poor local contrast and unreliable colour. LUMEN-X addresses these degradations with a hybrid pipeline that combines a classical branch (guided-filter illumination estimation, Retinex, CLAHE and partial white balance) with a lightweight Zero-DCE++ curve-estimation network. The two branches are combined by an illumination-driven spatial blend, a heuristic no-reference quality-aware weighting of candidate outputs, detail transfer and a highlight-safe tone-mapping stage. All adaptive parameters are derived from image statistics through rule-based heuristics. On the 15-image LOL evaluation set, the deep-learning branch attains the best PSNR (16.90 dB), SSIM (0.56) and NIQE (8.00), whereas the hybrid pipeline attains the lowest colour difference (20.23) and no measured new overexposure; the classical branch is the fastest (56 ms). On three unseen datasets (DICM, LIME, ExDark) the hybrid pipeline produces no measured new overexposure. All results were obtained with demo-trained weights and therefore constitute a baseline evaluation, not final trained-model performance.

**Revision note.** This edition replaces the pending benchmark table with measured results, adds an unseen-dataset evaluation, corrects the technical description to match the source code, expands the failure analysis, adds visual evidence taken from the project results directory, and separates implemented functionality from pending and future work (Table 3). Every number in this report is traceable to results/baseline_v0/ or docs/AUDIT_AND_BASELINE_V0.md in the project archive.

# 1. Problem Understanding and Solution Strategy

## 1.1 Problem Statement

Low-light photographs exhibit a coupled set of degradations: (i) low global and local luminance, (ii) a low signal-to-noise ratio because the sensor signal is small relative to shot and read noise, (iii) compressed contrast and loss of detail in shadows, and (iv) colour casts arising from the mixture of artificial light sources. A naive brightness increase amplifies noise together with the signal, clips already-lit regions such as lamps and windows, and may shift hue. The task is therefore not merely to brighten an image but to do so adaptively, without introducing new artefacts.

## 1.2 Design Objectives

- Increase visibility in dark regions while preserving already-lit regions (no new clipping).

- Limit noise amplification and colour distortion.

- Adapt processing to each image rather than rely on fixed constants.

- Remain lightweight and reproducible: a single CPU-capable package, a documented data protocol and an interactive demonstration.

## 1.3 Solution Strategy

Classical methods are fast and interpretable but sensitive to noise and colour assumptions, whereas learned curve-estimation methods adapt well to scene content but may over-brighten and are only as reliable as their training data. LUMEN-X combines the two: the classical branch supplies an illumination map and a conservative enhanced image, the network supplies a data-driven enhancement, and a spatial blend driven by the illumination map assigns each pixel mainly to the branch that is more appropriate for its brightness. A no-reference quality score then weights the candidate outputs, and post-processing controls detail and highlights.

**Scope of the claims.** The current implementation performs *illumination-driven spatial fusion*; it does not perform semantic segmentation or object-level region understanding. Parameter adaptation is *heuristic and rule-based*, and there is no iterative self-correcting feedback loop. Features of this kind are listed as future work in Section 2.9.

# 2. Technical Architecture and Implementation

## 2.1 System Overview

The package lowlight/ contains the processing modules; app.py, train.py and evaluate.py provide the interface, training and evaluation entry points. Table 1 summarises the modules. The hybrid data flow is: image statistics and adaptive parameters → edge-preserving pre-denoising → classical branch (candidate *C*) → Zero-DCE++ on the denoised image with an illumination-prior channel (candidate *D*) → illumination-driven blend (candidate *B*) → quality-aware weighting over {*C*, *D*, *B*} → detail transfer → soft tone mapping.

**Table 1.** Software modules of the LUMEN-X package.

| **Module**                    | **Responsibility**                                                                                                              |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| analysis.py                   | Image statistics (luminance, noise, non-uniformity, bright fraction) and the rule-based parameter policy                        |
| classical.py                  | Guided-filter illumination estimate, Retinex, denoising, CLAHE, shades-of-gray white balance, saturation                        |
| zerodce.py                    | Zero-DCE++ network with optional fourth (illumination-prior) input channel; non-reference and supervised losses                 |
| fusion.py                     | Blend-weight map, quality score and soft candidate weighting, detail transfer, soft tone mapping                                |
| pipeline.py                   | Modes classical, dl, hybrid, auto; PipelineConfig switches for ablation                                                         |
| metrics.py                    | PSNR, SSIM, NIQE (pyiqa), LPIPS (optional), colour difference, colour cast, overexposure, noise gain, edge correlation, latency |
| train.py, evaluate.py, app.py | Training, comparative/ablation evaluation, Gradio demonstration                                                                 |

## 2.2 Image Analysis and Heuristic Adaptive Parameter Selection

For each input, analyze computes the mean and median luminance, the fraction of dark pixels (max-channel below 0.1), skewness, the Immerkaer noise estimate *σ*ₙ, local contrast, illumination non-uniformity *u* (coefficient of variation of a heavily blurred luminance map) and the fraction *b* of already-bright pixels (max-channel above 0.85). The image is labelled *extreme*, *dark*, *dim* or *normal* by thresholds of 0.04, 0.12 and 0.28 on mean luminance. The function choose_params then maps these statistics to processing parameters by fixed hand-written rules; no parameter is learned. With *m* the median luminance and *τ* a target brightness,

τ = clip(0.42 − 2b, 0.30, 0.45), γ = clip( ln τ / ln(m + 0.05), 0.30, 1.0 ) (1)

ν = clip((σₙ − 0.004)/0.03, 0, 1) (2)

where *ν* ∈ \[0, 1\] is a normalised noise level. Table 2 lists the resulting rules.

**Table 2.** Heuristic rules mapping image statistics to parameters (from analysis.choose_params).

| **Parameter**          | **Rule**                              | **Purpose**                                    |
|------------------------|---------------------------------------|------------------------------------------------|
| Retinex gamma γ        | Equation (1); lower for darker images | Brightness lift limited when lamps are present |
| Denoise strength       | 0.25 + 0.75 ν                         | Stronger smoothing for noisier input           |
| CLAHE clip limit       | clip(3.0 − 2.2 ν, 0.8, 3.0)           | Gentler equalisation on noisy images           |
| CLAHE tile             | 8 if local contrast \> 0.05, else 4   | Tile size by contrast                          |
| White-balance strength | 0.6 (extreme/dark), 0.3 otherwise     | Partial correction only                        |
| Saturation boost       | 1 + 0.25 (1 − ν)                      | Less boost for noisy images                    |
| Curve iterations       | 8 (extreme/dark), 5 (dim), 3 (normal) | Strength of network enhancement (hybrid mode)  |
| Blend threshold θ      | clip(0.25 + 0.4 u, 0.25, 0.6)         | Illumination level separating the two branches |

Setting PipelineConfig(adaptive=False) replaces all of these by fixed constants, which is the basis of the planned "no adaptive parameters" ablation. The rules are heuristics designed by inspection; they have not yet been tuned or validated against a labelled objective. In particular, the audit recorded that the number of curve iterations equals 8 for every LOL test image and that the absolute noise estimate under-estimates noise in very dark images, so adaptivity is only weakly exercised on that dataset.

## 2.3 Classical Branch

After optional edge-preserving denoising (Section 2.2), the classical branch estimates illumination in the manner of LIME: the max-RGB map is refined by a guided filter \[He et al., 2013\] with the luminance image *Y* as guide,

L = clip( G\[r=15, ε=10⁻²\]( Y ; max(R, G, B) ), 10⁻³, 1 ) (3)

followed by a single-scale Retinex decomposition and illumination adjustment,

R = clip( I / (L + 10⁻³), 0, 1.5 ), J = clip( R · L^γ, 0, 1 ) (4)

Contrast-limited adaptive histogram equalisation (CLAHE) is applied to the L channel in CIELAB, followed by partial shades-of-gray white balance with gain *g* = 1 + *s*(ĝ − 1), where ĝ is the full correction and *s* the white-balance strength, and a saturation adjustment in HSV. The denoiser blends the input with the average of a guided-filter result and a bilateral-filter result according to the denoise strength *δ*: Î = (1 − *δ*)·I + *δ*·(½·GF(I) + ½·BF(I)). A multi-scale Retinex with colour restoration (msrcr) is also provided as a baseline function but is not used in the pipeline.

## 2.4 Deep-Learning Branch (Zero-DCE++)

The network DCENet is a seven-layer depthwise-separable convolutional network with skip connections that predicts *N* = 8 sets of per-pixel, per-channel curve parameters *A*ₙ (24 output channels, tanh activation). The enhanced image is obtained by iterating a quadratic curve,

Eₙ = Eₙ₋₁ + Aₙ ⊙ (Eₙ₋₁² − Eₙ₋₁), E₀ = I, n = 1, …, N (5)

For large images the curve maps are estimated on a down-scaled copy and up-sampled (Zero-DCE++ acceleration). In hybrid mode, a fourth input channel carries the classical illumination map *L* as a prior. Training uses the non-reference losses of Zero-DCE (spatial consistency, exposure with target 0.6, colour constancy and illumination smoothness) and, optionally, supervised L1 and SSIM terms for fine-tuning on paired data. **The shipped checkpoints** (zerodce_rgb.pt, zerodce_hybrid.pt) **are demo-only**: they were trained for approximately 200 steps on six synthetically darkened sample photographs solely so that the application runs end to end.

## 2.5 Illumination-Driven Spatial Fusion

The hybrid candidates are combined by a per-pixel weight that depends only on the estimated illumination, not on any semantic label or object category. A sigmoid maps low illumination to a weight close to 1 (use the network output *D*) and high illumination to a weight close to 0 (keep the classical output *C*), and the map is then smoothed by a guided filter whose guide is the luminance *Y*′ of the denoised input, so that the weight follows scene edges and avoids seams:

w = G( σ((θ − L)/s) ; Y′ ), s = 0.08, B = w ⊙ D + (1 − w) ⊙ C (6)

Figure 5 (panel "DL blend weight w") shows this map for a night-street image: the weight is close to one over most of the dark scene and falls towards zero at the street lamp and the illuminated windows, so that these regions are taken from the classical output. This is a statement about illumination level only; the method has no notion of which objects are present.

## 2.6 Quality-Aware Candidate Selection

The three candidates {*C*, *D*, *B*} are scored by a hand-designed, no-reference quality function *Q* (lower is better) that penalises deviation from a mean luminance of 0.42, newly introduced clipping, increased colour cast and increased relative noise, and rewards luminance entropy and contrast. Writing *μ* for mean luminance, *o* for the fraction of pixels with max-channel ≥ 0.98, *κ* for the Lab colour-cast measure, *ρ* for noise relative to brightness, *H* for luminance entropy, *σ*(*Y*) for the luminance standard deviation and *r* = min(2, *ρ*(*E*)/*ρ*(*I*)) for the capped noise ratio, the score is

Q = 2\|μ − 0.42\| + 6Δo + 0.015Δκ + 0.3(0.3 + ν)·r − 0.05H − 0.5·min(1, σ(Y)/0.25) (7)

with Δo and Δκ the positive increases relative to the denoised input. The scores are converted to *image-level* weights by a soft-min with temperature *T* = 0.15, and the output is the weighted sum of the candidates:

ωₖ = exp(−(Qₖ − Qₘᵢₙ)/T) / Σⱼ exp(−(Qⱼ − Qₘᵢₙ)/T), E = Σₖ ωₖ Eₖ, k ∈ {C, D, B} (8)

This mechanism is therefore a single-pass, soft weighting of three whole-image candidates by a heuristic score. It is *not* an iterative loop: the output is not re-evaluated or re-processed, and no feedback adjusts earlier stages. The constants were set by design and not fitted. The audit reported that the rank correlation between *Q* and PSNR on the LOL test set is 0.00 and that the candidate with the best PSNR is the highest-weighted in 4 of 15 images; *Q* should therefore be regarded as an unvalidated heuristic rather than a calibrated quality predictor.

## 2.7 Detail Transfer and Soft Tone Mapping

Fine detail of the denoised input (its guided-filter detail layer *d* = *Y*′ − G(*Y*′)) is re-injected into the fused image, scaled by the local amplification *g* = clip(*Y*(*E*) / *Y*′, 1, 6) and damped by the noise level so that noisy images receive little added detail:

E′ = clip( E + 0.5 · a · g · d ), a = 0.6 (1 − ν) (9)

Finally a hue-preserving soft shoulder is applied to the maximum channel *m* of *E*′ with knee *k* = 0.88, avoiding hard clipping near lamps:

m′ = k + (1 − k)·tanh((m − k)/(1 − k)) for m \> k; Eₒᵤₜ = E′ · m′/m (10)

## 2.8 Operating Modes, Interface, Training Data and Evaluation Protocol

The pipeline offers classical, dl, hybrid and auto modes (auto selects hybrid when weights are available and classical otherwise). The Gradio application displays the input and the three outputs, no-reference metrics with latency, the adaptive parameters, and intermediate maps. Note that the dl mode applies the network with a fixed eight curve iterations and the RGB checkpoint; the adaptive rules affect only the classical and hybrid modes.

**Leakage-safe data protocol.** The LOL our485 partition contains many near-duplicate captures of the same scene, and 14 of the 15 eval15 scenes have sibling captures inside our485. scripts/prepare_data.py clusters scenes by thumbnail correlation (threshold 0.95), removes every our485 scene linked to eval15, and splits the remainder by scene group. The resulting counts are 395 training, 66 validation, 24 excluded-overlap and 15 test images, with a maximum cross-split correlation of 0.85 (train–validation), 0.76 (train–test) and 0.64 (validation–test). The split is recorded in configs/splits_reference.json and can be verified with scripts/verify_data.py.

## 2.9 Implementation Status

Table 3 separates what is implemented, what has been experimentally evaluated, what is pending and what is proposed. Nothing listed under C or D should be read as completed.

**Table 3.** Implementation status of LUMEN-X.

| **Category**                | **Item**                                                                                                                  | **Evidence / note**                               |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| A. Implemented              | Image analysis and rule-based adaptive parameters                                                                         | analysis.py                                       |
|                             | Classical branch (guided-filter illumination, Retinex, denoise, CLAHE, partial WB, saturation)                            | classical.py                                      |
|                             | Zero-DCE++ network with illumination-prior channel; non-reference and supervised losses; training script                  | zerodce.py, train.py                              |
|                             | Illumination-driven spatial fusion, heuristic quality-aware weighting, detail transfer, soft tone mapping                 | fusion.py, pipeline.py                            |
|                             | Ablation switches (leave-one-out configurations)                                                                          | pipeline.ablation_configs; evaluate.py --ablation |
|                             | Gradio demonstration, evaluation and data-preparation scripts                                                             | app.py, evaluate.py, scripts/                     |
| B. Experimentally evaluated | Classical, DL-only and hybrid modes on LOL eval15 (15 images; PSNR, SSIM, NIQE, colour difference, overexposure, latency) | Section 3.2; results/baseline_v0/lol_eval15       |
|                             | Hybrid-mode no-reference evaluation on DICM (64), LIME (10), ExDark subset (100)                                          | Section 3.3                                       |
|                             | Preliminary night-street comparison (qualitative, with simple measurements)                                               | Section 3.4, Section 4                            |
| C. Pending work             | Proper training and validation of Zero-DCE++ checkpoints on the leakage-safe splits                                       | Current weights are demo-only                     |
|                             | Completed leave-one-out ablation study                                                                                    | Configurations exist; no results yet (Table 8)    |
|                             | LPIPS evaluation                                                                                                          | Unavailable in the evaluation environment         |
|                             | Calibration or tuning of heuristic constants and of the quality score; GPU latency measurement                            | Not performed                                     |
| D. Future enhancements      | Learned or data-driven parameter prediction replacing the heuristic rules                                                 | Proposed, not implemented                         |
|                             | Semantic or object-aware region understanding for fusion                                                                  | Proposed, not implemented                         |
|                             | Iterative, feedback-driven self-correction of the output                                                                  | Proposed, not implemented                         |
|                             | Dedicated colour-cast correction and improved noise estimation for very dark inputs                                       | Proposed; motivated by Section 4                  |

**Limitation of the checkpoints.** Because the supplied weights are demo-only, every learned-component result in this report reflects an essentially untrained network and must not be taken as the capability of the Zero-DCE++ architecture. Proper training on the leakage-safe training split, model selection on the validation split and a single final evaluation on the test split are required before final performance is claimed.

# 3. Experimental Benchmarks and Evaluation

## 3.1 Protocol and Metrics

All results were produced by evaluate.py and stored in results/baseline_v0/. LOL eval15 images were processed at their full 600 × 400 resolution; unseen datasets were resized to a maximum side of 720 pixels. Evaluation was run on a CPU with one thread (PyTorch 2.15.0). The following measures are used.

- **PSNR** (dB, higher is better) and **SSIM** (higher is better) against the ground-truth image; PSNR = 10 log₁₀(1/MSE) for images in \[0, 1\].

- **NIQE** (lower is better), a no-reference naturalness measure computed with pyiqa. It tends to favour dark, smooth images and is read together with the other measures.

- **Colour difference** (lower is better): the mean Euclidean distance in CIELAB between the output and the ground truth (dE_vs_ref).

- **New overexposure (%)** (lower is better): the increase in the fraction of pixels with max-channel ≥ 0.98 relative to the input,

OE = 100 · max(0, o(Eₒᵤₜ) − o(I)) (11)

- **Latency** (ms): wall-clock enhancement time per image.

LPIPS could not be computed because the pretrained backbone could not be downloaded in the evaluation environment; it is therefore not reported.

## 3.2 Results on the LOL Dataset

Table 4 reports the mean over the 15 LOL test images for the three modes. As a reference point, the unprocessed dark input yields PSNR 7.77 dB, SSIM 0.19, NIQE 6.75 and colour difference 47.20.

**Table 4.** Measured results on LOL eval15 (15 images). Best value per row in bold.

| **Metric**             | **Classical CV** | **Deep Learning** | **Hybrid** |
|------------------------|------------------|-------------------|------------|
| PSNR (dB) ↑            | 14.52            | 16.90             | 15.78      |
| SSIM ↑                 | 0.40             | 0.56              | 0.45       |
| NIQE ↓                 | 8.67             | 8.00              | 8.47       |
| Colour Difference ↓    | 20.59            | 22.62             | 20.23      |
| New Overexposure (%) ↓ | 0.50             | 0.67              | 0.00       |
| Latency (ms) ↓         | 56               | 802               | 957        |

**Table 5.** Dispersion of the LOL results (standard deviation across the 15 images).

| **Metric**           | **Classical CV** | **Deep Learning** | **Hybrid** |
|----------------------|------------------|-------------------|------------|
| PSNR (dB)            | ± 1.46           | ± 3.50            | ± 2.49     |
| SSIM                 | ± 0.08           | ± 0.11            | ± 0.11     |
| NIQE                 | ± 1.13           | ± 1.14            | ± 1.11     |
| Colour Difference    | ± 3.67           | ± 6.54            | ± 4.93     |
| New Overexposure (%) | ± 0.72           | ± 2.29            | ± 0.00     |
| Latency (ms)         | ± 4              | ± 40              | ± 38       |

**Interpretation.**

- The deep-learning mode achieves the best PSNR and SSIM, and it also achieves the best NIQE of the three modes in this experiment. It is the best mode in PSNR on 9 of the 15 images (hybrid on 4, classical on 2).

- The hybrid mode achieves the lowest colour difference (20.23) and zero measured new overexposure on all 15 images, whereas the classical and deep-learning modes show mean new overexposure of 0.50 % and 0.67 %.

- The classical mode has the lowest latency (56 ms). The hybrid mode is the slowest (957 ms) because it executes the classical branch, the network and the fusion stages (blend map, quality scoring, detail transfer, tone mapping) in sequence.

- The hybrid mode therefore is **not** the best in every metric. Its measured advantage is robustness (colour fidelity and absence of new clipping), obtained at a cost in PSNR and SSIM relative to the deep-learning mode and at the highest latency.

- Every enhanced output has a higher (worse) mean NIQE than the dark input (6.75), and for 13 of the 15 images all three modes score worse than the input. Because NIQE favours dark, smooth images, this should not be read as the outputs being less visually natural, but it indicates that artefact and noise amplification remain, consistent with the noise-gain values of 1.56 (classical), 0.81 (deep learning) and 1.33 (hybrid) relative to the input.

**Status of these results.** They use the supplied demo-trained weights (approximately 200 steps on six synthetic images, not trained on LOL) and therefore represent a baseline evaluation, not final trained-model performance. The ranking between the deep-learning mode and the others may change after proper training.

## 3.3 Generalisation on Unseen Datasets

To assess behaviour on images that were never used for any design or training step, the hybrid pipeline was applied to three public low-light datasets: DICM (64 images), LIME (10 images) and a deterministic 100-image subset of the ExDark test split. These datasets are used for additional *no-reference* evaluation. PSNR and SSIM are deliberately not reported for them because corresponding normal-light ground-truth images are not available in this evaluation.

**Table 6.** Hybrid-mode results on unseen datasets (no-reference).

| **Dataset** | **Number of Images** | **Hybrid NIQE ↓** | **New Overexposure** |
|-------------|----------------------|-------------------|----------------------|
| DICM        | 64                   | 3.92              | 0 %                  |
| LIME        | 10                   | 4.03              | 0 %                  |
| ExDark      | 100                  | 3.49              | 0 %                  |

For context, Table 7 adds the other modes on the same images. The hybrid mode has the lowest mean NIQE on DICM and LIME, while on ExDark the classical mode is marginally lower (3.41 versus 3.49), so no single mode is best everywhere. The decisive difference is exposure safety: the deep-learning mode with demo weights produces mean new overexposure of 31.53 % (DICM), 10.11 % (LIME) and 8.06 % (ExDark), and on DICM it raises mean luminance to 0.70 from 0.33. The hybrid mode instead gives 0 %. Its noise gain relative to the input (2.19, 1.22 and 1.15 on DICM, LIME and ExDark) is lower than that of the classical mode (2.60, 1.65 and 1.64), while edge correlation with the input is similar or slightly higher (0.80, 0.72, 0.67 versus 0.79, 0.70, 0.64). These are no-reference proxies; they do not establish perceptual preference, which would need a user study.

**Table 7.** Supplementary no-reference comparison of all modes on unseen datasets (mean values).

| **Dataset**  | **Input NIQE** | **Classical NIQE** | **DL NIQE** | **Hybrid NIQE** | **Classical new OE (%)** | **DL new OE (%)** | **Hybrid new OE (%)** |
|--------------|----------------|--------------------|-------------|-----------------|--------------------------|-------------------|-----------------------|
| DICM (64)    | 4.45           | 3.98               | 4.34        | 3.92            | 0.58                     | 31.53             | 0.00                  |
| LIME (10)    | 4.41           | 4.43               | 4.41        | 4.03            | 0.85                     | 10.11             | 0.00                  |
| ExDark (100) | 4.71           | 3.41               | 3.97        | 3.49            | 0.55                     | 8.06              | 0.00                  |

## 3.4 Visual Evidence

Figures 1 to 3 are the comparison images written by evaluate.py for LOL eval15 (original side-by-side strips, re-arranged into a labelled grid without altering the pixel content). Figures 4 and 5 concern the preliminary night-street test (samples/street_before.png). Figure 4 is the four-way comparison stored in the project results directory. Figure 5 shows intermediate maps; because no intermediate-map images were stored in the results directory, they were regenerated by calling the project’s unmodified analyze, choose_params, classical_enhance, denoise and blend_weight_map functions on the street image. The network branch could not be executed in the environment used to prepare this report, so the Zero-DCE++ candidate and fused hybrid output are shown only in Figure 4, and no fusion visualisation beyond the blend-weight map is available.

![](figures/fig_lol_146.png)

**Figure 1.** LOL eval15, image 146. PSNR: classical 15.15 dB, deep learning 17.64 dB, hybrid 18.96 dB (hybrid highest for this image).

![](figures/fig_lol_22.png)

**Figure 2.** LOL eval15, image 22. PSNR: classical 14.25 dB, deep learning 20.18 dB, hybrid 19.02 dB.

![](figures/fig_lol_179.png)

**Figure 3.** LOL eval15, image 179, the weakest case for the deep-learning mode. PSNR: input 12.38 dB, classical 15.79 dB, deep learning 9.96 dB, hybrid 12.20 dB. Both learned outputs appear washed out and cooler than the ground truth.

![](figures/fig_street_compare.png)

**Figure 4.** Preliminary night-street test (stored comparison from results/street_before_compare.jpg; first four panels, in the order used by evaluate.py). The deep-learning output is cyan-tinted and strongly over-brightened; the hybrid output suppresses clipping around the lamp, although a cool cast remains. A fifth panel in the stored file, visually near-identical to the hybrid panel, is omitted because it is not labelled.

![](figures/fig_street_intermediates.png)

**Figure 5.** Intermediate maps for the night-street image from the unmodified classical branch: input, estimated illumination *L* (inferno colour map), denoised input, Retinex reflectance *R*, illumination-driven blend weight *w* (inferno; bright = use deep-learning output, dark = keep classical output) and classical output.

## 3.5 Ablation Study

The evaluation script supports a leave-one-out ablation of the hybrid pipeline (python evaluate.py … --ablation), which evaluates the full pipeline and seven variants, each with one component disabled. **No completed ablation results exist in the project files**: the stored runs were executed with ablation = false and contain only the input, classical, deep-learning and hybrid rows. The table is therefore retained as a planned experiment, and no scores are reported. The only observation in the repository is an informal one-image smoke test on a synthetic image with demo weights, which the authors themselves describe as not a result; it is not used here as evidence.

**Table 8.** Planned leave-one-out ablation of the hybrid pipeline (PENDING: no measurements available).

| **Variant**        | **Component removed**                 | **PSNR** | **SSIM** | **NIQE** | **New OE (%)** | **Status** |
|--------------------|---------------------------------------|----------|----------|----------|----------------|------------|
| Full               | None (reference)                      | —        | —        | —        | —              | Pending    |
| no_adaptive        | Adaptive parameters (fixed constants) | —        | —        | —        | —              | Pending    |
| no_predenoise      | Pre-denoising                         | —        | —        | —        | —              | Pending    |
| no_illum_prior     | Illumination-prior input channel      | —        | —        | —        | —              | Pending    |
| no_blend_map       | Spatial blend (replaced by ½(C + D))  | —        | —        | —        | —              | Pending    |
| no_quality_select  | Quality-aware weighting               | —        | —        | —        | —              | Pending    |
| no_detail_transfer | Detail transfer                       | —        | —        | —        | —              | Pending    |
| no_tonemap         | Soft tone mapping                     | —        | —        | —        | —              | Pending    |

The ablation should be run only after proper training, since conclusions about the illumination prior and the fusion stages depend on the quality of the network.

# 4. Failure Modes and Limitations

## 4.1 Observed Failures

The failures below were observed in the preliminary night-street test and in the baseline evaluation. Measurements on the street image were taken from the stored comparison panels (JPEG-compressed) using the project’s metric functions, and are indicative only. Their cause is discussed as an explanation consistent with the evidence; ablation experiments (Table 8) are required to confirm individual causes.

**Table 9.** Observed failures and supporting measurements.

| **Observed failure**                             | **Evidence**                                                                                                                                  | **Status of mitigation**                              |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Cyan / cool colour distortion (night street)     | Colour-cast measure: input 1.2, classical 3.5, deep learning 12.6, hybrid 10.4; blue-to-red mean ratio 1.15 → 1.14, 1.39, 1.32. Figure 4.     | Reduced by hybrid relative to DL-only, not eliminated |
| Excessive brightness and clipping (night street) | Pixels with max-channel ≥ 0.98: input 0.13 %, classical 2.19 %, deep learning 12.52 %, hybrid 0.70 %. Mean luminance 0.13 → 0.38, 0.54, 0.37. | Largely mitigated in hybrid                           |
| Over-brightening of unseen images by DL-only     | Mean new overexposure 31.53 % (DICM), 10.11 % (LIME), 8.06 % (ExDark); hybrid 0 % on all.                                                     | Mitigated by blending and tone mapping                |
| Washed-out, cooler rendering (LOL 179)           | DL PSNR 9.96 dB, below the input (12.38 dB); hybrid 12.20 dB; Figure 3.                                                                       | Not mitigated                                         |
| Noise amplification                              | Noise gain over input: classical 1.56 (LOL), 2.60 (DICM); hybrid 1.33, 2.19. NIQE worse than input on LOL.                                    | Reduced, not eliminated                               |
| Weakly adaptive behaviour on very dark images    | Curve iterations are 8 for all LOL images; noise estimate under-estimates in very dark images (audit).                                        | Open                                                  |

### Why colour distortion occurs

The night-street input already has a blue-dominated sky and cool-white lamp light (blue-to-red ratio 1.15), so the colour cast of the scene is partly genuine. A grey-world-type estimator such as shades-of-gray therefore returns gains close to one, and global white balancing has little to correct; moreover it is applied only partially by design to protect real dominant colours. Meanwhile the Retinex step divides each channel by the illumination, which multiplies the chroma of dark regions and of sensor chroma noise, and the saturation boost further amplifies already-saturated hues such as the sky. The network applies a separate curve to each channel without an explicit colour constraint at inference, and with demo-trained weights this adds a cast on neutral surfaces: the audit measured the *b*\* value of neutral brick moving from −0.4 to −11.3.

### Why bright objects become overexposed

Both enhancement branches are monotone brightening operators that are applied to every pixel. For the curve network, pixels that are already near saturation are pushed to 1 by repeated iterations of Equation (5); for Retinex, the reflectance is allowed to reach 1.5 before re-illumination, and CLAHE further stretches local contrast, so lamp and window pixels clip. With demo weights the network has not learned to leave lit regions untouched, which explains the 12.5 % clipped pixels in the street image and the 8–32 % new overexposure on unseen datasets.

### Why noise increases in very dark areas

In dark regions the local gain *g* = *E*/*I* is large (on LOL the mean luminance rises from 0.06 to about 0.42, a several-fold gain), and signal-independent sensor noise is multiplied by the same gain as the signal, so the signal-to-noise ratio is not improved and the absolute noise rises. Dividing by a small illumination estimate in Retinex, equalising histograms with CLAHE, and re-injecting fine detail each tend to amplify noise further, and the network may treat noise as texture.

### How the implemented stages address these problems

- **Adaptive denoising:** a guided and bilateral filter is applied *before* amplification with strength 0.25 + 0.75 ν, the CLAHE clip limit is lowered and detail transfer is damped as ν grows, and the network operates on the denoised image. Limitation: ν depends on the Immerkaer estimate, which under-estimates noise in very dark images (for the street image ν ≈ 0.10), so protection may be too weak where it is needed most.

- **Partial white balancing:** the shades-of-gray gain is applied only at strength 0.3–0.6, and the quality score penalises any increase in colour cast. This avoids neutralising genuine scene colours, but it also means a genuine-looking cast cannot be removed; the residual cast in the hybrid output (10.4) shows this is only a partial remedy.

- **Highlight protection:** the blend map gives lit regions to the conservative classical output (Figure 5), the blend threshold rises with illumination non-uniformity, and the quality score penalises newly clipped pixels with weight 6. On the street image this reduces clipped pixels from 12.52 % (deep learning) to 0.70 % (hybrid).

- **Tone mapping:** the soft shoulder of Equation (10) compresses values above 0.88 in a hue-preserving manner so that bright regions are not hard-clipped.

## 4.2 Potential Failure Cases (Not Yet Observed)

The following cases are anticipated from the design of the method but have not been observed or measured in the experiments reported here.

**Table 10.** Potential failure cases (hypothetical; no experimental evidence).

| **Potential failure**                                                                         | **Reason**                                                                                         |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Hallucinated or smoothed texture on faces and text at extreme darkness                        | Learned curves and aggressive denoising may remove or invent fine structure; untested              |
| Wrong colour correction under mixed colour temperatures or coloured light sources (e.g. neon) | A single global white-balance estimate is ambiguous; partial correction only limits the damage     |
| Blotchy chroma noise at very high ISO                                                         | Denoising strength depends on a noise estimate that may be too low                                 |
| Backlit or high-dynamic-range scenes with large bright regions                                | Illumination-threshold blending may assign shadow regions to the wrong branch                      |
| Motion blur and inputs with no recoverable signal                                             | Information is absent; enhancement cannot restore it                                               |
| Very large images                                                                             | Curve maps are estimated at reduced scale, which may blur fine illumination changes; not evaluated |

## 4.3 Limitations of the Present Evaluation

- The learned components use demo-only weights; all learned-model results are baseline results.

- The test set contains 15 images, and standard deviations are large (for example ± 3.50 dB PSNR for deep learning); small differences between modes, such as hybrid versus classical NIQE, should not be over-interpreted.

- NIQE is an imperfect proxy and, together with the other no-reference measures, does not replace a subjective study. LPIPS is not available.

- The street-image analysis is a single preliminary example without ground truth; the regenerated maps in Figure 5 cover only the classical branch and the blend map.

- The ablation study has not been completed, and the heuristic constants of Sections 2.2 and 2.6 have not been tuned or validated.

- Latency was measured on a single CPU thread and is not representative of GPU deployment.

# 5. Deployment and Quickstart Guide

The package requires Python with numpy, scipy, opencv-python-headless, scikit-image, pandas, tabulate, torch and gradio; pyiqa (NIQE) and lpips are optional. The Gradio application can be run locally or deployed as is on a Hugging Face Space with the Gradio SDK. Without network access to the model hosts, NIQE or LPIPS values may be reported as NaN.

### Installation and demonstration

> pip install -r requirements.txt
>
> python app.py \# Gradio UI: modes, metrics, intermediate maps

### Data preparation and verification

> bash scripts/download_data.sh data \# LOL, DICM, LIME, ExDark subset + leakage-safe splits
>
> python scripts/verify_data.py --data data

### Evaluation

> python evaluate.py --low_dir data/LOL/eval15/low --high_dir data/LOL/eval15/high \\
>
> --split_json data/splits.json --max_side 0 --out results/lol_eval15
>
> python evaluate.py --low_dir data/unseen/DICM --out results/unseen_DICM
>
> python evaluate.py --low_dir data/LOL/eval15/low --high_dir data/LOL/eval15/high --ablation

### Training (required before reporting final performance)

> python train.py --low_dir data/LOL/our485/low --epochs 100 --bs 8 --crop 256
>
> python train.py --low_dir data/LOL/our485/low --epochs 100 --bs 8 --crop 256 --hybrid
>
> python train.py --low_dir data/LOL/our485/low --high_dir data/LOL/our485/high \\
>
> --init weights/zerodce_rgb.pt --lr 5e-5 \# supervised fine-tuning

Training should use only the train list of data/splits.json (the example paths above point at the full our485 folder for brevity; restrict them to the split before reporting results), validation images for checkpoint selection, and eval15 only for the final test. On the CPU configuration used here, a 600 × 400 image takes about 56 ms in classical mode and about 0.96 s in hybrid mode, so real-time operation would require a GPU or further optimisation, which has not been evaluated.

# 6. Expected Outcomes

The following outcomes are separated into those supported by the present measurements and those that are expectations conditional on further work.

**Supported by current measurements.** The hybrid pipeline avoids new clipping on every evaluated image across four datasets (LOL eval15, DICM, LIME, ExDark), gives the lowest colour difference to the reference on LOL, and substantially reduces the over-brightening of the deep-learning branch on unseen images. The deep-learning branch delivers the highest full-reference scores even with demo weights, and the classical branch provides a fast, deterministic fallback.

**Expected after further work (not yet demonstrated).** Training the network on the leakage-safe training split is expected to improve the full-reference scores of the learned and hybrid modes and to reduce the colour cast introduced by the demo weights. A completed ablation study is expected to identify which fusion components contribute measurably. Replacing heuristic rules and the hand-designed quality score by validated or learned components is expected to improve consistency, but this is a hypothesis to be tested. No numerical improvement is claimed here.

# 7. Conclusion

LUMEN-X is a reproducible low-light enhancement framework that combines a classical Retinex/CLAHE branch and a Zero-DCE++ network through illumination-driven spatial fusion, heuristic quality-aware weighting, detail transfer and tone mapping, with image-statistics-driven heuristic parameters. The baseline evaluation with demo-trained weights shows that no single mode dominates: deep learning leads in PSNR, SSIM and NIQE on LOL, classical processing is fastest, and the hybrid mode gives the best colour fidelity and no measured new overexposure on LOL and three unseen datasets, at the cost of the highest latency. The failure analysis shows that colour casts and noise amplification persist in difficult night scenes even when over-exposure is controlled.

The principal outstanding tasks are proper training and validation of the network, completion of the ablation study, and refinement or learning of the heuristic components. Semantic region understanding and iterative self-correction are proposals for future work and are not part of the present system.

# 8. References

1.  C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, R. Cong, “Zero-reference deep curve estimation for low-light image enhancement,” in Proc. IEEE/CVF CVPR, 2020.

2.  C. Li, C. Guo, C. C. Loy, “Learning to enhance low-light image via zero-reference deep curve estimation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 8, 2022.

3.  X. Guo, Y. Li, H. Ling, “LIME: Low-light image enhancement via illumination map estimation,” IEEE Trans. Image Process., vol. 26, no. 2, 2017.

4.  C. Wei, W. Wang, W. Yang, J. Liu, “Deep Retinex decomposition for low-light enhancement,” in Proc. BMVC, 2018 (LOL dataset).

5.  D. J. Jobson, Z. Rahman, G. A. Woodell, “A multiscale Retinex for bridging the gap between color images and the human observation of scenes,” IEEE Trans. Image Process., vol. 6, no. 7, 1997.

6.  K. He, J. Sun, X. Tang, “Guided image filtering,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 6, 2013.

7.  C. Tomasi, R. Manduchi, “Bilateral filtering for gray and color images,” in Proc. ICCV, 1998.

8.  K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in Graphics Gems IV, Academic Press, 1994.

9.  G. D. Finlayson, E. Trezzi, “Shades of gray and colour constancy,” in Proc. IS&T/SID Color Imaging Conference, 2004.

10. J. Immerkær, “Fast noise variance estimation,” Computer Vision and Image Understanding, vol. 64, no. 2, 1996.

11. A. Mittal, R. Soundararajan, A. C. Bovik, “Making a ‘completely blind’ image quality analyzer,” IEEE Signal Process. Lett., vol. 20, no. 3, 2013.

12. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, 2004.

13. C. Lee, C. Lee, C.-S. Kim, “Contrast enhancement based on layered difference representation of 2D histograms,” IEEE Trans. Image Process., vol. 22, no. 12, 2013 (DICM).

14. Y. P. Loh, C. S. Chan, “Getting to know low-light images with the Exclusively Dark dataset,” Computer Vision and Image Understanding, vol. 178, 2019 (ExDark).

15. C. Chen, J. Mo, “IQA-PyTorch: PyTorch toolbox for image quality assessment,” 2022 (pyiqa).

# 9. Submission Checklist

**Table 11.** Submission checklist and status.

| **Item**                                               | **Status**           | **Note**                                               |
|--------------------------------------------------------|----------------------|--------------------------------------------------------|
| Technical solution report (this document)              | Complete             | DOCX and Markdown text provided in docs/               |
| Source code (lowlight/, app.py, train.py, evaluate.py) | Complete             | Unmodified in this revision                            |
| Interactive demonstration (Gradio)                     | Provided             | Run locally; public hosting not verified               |
| Baseline benchmark on LOL eval15                       | Complete             | Demo-trained weights; Section 3.2                      |
| Unseen-dataset evaluation (DICM, LIME, ExDark)         | Complete             | No-reference only; Section 3.3                         |
| Visual comparison figures                              | Complete             | Figures 1–5; maps for the network branch not available |
| Leakage-safe data splits and verification script       | Complete             | configs/splits_reference.json, scripts/                |
| Properly trained model checkpoints                     | Pending              | Current weights are demo-only                          |
| Ablation study results                                 | Pending              | Table 8 is a plan                                      |
| LPIPS results                                          | Pending              | Backbone unavailable in the evaluation environment     |
| Presentation slides / demonstration video              | Not in project files | To be prepared by the team                             |
| Team and institution details                           | Pending              | To be filled on the title page                         |
