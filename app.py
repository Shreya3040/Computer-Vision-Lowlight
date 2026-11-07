"""Gradio app: python app.py  (deploy as-is on Hugging Face Spaces; set sdk: gradio)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import gradio as gr
import numpy as np
import pandas as pd
from lowlight import LowLightEnhancer
from lowlight.metrics import evaluate_pair
from lowlight.utils import to_float, to_u8, resize_max_side

enh = LowLightEnhancer()
HAS_DL = bool(enh.models)


def _vis(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 2:
        x = cv2.applyColorMap(to_u8(x), cv2.COLORMAP_INFERNO)[..., ::-1] / 255.0
    return to_u8(x)


def run(img, mode, max_side):
    if img is None:
        return None, None, None, "Upload an image."
    x = to_float(img[..., :3])
    x = resize_max_side(x, int(max_side))
    modes = ["classical", "dl", "hybrid"] if HAS_DL else ["classical"]
    results = {m: enh.enhance(x, m) for m in modes}
    main = enh.enhance(x, mode) if mode == "auto" else results.get(mode, results["classical"])

    rows = []
    for m, r in results.items():
        mt = evaluate_pair(x, r.output)
        rows.append({"method": m, "latency_ms": round(r.latency_ms, 1), "mean_lum": round(mt["mean_lum"], 3),
                     "overexposed_%": round(mt["overexposed_%"], 2), "colour_cast": round(mt["colour_cast"], 1),
                     "noise_gain": round(mt["noise_gain"], 2), "edge_corr": round(mt["edge_corr"], 3)})
    table = pd.DataFrame(rows)

    cmp_gallery = [(_vis(x), "input")] + [(_vis(r.output), m) for m, r in results.items()]
    inter = [(_vis(v), k) for k, v in main.intermediates.items()]
    st = main.stats
    report = (f"**Darkness:** {st.darkness_level} (mean luminance {st.mean_lum:.3f}) | noise σ {st.noise_sigma:.4f} | "
              f"non-uniformity {st.nonuniformity:.2f}\n\n**Adaptive params:** "
              + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in main.params.items())
              + ("\n\n" + "\n\n".join(main.info) if main.info else ""))
    return cmp_gallery, inter, table, report


with gr.Blocks(title="Adaptive Low-Light Restoration") as demo:
    gr.Markdown("# Adaptive low-light restoration\nClassical Retinex/CLAHE + Zero-DCE++ hybrid with quality-aware fusion.")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Low-light image")
        with gr.Column():
            mode = gr.Radio(["auto", "classical", "dl", "hybrid"], value="auto", label="Mode")
            ms = gr.Slider(256, 2048, value=1024, step=64, label="Max side (px)")
            btn = gr.Button("Enhance", variant="primary")
    report = gr.Markdown()
    g1 = gr.Gallery(label="Input vs classical / DL-only / hybrid", columns=4, height=380)
    tbl = gr.Dataframe(label="Metrics (no-reference) and latency")
    g2 = gr.Gallery(label="Intermediate visualisations (illumination, reflectance, blend map, ...)", columns=4, height=380)
    btn.click(run, [inp, mode, ms], [g1, g2, tbl, report])

if __name__ == "__main__":
    demo.launch()
