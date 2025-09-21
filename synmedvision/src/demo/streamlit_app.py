"""Streamlit demo for the SynMedVision pipeline."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

repo_root = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from src.cli.generate import generate_samples
from src.eval.privacy import privacy_report
from src.eval.realism import compute_realism_metrics
from src.eval.utility import compute_utility_metrics

st.set_page_config(page_title="SynMedVision Demo", layout="wide")


def _class_mix_string(label: str) -> str:
    return f"{label}:1.0"


def _collect_gallery(paths) -> list[Image.Image]:
    gallery = []
    for path in paths[:16]:
        gallery.append(Image.open(path))
    return gallery


def _save_uploaded_mask(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    tmp_dir = Path(tempfile.mkdtemp())
    mask_path = tmp_dir / uploaded_file.name
    with mask_path.open("wb") as fh:
        fh.write(uploaded_file.read())
    return str(mask_path)


def main() -> None:
    st.sidebar.header("Generation Settings")
    model = st.sidebar.selectbox("Model", ["Diffusion (LoRA)", "StyleGAN2"], index=0)
    class_label = st.sidebar.selectbox("Class", ["normal", "tumor"], index=0)
    n_images = st.sidebar.slider("Number of images", 4, 32, 16, step=4)
    steps = st.sidebar.slider("Diffusion steps", 10, 100, 30)
    guidance = st.sidebar.slider("Guidance scale", 1.0, 15.0, 7.5)
    seed = st.sidebar.number_input("Seed", value=123, step=1)
    uploaded_mask = st.sidebar.file_uploader("Optional mask", type=["png"])

    out_dir = Path("out/demo_samples")
    mask_path = _save_uploaded_mask(uploaded_mask)

    if st.sidebar.button("Generate"):
        args = argparse.Namespace(
            n=n_images,
            class_mix=_class_mix_string(class_label),
            out=str(out_dir),
            model="diffusion_lora" if model.startswith("Diffusion") else "stylegan2",
            seed=int(seed),
            steps=int(steps),
            guidance=float(guidance),
            mask_path=mask_path,
            ckpt=None,
            base=None,
        )
        generated = generate_samples(args)
        st.session_state["generated_paths"] = [str(p) for p in generated]

    paths = st.session_state.get("generated_paths", [])
    st.title("SynMedVision Demo")

    if not paths:
        st.info("Use the sidebar to generate images.")
        return

    cols = st.columns(4)
    for idx, image in enumerate(_collect_gallery([Path(p) for p in paths])):
        with cols[idx % 4]:
            st.image(image, caption=f"Sample {idx+1}")

    metrics_realism = compute_realism_metrics("data/pcam/val_images", out_dir)
    metrics_utility = compute_utility_metrics(Path("data/pcam"), out_dir)
    metrics_privacy = privacy_report("data/pcam/val_images", out_dir, tau=0.3)

    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("KID", f"{metrics_realism.get('kid', float('nan')):.4f}")
    col2.metric("TSTR AUC", f"{metrics_utility.get('tstr', {}).get('roc_auc', float('nan')):.3f}")
    col3.metric("Privacy risk (%)", f"{metrics_privacy.get('near_duplicate_pct', float('nan')):.2f}")

    tab_realism, tab_utility, tab_privacy = st.tabs(["Realism", "Utility", "Privacy"])

    with tab_realism:
        st.json(metrics_realism)
    with tab_utility:
        st.json(metrics_utility)
    with tab_privacy:
        st.json(metrics_privacy)

    st.download_button("Download images manifest", data="\n".join(paths), file_name="synthetic_images.txt")


if __name__ == "__main__":
    main()
