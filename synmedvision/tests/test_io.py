from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.cli.generate import generate_samples
from src.data.prepare_pcam import prepare_pcam
from src.models.diffusion.finetune_lora import finetune_lora
from src.models.stylegan2.train import train_stylegan2


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        dataset: "pcam"
        root: "{root}"
        img_size: 256
        val_split: 0.2
        test_split: 0.2
        split_by_patient: false
        make_masks: false
        normalize:
          method: "none"
        output_csv: "{root}/manifest.csv"
        """.replace("{root}", str(tmp_path / "pcam")),
        encoding="utf-8",
    )
    return config_path


def test_prepare_pcam_outputs_expected_structure(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    prepare_pcam(config_path)
    root = tmp_path / "pcam"

    for split in ["train_images", "val_images", "test_images"]:
        for image_path in (root / split / "normal").glob("*.png"):
            with Image.open(image_path) as img:
                assert img.size == (256, 256)
                assert img.mode == "RGB"

    manifest_path = root / "manifest.csv"
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
        assert set(row.keys()) == {"path", "label", "split", "patient_id", "mask_path"}


def test_generate_samples_creates_manifest(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    prepare_pcam(config_path)

    stylegan_ckpt = train_stylegan2({"out_dir": tmp_path / "out/stylegan2", "seed": 123})
    finetune_lora({"out_dir": tmp_path / "out/diffusion_lora", "seed": 123})

    args = type("Args", (), {
        "n": 10,
        "class_mix": "normal:0.5,tumor:0.5",
        "out": str(tmp_path / "samples"),
        "model": "stylegan2",
        "seed": 123,
        "steps": 30,
        "guidance": 7.5,
        "mask_path": None,
        "ckpt": str(stylegan_ckpt),
        "base": str(tmp_path / "out/diffusion_lora"),
    })

    paths = list(generate_samples(args))
    assert len(paths) == 10
    manifest_path = Path(args.out) / "manifest.csv"
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 10
    assert set(rows[0].keys()) == {"path", "class", "seed", "model", "params"}
