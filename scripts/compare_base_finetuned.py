#!/usr/bin/env python3
"""Run base vs finetuned SAM3 text-prompt inference and save comparison images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import COLORS, plot_bbox, plot_mask


DEFAULT_CASES = [
    ("0000.jpg", "A large blue semi-trailer truck"),
    ("0010.jpg", "A blue shipping container with identification markings"),
    ("0020.jpg", "A stack of shipping containers"),
    ("0030.jpg", "A blue truck trailer"),
    ("0040.jpg", "A row of blue semi-trailers parked in a lot"),
    ("0050.jpg", "A yellow crane or lifting equipment"),
    ("0060.jpg", "A shipping container with logo markings"),
    ("0069.jpg", "A truck in a port or container yard"),
]


def _bpe_path() -> str:
    return str(PROJECT_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def load_base_model(device: str):
    model = build_sam3_image_model(
        bpe_path=_bpe_path(),
        checkpoint_path=str(PROJECT_ROOT / "pretrained" / "sam3" / "sam3.pt"),
        load_from_HF=False,
        device=device,
    )
    model.eval()
    return model


def load_finetuned_model(device: str):
    model = build_sam3_image_model(
        bpe_path=_bpe_path(),
        checkpoint_path=None,
        load_from_HF=False,
        device=device,
    )
    ckpt_path = PROJECT_ROOT / "workdir" / "exp-001" / "checkpoints" / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference(processor: Sam3Processor, image_path: Path, prompt: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    return {
        "image": image,
        "masks": state["masks"],
        "boxes": state["boxes"],
        "scores": state["scores"],
        "num_objects": len(state["scores"]),
    }


def render_overlay(image: Image.Image, result: dict) -> Image.Image:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")
    w, h = image.size
    for i in range(result["num_objects"]):
        color = COLORS[i % len(COLORS)]
        plot_mask(result["masks"][i].squeeze(0).cpu(), color=color, ax=ax)
        prob = result["scores"][i].item()
        plot_bbox(
            h,
            w,
            result["boxes"][i].cpu(),
            text=f"(id={i}, prob={prob:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
            ax=ax,
        )
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    rgba = fig.canvas.buffer_rgba()
    out = Image.frombuffer("RGBA", fig.canvas.get_width_height(), rgba).convert("RGB")
    plt.close(fig)
    return out


def save_side_by_side(
    base_img: Image.Image,
    ft_img: Image.Image,
    prompt: str,
    out_path: Path,
) -> None:
    gap = 12
    width = base_img.width + ft_img.width + gap
    height = max(base_img.height, ft_img.height) + 48
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(base_img, (0, 40))
    canvas.paste(ft_img, (base_img.width + gap, 40))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(
        f"Baseline (left) vs Finetuned exp-001 (right)\nPrompt: {prompt}",
        fontsize=11,
        loc="left",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        default=str(PROJECT_ROOT / "datasets" / "raw_images_test"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "records" / "results" / "exp-001"),
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument(
        "--cases_json",
        default="",
        help="Optional JSON list of [filename, prompt] pairs",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cases_json:
        with open(args.cases_json) as f:
            cases = [tuple(x) for x in json.load(f)]
    else:
        cases = DEFAULT_CASES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = []

    for label, loader in [("baseline", load_base_model), ("finetuned", load_finetuned_model)]:
        print(f"Loading {label} model on {device}...")
        model = loader(device)
        processor = Sam3Processor(
            model,
            resolution=args.resolution,
            confidence_threshold=args.confidence_threshold,
        )

        for filename, prompt in cases:
            image_path = image_dir / filename
            if not image_path.exists():
                print(f"Skip missing image: {image_path}")
                continue

            stem = Path(filename).stem
            safe_prompt = prompt.replace("/", "_")[:40].replace(" ", "_")
            desc = f"{stem}_{safe_prompt}"

            print(f"[{label}] {filename} | {prompt[:60]}...")
            result = run_inference(processor, image_path, prompt)
            overlay = render_overlay(result["image"], result)

            out_png = output_dir / f"{desc}_{label}.png"
            overlay.save(out_png)

            entry = next((s for s in summary if s["image"] == filename and s["prompt"] == prompt), None)
            if entry is None:
                entry = {"image": filename, "prompt": prompt}
                summary.append(entry)
            entry[f"{label}_objects"] = result["num_objects"]
            entry[f"{label}_scores"] = [round(s, 4) for s in result["scores"].tolist()]
            entry[f"{label}_png"] = str(out_png.relative_to(PROJECT_ROOT))

        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    for entry in summary:
        stem = Path(entry["image"]).stem
        safe_prompt = entry["prompt"].replace("/", "_")[:40].replace(" ", "_")
        desc = f"{stem}_{safe_prompt}"
        base_path = output_dir / f"{desc}_baseline.png"
        ft_path = output_dir / f"{desc}_finetuned.png"
        compare_path = output_dir / f"{desc}_compare.png"
        if base_path.exists() and ft_path.exists():
            save_side_by_side(
                Image.open(base_path),
                Image.open(ft_path),
                entry["prompt"],
                compare_path,
            )
            entry["compare_png"] = str(compare_path.relative_to(PROJECT_ROOT))

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(summary)} comparisons to {output_dir}")


if __name__ == "__main__":
    main()
