#!/usr/bin/env python3
"""Compare SAM3 baseline vs exp-001 vs exp-002 on the same text prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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

MODEL_SPECS = {
    "baseline": {
        "title": "Baseline (sam3.pt)",
        "checkpoint": PROJECT_ROOT / "pretrained" / "sam3" / "sam3.pt",
        "load_ckpt_key": None,
    },
    "exp001": {
        "title": "exp-001 (query=object)",
        "checkpoint": PROJECT_ROOT / "workdir" / "exp-001" / "checkpoints" / "checkpoint.pt",
        "load_ckpt_key": "model",
    },
    "exp002": {
        "title": "exp-002 (noun_phrase)",
        "checkpoint": PROJECT_ROOT / "workdir" / "exp-002" / "checkpoints" / "checkpoint.pt",
        "load_ckpt_key": "model",
    },
}


def _bpe_path() -> str:
    return str(PROJECT_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")


def load_model(spec: dict, device: str):
    if spec["load_ckpt_key"] is None:
        model = build_sam3_image_model(
            bpe_path=_bpe_path(),
            checkpoint_path=str(spec["checkpoint"]),
            load_from_HF=False,
            device=device,
        )
    else:
        model = build_sam3_image_model(
            bpe_path=_bpe_path(),
            checkpoint_path=None,
            load_from_HF=False,
            device=device,
        )
        ckpt = torch.load(spec["checkpoint"], map_location="cpu")
        model.load_state_dict(ckpt[spec["load_ckpt_key"]], strict=False)
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


def save_triple_compare(
    images: list[Image.Image],
    titles: list[str],
    prompt: str,
    out_path: Path,
) -> None:
    gap = 8
    label_h = 28
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images) + label_h + 56
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    x = 0
    for img, title in zip(images, titles):
        canvas.paste(img, (x, label_h + 40))
        fig_tmp, ax_tmp = plt.subplots(figsize=(img.width / 100, 0.3), dpi=100)
        ax_tmp.text(0.5, 0.5, title, ha="center", va="center", fontsize=9)
        ax_tmp.axis("off")
        fig_tmp.canvas.draw()
        label = Image.frombuffer(
            "RGBA",
            fig_tmp.canvas.get_width_height(),
            fig_tmp.canvas.buffer_rgba(),
        ).convert("RGB")
        plt.close(fig_tmp)
        canvas.paste(label.resize((img.width, label_h)), (x, 40))
        x += img.width + gap

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f"Prompt: {prompt}", fontsize=10, loc="left")
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
        default=str(PROJECT_ROOT / "records" / "results" / "exp-002"),
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = [{ "image": fn, "prompt": p } for fn, p in DEFAULT_CASES]
    overlays_by_case: dict[str, dict[str, Image.Image]] = {}

    for label, spec in MODEL_SPECS.items():
        if not spec["checkpoint"].exists():
            print(f"Skip {label}: missing {spec['checkpoint']}")
            continue

        print(f"Loading {label} on {device}...")
        model = load_model(spec, device)
        processor = Sam3Processor(
            model,
            resolution=args.resolution,
            confidence_threshold=args.confidence_threshold,
        )

        for entry in summary:
            filename = entry["image"]
            prompt = entry["prompt"]
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

            entry[f"{label}_objects"] = result["num_objects"]
            entry[f"{label}_scores"] = [round(s, 4) for s in result["scores"].tolist()]
            entry[f"{label}_png"] = str(out_png.relative_to(PROJECT_ROOT))
            overlays_by_case.setdefault(desc, {})[label] = overlay

        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()

    for entry in summary:
        stem = Path(entry["image"]).stem
        safe_prompt = entry["prompt"].replace("/", "_")[:40].replace(" ", "_")
        desc = f"{stem}_{safe_prompt}"
        overlays = overlays_by_case.get(desc, {})
        if all(k in overlays for k in MODEL_SPECS):
            compare_path = output_dir / f"{desc}_compare.png"
            save_triple_compare(
                [overlays["baseline"], overlays["exp001"], overlays["exp002"]],
                [
                    MODEL_SPECS["baseline"]["title"],
                    MODEL_SPECS["exp001"]["title"],
                    MODEL_SPECS["exp002"]["title"],
                ],
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
