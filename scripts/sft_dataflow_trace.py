#!/usr/bin/env python3
"""Trace SAM3 exp-001 fine-tuning data flow with measured shapes.

Usage:
  python scripts/sft_dataflow_trace.py --doc
  python scripts/sft_dataflow_trace.py --run
  python scripts/sft_dataflow_trace.py --run --forward
  python scripts/sft_dataflow_trace.py --run --step A3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

STEPS = {
    "A1": "Load annotations.json (COCO JSON)",
    "A2": "COCO_FROM_JSON → queries + annotations",
    "A3": "Sam3ImageDataset.__getitem__ + transforms",
    "A4": "collate_fn_api → BatchedDatapoint",
    "A5": "Trainer._step batch → device",
    "A6": "Phase A summary (single-sample + batch)",
    "B1": "Sam3Image.forward: vision + text backbone",
    "B2": "forward_grounding: encoder + decoder",
    "B3": "Hungarian matching (_compute_matching)",
    "B4": "Sam3LossWrapper.compute_loss",
    "B5": "Loss components → core_loss scalar",
    "B6": "Phase B summary",
}

RAW_SAMPLE = {
    "image_file": "datasets/raw_images_train/0000.jpg",
    "category_name": "object",
    "note": "exp-001 uses category name as query_text, not noun_phrase",
}


def load_cfg():
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    with initialize_config_module(config_module="sam3.train", version_base="1.2"):
        cfg = compose(config_name="configs/mydata/text_only_train")
    OmegaConf.resolve(cfg)
    return cfg


def build_dataset(cfg, sample_idx: int = 0):
    from hydra.utils import instantiate

    ds = instantiate(cfg.trainer.data.train.dataset)
    ds.curr_epoch = 0
    return ds, ds[sample_idx]


def build_collator(cfg):
    from hydra.utils import instantiate

    return instantiate(cfg.train_args.collate_fn)


def trace_a1():
    ann_path = PROJECT_ROOT / "datasets/custom0_exp001/annotations.json"
    with open(ann_path) as f:
        data = json.load(f)
    img0 = data["images"][0]
    anns0 = [a for a in data["annotations"] if a["image_id"] == img0["id"]]
    return {
        "ann_path": str(ann_path),
        "num_images": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "sample_image": img0,
        "sample_ann_count": len(anns0),
        "sample_noun_phrase": anns0[0].get("noun_phrase", "")[:80] if anns0 else "",
        "categories": data.get("categories", []),
    }


def trace_a2(ds, sample_idx: int = 0):
    loader = ds.coco
    queries, anns = loader.loadQueriesAndAnnotationsFromDatapoint(sample_idx)
    q0 = queries[0]
    return {
        "num_queries": len(queries),
        "query_text": q0["query_text"],
        "num_gt_objects": len(q0["object_ids_output"]),
        "sample_bbox_xywh_norm": anns[0]["bbox"].tolist() if anns else None,
        "has_segmentation_rle": anns[0]["segmentation"] is not None if anns else False,
    }


def trace_a3(datapoint):
    img = datapoint.images[0]
    q = datapoint.find_queries[0]
    return {
        "img_tensor_shape": tuple(img.data.shape),
        "num_objects": len(img.objects),
        "query_text": q.query_text,
        "num_output_objects": len(q.object_ids_output),
        "sample_bbox_cxcywh": img.objects[0].bbox.tolist() if img.objects else None,
        "sample_mask_shape": tuple(img.objects[0].segment.shape)
        if img.objects and img.objects[0].segment is not None
        else None,
    }


def trace_a4(collator, datapoint):
    batch = collator([datapoint])
    key = next(iter(batch))
    bd = batch[key]
    ft = bd.find_targets[0]
    fi = bd.find_inputs[0]
    return {
        "dict_key": key,
        "img_batch_shape": tuple(bd.img_batch.shape),
        "find_text_batch": bd.find_text_batch,
        "num_boxes": ft.num_boxes.tolist(),
        "boxes_padded_shape": tuple(ft.boxes_padded.shape),
        "object_ids_padded_shape": tuple(ft.object_ids_padded.shape),
        "input_boxes_shape": tuple(fi.input_boxes.shape),
        "input_points_shape": tuple(fi.input_points.shape),
        "segments_count": len(ft.segments),
    }


def _ensure_dist():
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def trace_forward(cfg, batch_dict, device: str = "cuda"):
    import torch
    from hydra.utils import instantiate
    from sam3.model.utils.misc import copy_data_to_device

    if not torch.cuda.is_available():
        device = "cpu"

    _ensure_dist()
    key = next(iter(batch_dict))
    batch = copy_data_to_device(batch_dict[key], device)
    model = instantiate(cfg.trainer.model)
    model = model.to(device)
    model.train()

    with torch.no_grad():
        find_stages = model(batch)
        from sam3.model.model_misc import SAM3Output

        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as stages_iter:
            stage_outputs = list(stages_iter)[0]
            out = stage_outputs[0]
        find_targets = [model.back_convert(x) for x in batch.find_targets]
        loss_fn = instantiate(cfg.trainer.loss.all)
        losses = loss_fn(find_stages, find_targets)

    result = {
        "device": device,
        "pred_logits_shape": tuple(out["pred_logits"].shape),
        "pred_boxes_shape": tuple(out["pred_boxes"].shape),
        "pred_logits_o2m_shape": tuple(out["pred_logits_o2m"].shape)
        if "pred_logits_o2m" in out
        else None,
        "pred_masks_shape": tuple(out["pred_masks"].shape)
        if out.get("pred_masks") is not None
        else None,
        "targets_num_boxes": find_targets[0]["num_boxes"].tolist(),
        "targets_boxes_padded_shape": tuple(find_targets[0]["boxes_padded"].shape),
        "loss_keys": sorted(losses.keys()),
        "core_loss": float(losses["core_loss"].item()),
        "loss_bbox": float(losses.get("loss_bbox", 0)),
        "loss_giou": float(losses.get("loss_giou", 0)),
        "loss_ce": float(losses.get("loss_ce", 0)),
    }
    return result


def print_doc():
    print("SAM3 exp-001 Fine-tuning Data Flow — Step Index")
    print("=" * 60)
    print("Scope: text-instruction grounding fine-tune (NOT LLM SFT / DPO)")
    print()
    for step, desc in STEPS.items():
        print(f"  {step}: {desc}")
    print()
    print("Artifacts:")
    print("  SFT_DATAFLOW.md       — spec (shapes + line numbers)")
    print("  SFT_DATAFLOW_LEARN.md — learning path")
    print("  scripts/sft_dataflow_trace.py — this script")


def run_trace(forward: bool = False, step: str | None = None):
    steps = [step] if step else list(STEPS)

    if "A1" in steps or not step:
        print("\n=== A1: annotations.json ===")
        print(json.dumps(trace_a1(), indent=2, ensure_ascii=False))

    cfg = load_cfg()
    ds, datapoint = build_dataset(cfg, sample_idx=0)

    if "A2" in steps or not step:
        print("\n=== A2: COCO_FROM_JSON ===")
        print(json.dumps(trace_a2(ds, 0), indent=2, ensure_ascii=False))

    if "A3" in steps or not step:
        print("\n=== A3: Dataset __getitem__ ===")
        print(json.dumps(trace_a3(datapoint), indent=2, ensure_ascii=False))

    collator = build_collator(cfg)
    batch_dict = collator([datapoint])

    if "A4" in steps or "A5" in steps or "A6" in steps or not step:
        a4 = trace_a4(collator, datapoint)
        if "A4" in steps or not step:
            print("\n=== A4: collate_fn_api ===")
            print(json.dumps(a4, indent=2, ensure_ascii=False))
        if "A5" in steps or "A6" in steps or not step:
            print("\n=== A5/A6: batch ready for model(**batch) ===")
            print(json.dumps({"device_target": "cuda", **a4}, indent=2))

    if forward and (step is None or step.startswith("B")):
        print("\n=== B1–B6: forward + loss (measured) ===")
        print(json.dumps(trace_forward(cfg, batch_dict), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--step", type=str, default=None, choices=list(STEPS))
    args = parser.parse_args()

    if args.doc:
        print_doc()
    if args.run or args.step:
        run_trace(forward=args.forward, step=args.step)
    if not args.doc and not args.run and not args.step:
        print_doc()


if __name__ == "__main__":
    main()
