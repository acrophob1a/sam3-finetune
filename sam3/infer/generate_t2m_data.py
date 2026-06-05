import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from tqdm import tqdm
import numpy as np
import re
import torch
import shutil
import argparse
from collections import defaultdict
import uuid
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from datasets import (
    Dataset, Features, Image, Sequence, Value,
    load_from_disk, concatenate_datasets
)

import sam3
sam3_root = os.path.dirname(os.path.join(os.path.dirname(sam3.__file__)))
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.visualize import show_mask, show_points, show_box, show_masks
from utils.misc import mask_to_img, img_to_mask, overlay_mask_on_image, save_datalist_to_disk


def set_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam_path",
        type=str,
        default='pretrained/sam3/sam3.pt',
        help="checkpoint path",
    )
    parser.add_argument(
        "--qwen_path",
        type=str,
        default='pretrained/Qwen2.5-VL-7B-Instruct',
        help="directory of saved images",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=4,
        help="batch size per gpu",
    )
    parser.add_argument(
        "--vlm_batchsize",
        type=int,
        default=4,
        help="batch size per gpu",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default='datasets/custom0',
        help="directory of saved images",
    )
    parser.add_argument(
        "--num_pts",
        type=int,
        default=512,
        help="number of random sampled points",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="score threshold",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.25,
        help="iou threshold",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="directory of saved images",
    )
    args = parser.parse_args()
    return args


def get_sam_model_and_processor(args):
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    checkpoint_path = f'{sam3_root}/{args.sam_path}'
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model)
    return model, processor


def image_batch_preload(path_list, processor):
    image_pils = []
    for path in path_list:
        image = PILImage.open(path)
        image_pils.append(image)
    inference_state = processor.set_image_batch(image_pils)
    return inference_state, image_pils


def generate_random_points_per_image(size, num_pts=100):
    normalized_pts = np.random.uniform(0.0, 1.0, size=(num_pts, 1, 2))
    img_size = np.array(size)    # 先变成 NumPy 数组
    img_size = img_size.reshape(1, 1, 2)
    pts = (normalized_pts * img_size).astype(int)
    return pts


def compute_pairwise_iou(masks):
    """
    计算 pairwise IoU 矩阵 (n, n)，上三角镜像重复，
    iou_matrix[i,j] 表示 masks[i] 和 masks[j] 的 IoU。
    """
    n = masks.shape[0]
    masks_flat = masks.reshape(n, -1)
    # 先计算交集
    inter = np.dot(masks_flat, masks_flat.T)  # 每对 m_i 和 m_j 的交集像素
    area = masks_flat.sum(axis=1, keepdims=True)
    union = area + area.T - inter
    # 避免除0
    union = np.where(union == 0, 1e-6, union)
    iou_matrix = inter / union
    return iou_matrix


def filter_and_dedup_masks(
    masks, 
    scores, 
    score_thresh=0.5, 
    iou_thresh=0.5,
):
    # 1) 筛分
    keep_idx = np.where(scores >= score_thresh)[0]
    if len(keep_idx) == 0:
        return None, None
    masks = masks[keep_idx]
    scores = scores[keep_idx]

    # 2) 按 scores 排序
    order = np.argsort(scores)[::-1]
    masks = masks[order]
    scores = scores[order]
    n = masks.shape[0]
    iou_mat = compute_pairwise_iou(masks)

    # 标记是否 suppress
    suppressed = np.zeros(n, dtype=bool)
    final_masks, final_scores = [], []
    for i in range(n):
        if suppressed[i]:
            continue
        # 当前 i 是高分，保留
        final_masks.append(masks[i])
        final_scores.append(scores[i])
        # 抑制其余
        # 只检查后面的 j > i（矩阵上三角）
        ious = iou_mat[i]
        too_close = ious > iou_thresh
        suppressed = suppressed | too_close  # 把 overlaps > 阈值 的都标记
        suppressed[i] = False  # 保留当前 i，不 suppress 自己
    
    return final_masks, final_scores


def aggregate_by_image_id(ds, image_key="image", mask_key="mask", score_key="score", ref_text_key="ref_text"):
    """
    按 image_id 聚合 dataset，把 mask/score/ref_text 聚合成 list。

    Args:
        ds: 原 Hugging Face Dataset
        image_key: 原 image 字段名
        mask_key: 原 mask 字段名
        score_key: 原 score 字段名
        ref_text_key: 原 ref_text 字段名

    Returns:
        new_ds: 聚合后的 Dataset
    """
    agg_data = defaultdict(lambda: {"image": None, "masks": [], "scores": [], "ref_texts": []})

    for row in ds:
        image_id = row["image_id"]
        if agg_data[image_id]["image"] is None:
            agg_data[image_id]["image"] = row[image_key]
        agg_data[image_id]["masks"].append(row[mask_key])
        agg_data[image_id]["scores"].append(row[score_key])
        agg_data[image_id]["ref_texts"].append(row[ref_text_key])

    # 转成 list of dict
    new_rows = []
    for image_id, vals in agg_data.items():
        new_rows.append({
            "image_id": image_id,
            "image": vals["image"],
            "masks": vals["masks"],
            "scores": vals["scores"],
            "ref_texts": vals["ref_texts"]
        })

    # 定义新的 features
    new_features = Features({
        "image_id": Value("string"),
        "image": Image(),
        "masks": Sequence(Image()),
        "scores": Sequence(Value("float32")),
        "ref_texts": Sequence(Value("string"))
    })

    # 创建新的 Dataset
    new_ds = Dataset.from_list(new_rows, features=new_features)
    return new_ds


def get_qwen_model_and_processor(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_path,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    processor = AutoProcessor.from_pretrained(
        args.qwen_path,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
    )
    return model, processor


def batch_inference(messages, model, processor):
    msgs = [msg['conversations'] for msg in messages]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) \
        for msg in msgs]
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side='left',
        return_tensors="pt",
    ).to(model.device)

    # Batch Inference
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1, # 增加多样性
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False,
    )
    return output_texts


task_prompt = (
    "You are a helpful assistant that describes what a specific object is in an image.\n"
    "Below is an image and a half-transparent overlay mask highlighting the object of interest. "
    "Output the description of the masked region in the image. It has noun attribute, in one sentence.\n"
    "Following the format: \"the highlighted area: <noun description sentence>\""
)

def make_conversation(example):
    example['conversations'] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example['image_with_mask']},
                {"type": "text", "text": task_prompt},
            ],
        }
    ]
    return example



if __name__ == '__main__':
    set_device()
    args = parse_args()
    model, processor = get_sam_model_and_processor(args)

    # 替换成你选择的图像数据集，处理成path list
    image_list = [
        f"{sam3_root}/examples/images/truck.jpg",
        f"{sam3_root}/examples/images/test_image.jpg",
        f"{sam3_root}/examples/images/groceries.jpg",
    ]

    make_batches = lambda lst, bs: [lst[i:i + bs] for i in range(0, len(lst), bs)]
    image_batches = make_batches(image_list, args.batchsize)

    data_list = []
    for batch in tqdm(image_batches, desc=f'Segmenting'):
        inference_state, img_pils = image_batch_preload(batch, processor)

        pts_batch, pts_labels_batch = [], []
        for img_pil in img_pils:
            w, h = img_pil.size
            pts = generate_random_points_per_image((h, w), num_pts=args.num_pts)
            pts_labels = np.ones(pts.shape[:-1])
            pts_batch.append(pts)
            pts_labels_batch.append(pts_labels)

        masks_batch, scores_batch, _ = model.predict_inst_batch(
            inference_state,
            point_coords_batch=pts_batch, 
            point_labels_batch=pts_labels_batch, 
            box_batch=None, 
            multimask_output=True,
        )
        
        for b in range(len(batch)):
            masks = masks_batch[b].reshape(-1, *masks_batch[b].shape[2:])
            scores = scores_batch[b].flatten()
            final_masks, final_scores = filter_and_dedup_masks(
                masks, 
                scores, 
                score_thresh=args.score_thresh,
                iou_thresh=args.iou_thresh,
            )
            if final_masks is not None and final_scores is not None:
                output_masks = [mask_to_img(mask) for mask in final_masks]
                output_scores = final_scores
            
            data_item = {
                'image_id': str(uuid.uuid4()),
                'image_path': batch[b],
                'masks': output_masks,
                'scores': output_scores,
            }
            data_list.append(data_item)
    
    # qwenvl processing  
    data_items = []
    for d in data_list:
        for mask, score in zip(d['masks'], d['scores']):
            image_with_mask = overlay_mask_on_image(d['image_path'], mask)
            data_items.append({
                'image_id': d['image_id'],
                'image_path': d['image_path'],
                'image_with_mask': image_with_mask,
                'mask': mask,
                'score': score,
            })
    messages = [make_conversation(example) for example in tqdm(data_items, desc='Processing Items')]
    
    model, processor = get_qwen_model_and_processor(args)

    make_batches = lambda lst, bs: [lst[i:i + bs] for i in range(0, len(lst), bs)]
    message_batches = make_batches(messages, args.vlm_batchsize)
    results = []
    for batch in tqdm(message_batches, desc=f'Generating'):
        output_texts = batch_inference(batch, model, processor)
        for example, text in zip(batch, output_texts):
            match = re.search(r"the highlighted area:\s*(.+)", text, re.IGNORECASE)
            ref_text = text if match is None else match.group(1).strip()
            results.append({
                'image_id': example['image_id'],
                'image_path': example['image_path'],
                'mask': example['mask'],
                'score': example['score'],
                'text': ref_text,
            })
    
    features = Features({
        "image_id": Value('string'),
        "image_path": Value('string'),
        "mask": Image(),
        "score": Value("float32"),
        "text": Value('string'),
    })
    save_datalist_to_disk(results, features, args.save_root)
