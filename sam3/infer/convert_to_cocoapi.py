import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default='datasets/custom0',
        help="data path",
    )
    parser.add_argument(
        "--save_json_path",
        type=str,
        default='datasets/custom0',
        help="directory of saved images",
    )
    args = parser.parse_args()
    return args


def mask_to_polygon_bbox_area(mask_pil):
    """
    输入:
        mask_pil: PIL Image (binary mask)

    返回:
        segmentation, bbox, area
    """
    mask = np.array(mask_pil)
    
    # 保证是0/1
    mask = (mask > 0).astype(np.uint8)

    # 面积
    area = float(mask.sum())

    # 找轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segmentation = []

    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour.squeeze(1)
        poly = contour.flatten().tolist()

        if len(poly) >= 6:
            segmentation.append(poly)

    # bbox
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        bbox = [0, 0, 0, 0]
    else:
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        bbox = [
            int(x_min),
            int(y_min),
            int(x_max - x_min),
            int(y_max - y_min),
        ]

    return segmentation, bbox, area


def convert_to_coco(dataset, save_json_path):
    """
    dataset: huggingface dataset or list of dict
    output_json: 输出路径
    """

    coco = {
        "info": {"description": "example image dataset"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }

    image_id_map = {}
    image_counter = 0
    ann_counter = 0

    for item in dataset:

        image_path = item["image_path"]
        image_id_str = item["image_id"]

        # 如果这张图没出现过
        if image_id_str not in image_id_map:

            img = Image.open(image_path)
            width, height = img.size

            coco["images"].append(
                {
                    "id": image_counter,
                    "file_name": image_path,
                    "width": width,
                    "height": height,
                }
            )

            image_id_map[image_id_str] = image_counter
            image_counter += 1

        image_id = image_id_map[image_id_str]

        mask = item["mask"]

        segmentation, bbox, area = mask_to_polygon_bbox_area(mask)

        ann = {
            "id": ann_counter,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "noun_phrase": item["text"],
        }

        coco["annotations"].append(ann)

        ann_counter += 1

    with open(save_json_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"Saved to {save_json_path}")



if __name__ == '__main__':
    args = parse_args()

    data_list = load_from_disk(args.data_path)
    convert_to_coco(data_list, args.save_json_path)