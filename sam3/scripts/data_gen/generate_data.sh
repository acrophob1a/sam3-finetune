export HF_ENDPOINT=https://hf-mirror.com/

SAM_PATH=pretrained/sam3/sam3.pt
QWEN_PATH=pretrained/Qwen2.5-VL-7B-Instruct
IMAGE_DIR=datasets/raw_images_train
SAVE_ROOT=datasets/custom0_exp001
COCO_JSON_PATH=$SAVE_ROOT/annotations.json

python sam3/infer/generate_t2m_data.py \
    --sam_path $SAM_PATH \
    --qwen_path $QWEN_PATH \
    --image_dir $IMAGE_DIR \
    --save_root $SAVE_ROOT \
    --batchsize 1 \
    --vlm_batchsize 2 \
    --max_image_size 1536 \
    --num_pts 256 \
    --score_thresh 0.75 \
    --iou_thresh 0.1

python sam3/infer/convert_to_cocoapi.py \
    --data_path $SAVE_ROOT \
    --save_json_path $COCO_JSON_PATH
