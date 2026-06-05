export HF_ENDPOINT=https://hf-mirror.com/

SAM_PATH=pretrained/sam3/sam3.pt
QWEN_PATH=pretrained/Qwen2.5-VL-7B-Instruct
SAVE_ROOT=datasets/custom1
COCO_JSON_PATH=$SAVE_ROOT/annotations.json

python sam3/infer/generate_t2m_data.py \
    --sam_path $SAM_PATH \
    --qwen_path $QWEN_PATH \
    --save_root $SAVE_ROOT \
    --batchsize 4 \
    --vlm_batchsize 4 \
    --num_pts 512 \
    --score_thresh 0.75 \
    --iou_thresh 0.1 \

python sam3/infer/convert_to_cocoapi.py \
    --data_path $SAVE_ROOT \
    --save_json_path $COCO_JSON_PATH \
