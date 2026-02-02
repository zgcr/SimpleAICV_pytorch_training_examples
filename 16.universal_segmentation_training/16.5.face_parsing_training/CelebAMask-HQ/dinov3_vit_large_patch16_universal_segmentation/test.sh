CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_addr 127.0.1.0 \
    --master_port 10000 \
    ../../../../tools/test_universal_segmentation_model_for_face_parsing_dataset.py \
    --work-dir ./
