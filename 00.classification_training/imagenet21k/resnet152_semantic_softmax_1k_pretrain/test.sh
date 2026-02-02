CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_addr 127.0.1.1 \
    --master_port 10001 \
    ../../../tools/test_imagenet21k_classification_model.py \
    --work-dir ./