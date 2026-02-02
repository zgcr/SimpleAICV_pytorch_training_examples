CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_addr 127.0.1.0 \
    --master_port 10000 \
    ../../../tools/train_classification_model.py \
    --work-dir ./