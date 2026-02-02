CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_addr 127.0.1.1 \
    --master_port 10001 \
    ../../tools/train_human_matting_model.py \
    --work-dir ./