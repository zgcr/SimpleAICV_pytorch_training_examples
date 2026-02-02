CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_addr 127.0.1.0 \
    --master_port 10000 \
    ../../tools/train_text_recognition_model.py \
    --work-dir ./