# DataParallel mode train.sh. config.distributed=False
# CUDA_VISIBLE_DEVICES=0,1 python ../../../tools/train_detection_model.py --work-dir ./
# DistributedDataParallel mode train.sh. config.distributed=True
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 20001 ../../../tools/train_detection_model.py --work-dir ./