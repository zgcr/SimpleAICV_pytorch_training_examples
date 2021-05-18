# DataParallel mode test.sh. config.distributed=False
# CUDA_VISIBLE_DEVICES=0 python ../../../tools/test_classification_model.py --work-dir ./
# DistributedDataParallel mode test.sh. config.distributed=True
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 20001 ../../../tools/test_classification_model.py --work-dir ./