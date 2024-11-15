CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_addr 127.0.1.0 --master_port 10000 ../../../tools/test_human_parsing_model.py --work-dir ./
