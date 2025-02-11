# master node
NCCL_SOCKET_IFNAME=eth0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.230.31.63 --master_port=10000 ../../../tools/train_interactive_segmentation_distill_encoder_model_multi_node.py --work-dir ./
# # worker node1
# NCCL_SOCKET_IFNAME=eth0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=10.230.31.63 --master_port=10000 ../../../tools/train_interactive_segmentation_distill_encoder_model_multi_node.py --work-dir ./