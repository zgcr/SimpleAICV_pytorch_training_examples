CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_addr 127.0.1.0 \
    --master_port 10000 \
    ../../../tools/train_interactive_segmentation_distill_encoder_model.py \
    --work-dir ./

# sudo apt install net-tools
# ifconfig
# eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
#         inet 172.17.0.5  netmask 255.255.0.0  broadcast 172.17.255.255
#         ether 02:42:ac:11:00:05  txqueuelen 0  (Ethernet)
#         RX packets 3731119  bytes 12922820808 (12.9 GB)
#         RX errors 0  dropped 0  overruns 0  frame 0
#         TX packets 2294616  bytes 203528253 (203.5 MB)
#         TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

# NCCL_SOCKET_IFNAME 值是 eth0
# master_addr 值是 inet 172.17.0.5

# # master node
# NCCL_SOCKET_IFNAME值通过ifconfig命令查询
# NCCL_SOCKET_IFNAME=eth0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr 172.17.0.5 \
#     --master_port 10000 \
#     ../../../tools/train_interactive_segmentation_distill_encoder_model_multi_node.py \
#     --work-dir ./

# # worker node1
# NCCL_SOCKET_IFNAME值通过ifconfig命令查询
# NCCL_SOCKET_IFNAME=eth0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr 172.17.0.5 \
#     --master_port 10000 \
#     ../../../tools/train_interactive_segmentation_distill_encoder_model_multi_node.py \
#     --work-dir ./