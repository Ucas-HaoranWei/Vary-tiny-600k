python -m torch.distributed.run --nproc_per_node=8 --master_port=29501 train.py --cfg-path lavis/projects/varytiny/train/pretrain.yaml


python -m torch.distributed.run --master_addr $MLP_WORKER_0_HOST --master_port $MLP_WORKER_0_PORT --node_rank $MLP_ROLE_INDEX --nnodes $MLP_WORKER_NUM --nproc_per_node $MLP_WORKER_GPU  train.py --cfg-path lavis/projects/varytiny/train/pretrain.yaml