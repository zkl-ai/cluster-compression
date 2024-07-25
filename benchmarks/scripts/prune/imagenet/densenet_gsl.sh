OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=4 --master_port 18119 --use_env main_imagenet.py --model densenet121 --epochs 90 --batch-size 512 --lr-step-size 30 --lr 0.08 --prune --method group_sl --global-pruning --soft-keeping-ratio 0.25 --pretrained --output-dir run/imagenet/densenet121_sl --target-flops 1.38  --sl-epochs 30 --sl-lr 0.08 --sl-lr-step-size 10 --cache-dataset --reg 1e-4 --print-freq 100 --workers 16 --amp