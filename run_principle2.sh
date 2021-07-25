cd CIFAR10
GPUS=`nvidia-smi -L | wc -l`
DEFAULT_LR=0.025
LR=`echo "scale=4; sqrt($GPUS) * $DEFAULT_LR" | bc`
# Train from Scratch
PORT=$RANDOM
echo "random port $PORT"
WORLD_SIZE=$GPUS python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$GPUS train.py --auxiliary --batch_size=96 --init_ch=32 --layer=24 --model_ema --model-ema-decay 0.999 --auto_augment --epochs 600 --name="principle2" --lr=$LR --mixup --cutout --no_bias_decay --smoothing 0.1 --arch='[2, 1, 0, 2, 2, 2, 0, 3, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 1, 3, 0, 1, 1, 4, 5, 6, 3, 5]' >"../OUTPUT/training_log.txt" 2>&1
# Finetune
PORT=$RANDOM
echo "random port $PORT"
WORLD_SIZE=$GPUS python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$GPUS train.py --auxiliary --batch_size=96 --init_ch=32 --layer=24 --model_ema --model-ema-decay 0.999 --auto_augment --epochs 600 --name="principle2" --lr=$LR --mixup --cutout --no_bias_decay --smoothing 0.1 --arch='[2, 1, 0, 2, 2, 2, 0, 3, 2, 2, 2, 2, 3, 3, 0, 0, 0, 0, 1, 3, 0, 1, 1, 4, 5, 6, 3, 5]' --finetune 100 >"../OUTPUT/finetuning_log.txt" 2>&1
cd ".."