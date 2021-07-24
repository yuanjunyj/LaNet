## Training Script

```bash
CUDA_VISIBLE_DEVICES=9 python train.py --auxiliary --batch_size=64 --init_ch=32 --layer=24 --arch='[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]' --model_ema --model-ema-decay 0.9999 --auto_augment --epochs 600 --finetune 200
```
1. 由于batch_size按照他所给的参数只开到了64，所以单卡是能放的下的。
2. 我加了一个finetune的参数，代表的就是finetune模型的轮数；epochs表示的是原始的模型训练的轮数。最终的scheduler是按照两个轮次相加计算的学习率衰减。
3. Line 224-225是决定加载哪一个模型，train.pt是600轮之后得到的模型；finetune_model.pt是finetune过程中得到的最好的模型。目前我是先在model.pt的模型上finetune了100轮，得到一个finetune_model.pt，然后再不断的做增量式的调优。（实际上就是每一次去增大finetune的轮次）
