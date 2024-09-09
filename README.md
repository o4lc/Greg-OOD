# Introduction
This repository contains code for our experiments in [Gradient-Regularized Out-of-Distribution Detection](https://arxiv.org/pdf/2404.12368).

We based our code on an initial implementation of the work of [DOS: DIVERSE OUTLIER SAMPLING FOR OUT-OFDISTRIBUTION DETECTION](https://arxiv.org/pdf/2306.02031).


# Getting Started
The `requirements.txt` file contains the required python libraries. We used `Python 3.11.5`.

The training is done via the `train_sea.py` file. 
Here we present three sample training commands on the ImageNet dataset with 20 in distribution classes.:
1. DOS:
```
python train_sea.py --jacobianLoss 0 --batch_size 64 --seed 62 --beta 1 --oodMethod abs\
 --spt 16 --ept 384 --arch densenet121 --epochs 100 --id imagenet --size_candidate_ood 600000\
  --gpu_idx 0 --lr 0.01 --save_freq 10 --imagenetNumberOfClasses 20 --prefetch 2
```
2. GReg (no sampling is involved):
```
python train_sea.py --jacobianLoss 1 --batch_size 64 --seed 62 --beta 0.1 --oodMethod energy\
--arch densenet121 --epochs 100 --id imagenet --noCluster\
  --gpu_idx 0 --lr 0.01 --save_freq 10 --imagenetNumberOfClasses 20 --prefetch 2
```
3. GReg+ training, starting from pretrained weights:
```
python train_sea.py --jacobianLoss 1 --batch_size 32 --seed 62 --beta 0.1 --oodMethod energy\
 --spt 0 --ept 10000 --arch densenet121 --epochs 25 --id imagenet --size_candidate_ood 1000000 --sampleTwoWay\
  --gpu_idx 0 --lr 0.0001 --singleLrStone 10 --save_freq 5 --imagenetNumberOfClasses 20\
  --finetune --pretrainFile models/densenet121_20.pth --prefetch 2
```


# Citation
If you find our repository useful, consider citing it as follows:

    @article{sharifi2024gradient,
    title={Gradient-Regularized Out-of-Distribution Detection},
    author={Sharifi, Sina and Entesari, Taha and Safaei, Bardia and Patel, Vishal M and Fazlyab, Mahyar},
    journal={arXiv preprint arXiv:2404.12368},
    year={2024}
    }