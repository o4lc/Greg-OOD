#!/bin/bash
set -x

#python train_sea.py --id cifar100 --jacobianLoss 1 --batch_size 64 --seed 2 --beta 0.1 --oodMethod energy\
# --arch resnet18 --spt 0 --ept 38400 --singleLrStone 50 --epochs 60 --repr normFeature --sampleTwoWay

#python train_sea.py --id cifar100 --jacobianLoss 1 --batch_size 64 --seed 42 --beta 0.1 --oodMethod energy\
# --arch resnet18 --spt 0 --ept 38400 --singleLrStone 50 --epochs 60 --repr normFeature --sampleTwoWay

python train_sea.py --id cifar100 --jacobianLoss 1 --batch_size 64 --seed 42 --beta 0.1 --oodMethod energy\
 --arch resnet18 --spt 0 --ept 38400 --singleLrStone 50 --epochs 60 --repr normBatch --sampleTwoWay

python train_sea.py --id cifar100 --jacobianLoss 1 --batch_size 64 --seed 42 --beta 0.1 --oodMethod energy\
 --arch wrn40_2 --spt 0 --ept 38400 --singleLrStone 50 --epochs 60 --repr normFeature --sampleTwoWay