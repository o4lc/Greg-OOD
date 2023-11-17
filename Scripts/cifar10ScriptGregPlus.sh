#!/bin/bash
set -x
#seeds=(2 42 1234)
#architectures=(densenet101 resnet18 wrn40_2)
#for seed in ${seeds[@]}; do
#  for architecture in ${architectures[@]}; do
#    echo seed: $seed
#    echo architecture: $architecture
#    python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed $seed --beta 0.1 --oodMethod energy\
#     --arch $architecture --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay
#  done
#done

python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed 2 --beta 0.1 --oodMethod energy\
     --arch resnaet18 --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay

python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed 2 --beta 0.1 --oodMethod energy\
     --arch wrn40_2 --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay

python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed 42 --beta 0.1 --oodMethod energy\
     --arch densenet101 --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay

python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed 42 --beta 0.1 --oodMethod energy\
     --arch resnet18 --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay

python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed 42 --beta 0.1 --oodMethod energy\
     --arch wrn40_2 --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay

