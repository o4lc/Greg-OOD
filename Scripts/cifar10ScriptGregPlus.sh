#!/bin/bash
set -x
seeds=(2 42 1234)
#architectures=(densenet101 resnet18 wrn40_2)
architectures=(resnet18 wrn40_2)
for seed in ${seeds[@]}; do
  for architecture in ${architectures[@]}; do
    echo seed: $seed
    echo architecture: $architecture
    python train_sea.py --id cifar10 --jacobianLoss 1 --batch_size 64 --seed $seed --beta 0.1 --oodMethod energy\
     --arch $architecture --spt 0 --ept 10000 --singleLrStone 50 --epochs 60 --sampleTwoWay
  done
done
