#!/bin/bash
set -x
seeds=(2 42 1234)
#architectures=(densenet101 resnet18 wrn40_2)
architectures=(resnet18 wrn40_2)
for seed in ${seeds[@]}; do
  for architecture in ${architectures[@]}; do
    echo seed: $seed
    echo architecture: $architecture
    python train_sea.py --id cifar10 --jacobianLoss 0 --batch_size 64 --seed $seed --beta 1 --oodMethod abs --arch $architecture
  done
done
#python train_sea.py --id cifar10 --jacobianLoss 0 --batch_size 64 --seed 42 --beta 1 --oodMethod abs --arch densenet101
#python train_sea.py --id cifar10 --jacobianLoss 0 --batch_size 64 --seed 42 --beta 1 --oodMethod abs --arch resnet18
#python train_sea.py --id cifar10 --jacobianLoss 0 --batch_size 64 --seed 42 --beta 1 --oodMethod abs --arch wrn40_2
