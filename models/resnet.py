"""
resnet from the pytorch library

"""

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

class MyResNet(ResNet):
    def __init__(
            self,
            block,
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super(MyResNet, self).__init__(block, layers, num_classes,
            zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

    def forward(self, x, ret_feat=False):
        # based on the resnet model implementation of PyTorch:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logit = self.fc(x)
        if ret_feat:
            return logit, x
        return logit


def resnet(name, numberOfClasses):

    if name == "resnet18":
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif name == "resnet50":
        block = Bottleneck
        layers = [3, 4, 6, 3]
    return MyResNet(block, layers, num_classes=numberOfClasses)

