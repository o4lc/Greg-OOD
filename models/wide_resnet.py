"""
WRN architecture (https://arxiv.org/abs/1605.07146)
Code adapted from (https://github.com/JerryYLi/bg-resample-ood/blob/master/models/wide_resnet.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def weights_init(m):
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        
        if m.bias is not None:
            # multi classification head
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.constant_(m.bias.data, 0.0)
        else:
            # binary classificaiton head
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # nn.init.constant_(m.bias)
            # nn.init.constant_(m.weight.data, 0.0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, in_size=32):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 == 0), 'Wide ResNet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

        # multi classification head
        self.linear = nn.Linear(nStages[3], num_classes)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = self.linear.in_features
        self.pool_size = in_size // 4

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, ret_feat=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        logit = self.linear(out)
        
        if ret_feat:
            return logit, out
        return logit

if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))

    print(y.size())