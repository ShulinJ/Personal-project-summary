import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, dowansample=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.dowansample=dowansample
    def forward(self, x):
        out = self.left(x)
        residual = x if self.dowansample is None else self.dowansample(x)
        out += residual
        return F.relu(out)

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2 ,3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1)
        )
        self.layer1 = self._make_layer(64, 128, 3 ,stride=(2,1))              ### 3 个 64 通道的残差单元，输出 128通道，共6层
        self.layer2 = self._make_layer(128, 256, 4, stride=2)   ### 4 个 128通道的残差单元，输出 256通道，共8层
        self.layer3 = self._make_layer(256, 512, 6, stride=(2,1))   ### 6 个 256通道的残差单元，输出 512通道，共12层
        self.layer4 = self._make_layer(512, 512, 3, stride=(2,1))   ### 3 个 512通道的残差单元，输出 512通道，共6层

    def _make_layer(self, inchannel, outchannel, block_num, stride):
        dowansample= nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, dowansample))       ### 先来一个残差单元，主要是改变通道数
        for i in range(1, block_num+1):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
    def forward(self, x):
        ### 第1层
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
