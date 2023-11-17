import argparse

import cv2
import numpy as np
import torch
from sklearn import preprocessing
# from backbones import get_model

# from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']
using_ckpt = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = torch.nn.functional.interpolate(x, size=(112, 112), scale_factor=None, mode='bilinear',align_corners=None, recompute_scale_factor=None)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)

def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()

def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    feat = net(img.to("cuda"))
    feat = feat.cpu().detach().numpy()
    return feat
def score(img1,img2,net):
    feats1 = net(img1.to("cuda"))
    feats1 = feats1.cpu().detach().numpy()
    feats2 = net(img2.to("cuda"))
    feats2 = feats2.cpu().detach().numpy()
    p1 = feats1 / np.sqrt(np.sum(feats1 ** 2, -1, keepdims=True))
    p2 = feats2 / np.sqrt(np.sum(feats2 ** 2, -1, keepdims=True))
    similarity_score = np.sum(p1 * p2, -1)
    score = similarity_score.flatten()
    return (score+1)/2


import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='backbone.pth')
    parser.add_argument('--img1', type=str, default=r"C:\Users\deng\Desktop\rgb1\id_7_21.bmp")
    parser.add_argument('--img2', type=str, default=r"C:\Users\deng\Desktop\rgb1\id_7_51.bmp")
    args = parser.parse_args()
    input_img_path = r"C:\Users\deng\Desktop\input\ir"
    dirs = os.walk(input_img_path)
    a=0
    b=0
    net = get_model(args.network, fp16=False).to("cuda")
    net.load_state_dict(torch.load(args.weight))
    net.eval()
    file2 = open(r"C:\Users\deng\Desktop\test-HFR\all1.txt", "w")
    # r"C:\Users\deng\Desktop\test-HFR\var-far\{}_IR.png".format()
    list1 = []
    list2 = []
    A = os.listdir(r"C:\Users\deng\Desktop\test-HFR\var-far")
    lllll=r"C:\Users\deng\Desktop\test-HFR\var-far"
    for i in A:
        if "RGB" in i:
            img1 = os.path.join(lllll, i)
            feats1 = inference(net, img1)
            max1=0
            p1 = feats1 / np.sqrt(np.sum(feats1 ** 2, -1, keepdims=True))
            for j in A :
                # print("kaishi")
                if "IR" in j and  i.split("_")[0] !=j.split("_")[0]:
                    img2 = os.path.join(lllll, j)
                    feats2 = inference(net, img2)

                    p2 = feats2 / np.sqrt(np.sum(feats2 ** 2, -1, keepdims=True))
                    similarity_score = np.sum(p1 * p2, -1)
                    score = similarity_score.flatten()
                    if score> max1:
                        max1=score
                        str=r"C:\Users\deng\Desktop\test-HFR\var-far\{}_IR.png,C:\Users\deng\Desktop\test-HFR\var-far\{}_RGB.png,0".format(j,i)+"\n"
            print(max1)
            file2.write(str)



    # for roots, _, files in dirs:
    #     for file in files:
    #         img1=os.path.join(roots,file)
    #         img2=os.path.join(r"C:\Users\deng\Desktop\RGB",file.replace("_IR","").replace("RGB","IR").replace("jpg","png"))#)
    #         try:
    #             feats1=inference(net, img1)
    #             feats2=inference(net, img2)
    #         except:
    #             print(file)
    #         p1 = feats1 / np.sqrt(np.sum(feats1 ** 2, -1, keepdims=True))
    #         p2 = feats2 / np.sqrt(np.sum(feats2 ** 2, -1, keepdims=True))
    #         similarity_score = np.sum(p1 * p2, -1)
    #         score = similarity_score.flatten()
    #         a+=(score+1)/2
    #         print((score+1)/2)
    #         b+=1
    # print(a/b)
