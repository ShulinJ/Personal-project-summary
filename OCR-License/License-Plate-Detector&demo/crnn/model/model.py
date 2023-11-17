import torch.nn as nn
import torch.nn.functional as F
from .mobilv3 import MobileNetV3
from .ResNet34  import ResNet34
import torch
class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out9
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
class orgin_crnn(nn.Module):
    def __init__(self, imgH, nc , leakyRelu=False):
        super(orgin_crnn, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

    def forward(self, input):

        conv = self.cnn(input)
        return conv

# class CRNN(nn.Module):
    # def __init__(self,cfg):
    #     super(CRNN, self).__init__()
    #     self.imgH = cfg["model_parameter"]["imageH"]
    #     self.nc = 1#cfg["model_parameter"]["input_nc"]
    #     self.nclass = cfg["model_parameter"]["nclass"]
    #     self.cnn_ = cfg["model_parameter"]["backbone"]
    #     self.leakyRelu = cfg["model_parameter"]["leakyRelu"]
    #     self.nh = cfg["model_parameter"]["LSTM_nh"]
    #     if self.cnn_ == "origin":
    #         self.cnn = orgin_crnn(self.imgH, self.nc, leakyRelu=self.leakyRelu)
    #     elif self.cnn_ == "mobilenetv3-large":
    #         self.cnn = MobileNetV3(type='large')
    #     elif self.cnn_ == "mobilenetv3-small":
    #         self.cnn = MobileNetV3(type='small')
    #     elif self.cnn_ ==  "resnet":
    #         self.cnn = ResNet34()
    #     elif self.cnn_ ==  "mobilenetv3-small-chepai":
    #         self.cnn = MobileNetV3(type='small')
    #     self.rnn = nn.Sequential(
    #         BidirectionalLSTM(512, self.nh, self.nh),
    #         BidirectionalLSTM(self.nh, self.nh, self.nclass))
    # def forward(self, input):
    #     conv = self.cnn(input)
    #     b, c, h, w = conv.size()
    #     assert h == 1, "the height of conv must be 1"
    #     conv1 = conv.squeeze(2) # b *512 * width
    #     conv1 = conv1.permute(2, 0, 1)  # [w, b, c]
    #     output = F.log_softmax(self.rnn(conv1), dim=2)
    #     return output

class CRNN(nn.Module):
    def __init__(self,cfg):
        super(CRNN, self).__init__()
        self.imgH = cfg["model_parameter"]["imageH"]
        self.nc = 1#cfg["model_parameter"]["input_nc"]
        self.nclass = cfg["model_parameter"]["nclass"]
        self.cnn_ = cfg["model_parameter"]["backbone"]
        self.leakyRelu = cfg["model_parameter"]["leakyRelu"]
        self.nh = cfg["model_parameter"]["LSTM_nh"]
        if self.cnn_ == "origin":
            self.cnn = orgin_crnn(self.imgH, self.nc, leakyRelu=self.leakyRelu)
        elif self.cnn_ == "mobilenetv3-large":
            self.cnn = MobileNetV3(type='large')
        elif self.cnn_ == "mobilenetv3-small":
            self.cnn = MobileNetV3(type='small')
        elif self.cnn_ ==  "resnet":
            self.cnn = ResNet34()
        elif self.cnn_ ==  "mobilenetv3-small-chepai":
            self.cnn = MobileNetV3(type='small-chepai')
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, self.nh, self.nh),
            BidirectionalLSTM(self.nh, self.nh, self.nclass))
    def forward(self, input):
        conv = self.cnn(input)
        # print(conv.shape)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv1 = conv.squeeze(2) # b *512 * width
        conv1 = conv1.permute(2, 0, 1)  # [w, b, c]

        return F.log_softmax(conv1, dim=2)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn(config,converter):

    model = CRNN(config)
    model.apply(weights_init)

    return model