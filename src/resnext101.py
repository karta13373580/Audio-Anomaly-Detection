import torch
import torch.nn as nn
from torchvision import models
# from resnest.torch import resnest101

class resnext_wsl(torch.nn.Module):

    def __init__(self, gradient=False):
        super(resnext_wsl, self).__init__()

        layer = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        block1_conv1 = layer.conv1
        block1_bn1 = layer.bn1
        block1_relu = layer.relu
        block1_maxpool = layer.maxpool

        block2 = layer.layer1
        block3 = layer.layer2
        block4 = layer.layer3
        block5 = layer.layer4   # feature layers

        # change channel
        self.conv1x1 = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)

        # hierarchy 1 (level 1)
        self.conv1_1 = block1_conv1
        self.bn1_1 = block1_bn1
        self.relu1_1 = block1_relu
        self.maxpool1_1 = block1_maxpool

        # hierarchy 2 (level 2)
        self.bolck2_1 = block2[0]
        self.bolck2_2 = block2[1]
        self.bolck2_3 = block2[2]

        # hierarchy 3 (level 3)
        self.bolck3_1 = block3[0]
        self.bolck3_2 = block3[1]
        self.bolck3_3 = block3[2]
        self.bolck3_4 = block3[3]

        # hierarchy 4 (level 4)
        self.bolck4_1 = block4[0]
        self.bolck4_2 = block4[1]
        self.bolck4_3 = block4[2]
        self.bolck4_4 = block4[3]
        self.bolck4_5 = block4[4]
        self.bolck4_6 = block4[5]
        self.bolck4_7 = block4[6]
        self.bolck4_8 = block4[7]
        self.bolck4_9 = block4[8]
        self.bolck4_10 = block4[9]
        self.bolck4_11 = block4[10]
        self.bolck4_12 = block4[11]
        self.bolck4_13 = block4[12]
        self.bolck4_14 = block4[13]
        self.bolck4_15 = block4[14]
        self.bolck4_16 = block4[15]
        self.bolck4_17 = block4[16]
        self.bolck4_18 = block4[17]
        self.bolck4_19 = block4[18]
        self.bolck4_20 = block4[19]
        self.bolck4_21 = block4[20]
        self.bolck4_22 = block4[21]
        self.bolck4_23 = block4[22]

        # hierarchy 5 (level 5)
        self.bolck5_1 = block5[0]
        self.bolck5_2 = block5[1]
        self.bolck5_3 = block5[2]

        # don't need the gradients, just want the features
        if not gradient:
            for param in self.parameters():
                param.requires_grad = False

        # self.pad = nn.ReflectionPad2d(padding=1)

    def forward(self, x, feature_layers):

        # hierarchy 1 (level 1)
        x = self.conv1x1(x)
        conv1_1 = self.conv1_1(x)
        bn1_1 = self.bn1_1(conv1_1)
        relu1_1 = self.relu1_1(bn1_1)
        maxpool1_1 = self.maxpool1_1(relu1_1)

        # hierarchy 2 (level 2)
        bolck2_1 = self.bolck2_1(maxpool1_1)
        bolck2_2 = self.bolck2_2(bolck2_1)
        bolck2_3 = self.bolck2_3(bolck2_2)

        # hierarchy 3 (level 3)
        bolck3_1 = self.bolck3_1(bolck2_3)
        bolck3_2 = self.bolck3_2(bolck3_1)
        bolck3_3 = self.bolck3_3(bolck3_2)
        bolck3_4 = self.bolck3_4(bolck3_3)

        # hierarchy 4 (level 4)
        bolck4_1 = self.bolck4_1(bolck3_4)
        bolck4_2 = self.bolck4_2(bolck4_1)
        bolck4_3 = self.bolck4_3(bolck4_2)
        bolck4_4 = self.bolck4_4(bolck4_3)
        bolck4_5 = self.bolck4_5(bolck4_4)
        bolck4_6 = self.bolck4_6(bolck4_5)
        bolck4_7 = self.bolck4_7(bolck4_6)
        bolck4_8 = self.bolck4_8(bolck4_7)
        bolck4_9 = self.bolck4_9(bolck4_8)
        bolck4_10 = self.bolck4_10(bolck4_9)
        bolck4_11 = self.bolck4_11(bolck4_10)
        bolck4_12 = self.bolck4_12(bolck4_11)
        bolck4_13 = self.bolck4_13(bolck4_12)
        bolck4_14 = self.bolck4_14(bolck4_13)
        bolck4_15 = self.bolck4_15(bolck4_14)
        bolck4_16 = self.bolck4_16(bolck4_15)
        bolck4_17 = self.bolck4_17(bolck4_16)
        bolck4_18 = self.bolck4_18(bolck4_17)
        bolck4_19 = self.bolck4_19(bolck4_18)
        bolck4_20 = self.bolck4_20(bolck4_19)
        bolck4_21 = self.bolck4_21(bolck4_20)
        bolck4_22 = self.bolck4_22(bolck4_21)
        bolck4_23 = self.bolck4_23(bolck4_22)

        # hierarchy 5 (level 5)
        bolck5_1 = self.bolck5_1(bolck4_23)
        bolck5_2 = self.bolck5_2(bolck5_1)
        bolck5_3 = self.bolck5_3(bolck5_2)

        out = {
            'relu1_1' : relu1_1,
            'bolck2_3': bolck2_3,
            'bolck3_4': bolck3_4,
            'bolck4_23': bolck4_23,
        }
        return dict((key, value) for key, value in out.items() if key in feature_layers)
