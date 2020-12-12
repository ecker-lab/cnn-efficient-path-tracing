import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet


class ResNetSpecial(ResNet):
    """ adapted from the original implementation in torchvision: Added a channel_sizes parameter to control network width """
    def __init__(self, block, layers, channel_sizes, num_classes=1000, groups=1, input_channels=3, first_kernel=7,
                 width_per_group=64, norm_layer=None, stride1_layers=(), with_fc=True):
        super().__init__(block, layers, num_classes=num_classes, width_per_group=width_per_group,
                         replace_stride_with_dilation=None)

        self.inplanes = channel_sizes[0]
        self.sizes = channel_sizes
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        overlap = first_kernel - 1
        pad = overlap // 2, overlap - (overlap // 2)

        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=first_kernel, stride=2, padding=pad,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)

        self.layer1 = self._make_layer(block, channel_sizes[0], layers[0])
        self.layer2 = self._make_layer(block, channel_sizes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channel_sizes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channel_sizes[3], layers[3], stride=2)

        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

        if 0 in stride1_layers:
            self.layers[0][0].stride = 1

        for layer_id in set(stride1_layers) - set([0]):
            c, d = self.layers[layer_id][0].conv1, self.layers[layer_id][0].downsample[0]
            self.layers[layer_id][0].conv1 = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, 1,
                                                       c.padding, bias=False)
            self.layers[layer_id][0].downsample[0] = nn.Conv2d(d.in_channels, d.out_channels, d.kernel_size, 1,
                                                               d.padding, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if with_fc:
            self.fc = nn.Linear(channel_sizes[3], num_classes)
            self.flatten = True
        else:
            self.avgpool = nn.Identity()
            self.flatten = False
            self.fc = nn.Identity()


class RN18Shallow(ResNetSpecial):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7, norm_layer=None, with_fc=True):
        super().__init__(BasicBlock, [2, 2, 2, 2], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs, norm_layer=norm_layer, with_fc=with_fc)

    def forward(self, x):
        return super().forward(x),


class RN50Shallow(ResNetSpecial):
    def __init__(self, channels=(16, 32, 64, 128), inp=3, outputs=100, first_kernel=7):
        super().__init__(Bottleneck, [3, 4, 6, 3], channels, input_channels=inp,
                         first_kernel=first_kernel, num_classes=outputs)

    def forward(self, x):
        return super().forward(x),

