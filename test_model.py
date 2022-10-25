import os
import torch
import torch.nn as nn

# from binary_utils import BinarizeConv2d
from binary_f_utils import BinarizeConv2d
from thop import profile
from copy import deepcopy


class Test(nn.Module):

    def __init__(self, in_channels = 3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False,
                 downsample=None):
        super(Test, self).__init__()
        # self.binary_conv = BinarizeConv2d(in_channels, out_channels,
        conv_func  = BinarizeConv2d
        # conv_func  = nn.Conv2d
        self.conv = conv_func(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv1 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv3 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv4 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv5 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv6 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv7 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv8 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv9 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv10 = conv_func(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        # residual = x
        # x = self.conv1(self.conv2(self.conv3(self.conv4(self.conv5(self.conv6(self.conv7(self.conv8(self.conv9(self.conv10(self.conv(x)))))))))))
        x = self.conv(x)

        # if self.downsample is not None:
        #     residual = self.downsample(residual)
        # x += residual

        return x
def get_model_info(model: nn.Module) -> str:
    stride = 224
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=True)
    params /= 1e6
    flops /= 1e9
    return params, flops
if __name__ == "__main__":
    from torchsummary import torchsummary

    aa = torch.bitwise_xor(torch.tensor([True, True, False]), torch.tensor([False, True, False]))
    # print(aa)
    iii = torch.randn((1,3,256,256))
    # print(iii)
    model = Test().eval()
    re = model(iii)
    # print(re)
    # print(model)
    # print(get_model_info(model))
    # torchsummary.summary(model, (3, 32, 32))
