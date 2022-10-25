# A correct and dabnn-compatible PyTorch implementation of binary convolutions.
# It consists of a implementation of the binary convolution itself, and the way
# to make the implementation both ONNX- and dabnn-compatible
# 1. The input of binary convolutions should only be +1/-1, so we pad -1 instead
#    of 0 by a explicit pad operation.
# 2. Since PyTorch doesn't support exporting Sign ONNX operator (until 
#    https://github.com/pytorch/pytorch/pull/20470 gets merged), we perform sign 
#    operation on input and weight by directly accessing the `data`

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class SignSTE(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input.ge(-1) & input.le(1)
        grad_input = torch.where(
            mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class SignWeight(Function):
    @staticmethod
    def forward(ctx, input):
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(grad_output.size())
        grad_input.copy_(grad_output)
        return grad_input


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(BinarizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)

    def forward(self, input):
        if self.training:
            input = SignSTE.apply(input)
            self.weight_bin_tensor = SignWeight.apply(self.weight)
        else:
            # We clone the input here because it causes unexpected behaviors 
            # to edit the data of `input` tensor.
            input = input.clone()
            input.data = input.sign()
            # Even though there is a UserWarning here, we have to use `new_tensor`
            # rather than the "recommended" way
            self.weight_bin_tensor = self.weight.new_tensor(self.weight.sign())

        # 1. The input of binary convolution shoule be only +1 or -1, 
        #    so instead of padding 0 automatically, we need pad -1 by ourselves
        # 2. `padding` of nn.Conv2d is always a tuple of (padH, padW), 
        #    while the parameter of F.pad should be (padLeft, padRight, padTop, padBottom)
        input = F.pad(input, (self.padding[0], self.padding[0],
                              self.padding[1], self.padding[1]), mode='constant', value=-1)
        print("self.weight_bin_tensor  : ",self.weight_bin_tensor.shape)
        print("input  : ",input.shape)
        inp_shape = input.shape
        # out = torch.zeros((inp_shape[0], ,inp_shape[1],self.weight_bin_tensor.shape[3]*self.weight_bin_tensor.shape[4], inp_shape[2], inp_shape[3]))
        # for iii in range(3):
        #     for jjj in range(3):
        #         out[:,:,iii,jjj,:] = input[:,:,iii,jjj,:] * self.weight_bin_tensor[:,:,iii,jjj,:]

        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride,
                       0, self.dilation, self.groups)

        return out