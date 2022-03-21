# -*- coding: utf-8 -*
# --------------------------------------------------------
#   SNNformer Feature Extractor (SFE) - SNN branch
# --------------------------------------------------------

import torch.nn as nn
import torch

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase

thresh_bais = 0.3
# thresh = 0.3  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.2  # decay constants
global thresh

class SpatialGroupEnhance(nn.Module):
    """ Dynamic Spiking Threshold from spatial features"""
    def __init__(self):
        super(SpatialGroupEnhance, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.bias     = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        xn = x * self.avg_pool(x)
        xn = xn.mean(dim=1, keepdim=True)
        entro = torch.mean(xn, dim=0).squeeze()
        h,w = entro.size()
        entro = entro.view(-1)
        max = torch.max(entro)
        min = torch.min(entro)
        entro = (entro - min) / (max-min) * 255
        his = torch.histc(entro, bins=256, min=0, max=255) / (h*w)
        entro_final = torch.sum(his * -torch.log(his + 0.00000001))
        entro_final = entro_final / torch.count_nonzero(his)
        x = self.sig(xn)
        x = torch.mean(x)
        return x + entro_final*10

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike

cfg_cnn = [(6, 64, 2, 0, 11),
           (64, 128, 2, 0, 9),
           (128, 256, 2, 0, 5),
           (64, 128, 1, 1, 3),
           (128, 256, 1, 1, 3)]
# kernel size
cfg_kernel = [147, 70, 33, 31, 31]
cfg_kernel_first = [59, 26, 11, 15, 15]
# fc layer
cfg_fc = [128, 10]

@VOS_BACKBONES.register
@TRACK_BACKBONES.register


class SNN3(ModuleBase):
    r"""
    SNN branch

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self):
        super(SNN3, self).__init__()

        cfg_cnn = [(3, 64, 2, 0, 11),
                   (64, 128, 2, 0, 9),
                   (128, 256, 2, 0, 5),
                   (64, 128, 1, 1, 3),
                   (128, 256, 1, 1, 3)]
        # kernel size
        cfg_kernel = [147, 70, 33, 31, 31]
        cfg_kernel_first = [59, 26, 11, 15, 15]

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_tem = nn.BatchNorm2d(256)
        self.relu_tem = nn.ReLU()

        self.fuse_snn_transfor = nn.Conv2d(out_planes*2, out_planes, kernel_size=1, stride=1, padding=0)
        self.thre_w = SpatialGroupEnhance()
        self.conv33_11 = nn.Conv2d(256, 256, kernel_size=13, stride=2, padding=0)
        self.bn_spa = nn.BatchNorm2d(256)
        self.relu_spa = nn.ReLU()

    def forward(self, input_pos, input_neg, trans_snn, transformer_sig, transformer_fea, first_seq):
        global thresh
        if transformer_fea is None:
            thresh = 0.3
        else:
            thresh = self.thre_w(transformer_fea) * thresh_bais
        if first_seq:
            time_window = len(input_pos)
            tem_c3m = 0
            for step in range(time_window):
                x_pos = input_pos[step]
                x_neg = input_neg[step]
                x = torch.where(x_pos > x_neg, x_pos, x_neg)
                c1_mem, c1_spike = mem_update(self.conv1, x.float(), trans_snn[0], trans_snn[1])
                c2_mem, c2_spike = mem_update(self.conv2, c1_spike, trans_snn[2], trans_snn[3])
                c3_mem, c3_spike = mem_update(self.conv3, c2_spike, trans_snn[4], trans_snn[5])
                trans_snn = [c1_mem, c1_spike, c2_mem, c2_spike, c3_mem, c3_spike]
                tem_c3m = tem_c3m + c3_mem
            tem_fea = tem_c3m / time_window
            tem_fea = self.relu_tem(self.bn_tem(tem_fea))
            spa_fea = self.relu_spa(self.bn_spa(self.conv33_11(transformer_fea)))
            return tem_fea, spa_fea, trans_snn
        else:
            time_window = len(input_pos)
            tem_c3m = 0
            for step in range(time_window):
                x_pos = input_pos[step]
                x_neg = input_neg[step]
                x = torch.where(x_pos > x_neg, x_pos, x_neg)
                c1_mem, c1_spike = mem_update(self.conv1, x.float(), trans_snn[0], trans_snn[1])
                c2_mem, c2_spike = mem_update(self.conv2, c1_spike, trans_snn[2], trans_snn[3])
                c3_mem, c3_spike = mem_update(self.conv3, c2_spike, trans_snn[4], trans_snn[5])
                trans_snn = [c1_mem, c1_spike, c2_mem, c2_spike, c3_mem, c3_spike]
                tem_c3m = tem_c3m + c3_mem
            tem_fea = tem_c3m / time_window
            tem_fea = self.relu_tem(self.bn_tem(tem_fea))
            spa_fea = transformer_fea
            return tem_fea, spa_fea, trans_snn
