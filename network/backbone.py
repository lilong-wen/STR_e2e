# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .positional_encoding import build_position_encoding


from .chk import checkpoint_sequential_step, checkpoint
import math
import numpy as np
from torchvision.utils import save_image

import gin

def ginM(n): return gin.query_parameter(f'%{n}')
gin.external_configurable(nn.MaxPool2d, module='nn')
gin.external_configurable(nn.Upsample,  module='nn')


class LN(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

@gin.configurable
class PadPool(nn.Module):
    def forward(self, x):
        x = F.pad(x, [0, 0, 0, 1])
        x = F.max_pool2d(x,(2, 2), stride=(1, 2))
        return x

def pCnv(inp,out,groups=1):
  return nn.Sequential(
      nn.Conv2d(inp,out,1,bias=False,groups=groups),
      nn.InstanceNorm2d(out,affine=True)
  )

#regarding same padding in PT https://github.com/pytorch/pytorch/issues/3867
def dsCnv(inp,k):
  return nn.Sequential(
      nn.Conv2d(inp,inp,k,groups=inp,bias=False,padding=(k - 1) // 2),
      nn.InstanceNorm2d(inp,affine=True)
  )

ngates = 2

class Gate(nn.Module):
    def __init__(self,ifsz):
        super().__init__()
        self.ln = LN()

    def forward(self, x):
        t0,t1 = torch.chunk(x, ngates, dim=1)
        # t0 = torch.tanh_(t0)
        t0 = torch.tanh(t0)
        # t1.sub_(2)
        t1 = t1.sub(2)
        # t1 = torch.sigmoid_(t1)
        t1 = torch.sigmoid(t1)

        return t1*t0

def customGC(module):
    def custom_forward(*inputs):
        inputs = module(inputs[0])
        return inputs
    return custom_forward

@gin.configurable
class GateBlock(nn.Module):
    def __init__(self, ifsz, ofsz, gt = True, ksz = 3, GradCheck=gin.REQUIRED):
        super().__init__()

        cfsz   = int( math.floor(ifsz/2) )
        ifsz2  = ifsz + ifsz%2

        self.sq = nn.Sequential(
          pCnv(ifsz, cfsz),
          dsCnv(cfsz,ksz),
          nn.ELU(),
          ###########
          pCnv(cfsz, cfsz*ngates),
          dsCnv(cfsz*ngates,ksz),
          Gate(cfsz),
          ###########
          pCnv(cfsz, ifsz),
          dsCnv(ifsz,ksz),
          nn.ELU(),
        )

        self.gt = gt
        self.gc = GradCheck


    def forward(self, x):
        if self.gc >= 1:
            y = checkpoint(customGC(self.sq), x)
        else:
          y = self.sq(x)

        out = x + y
        return out

@gin.configurable
class InitBlock(nn.Module):
    def __init__(self, fup, n_channels):
        super().__init__()

        self.n1 = LN()
        self.Initsq = nn.Sequential(
          pCnv(n_channels, fup),
          nn.Softmax(dim=1),
          dsCnv(fup,11),
          LN()
        )

    def forward(self, x):
        x  = self.n1(x)
        xt = x
        x  = self.Initsq(x)
        x  = torch.cat([x,xt],1)
        return x

@gin.configurable
class OrigamiNet(nn.Module):
    def __init__(self, n_channels, o_classes, wmul, lreszs, lszs, nlyrs, fup, GradCheck, reduceAxis=3):
        super().__init__()

        self.lreszs = lreszs
        self.Initsq = InitBlock(fup)

        layers = []
        isz = fup + n_channels
        osz = isz
        for i in range(nlyrs):
            osz = int( math.floor(lszs[i] * wmul) ) if i in lszs else isz
            layers.append( GateBlock(isz, osz, True, 3) )

            if isz != osz:
              layers.append( pCnv(isz, osz) )
              layers.append( nn.ELU() )
            isz = osz

            if i in lreszs:
              layers.append( lreszs[i] )

        layers.append( LN() )
        self.Gatesq = nn.Sequential(*layers)

        self.Finsq = nn.Sequential(
          pCnv(osz, o_classes),
          nn.ELU(),
        )

        self.n1 = LN()
        self.it=0
        self.gc = GradCheck
        self.reduceAxis = reduceAxis

    def forward(self, x, t=[]):
        x = self.Initsq(x)

        if self.gc >=2:
          x = checkpoint_sequential_step(self.Gatesq,4,x)  #slower, more memory save
          # x = checkpoint_sequential_step(self.Gatesq,8,x)  #faster, less memory save
        else:
          x = self.Gatesq(x)

        x = self.Finsq(x)

        x = torch.mean(x, self.reduceAxis, keepdim=False)
        x = self.n1(x)
        x = x.permute(0,2,1)

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.backbone = backbone
        self.position_embedding = position_embedding

    def forward(self, x_input):
        # out = self[0](x_input)
        out = self.backbone(x_input)
        # pos = self[1](out).to(out.dtype)
        pos = self.position_embedding(out).to(out.dtype)
        pos = pos.repeat(out.shape[0], 1, 1)
        return out, pos

@gin.configurable
def build_backbone():
    position_embedding = build_position_encoding()
    backbone = OrigamiNet()
    # backbone = OrigamiNet()
    model = Joiner(backbone, position_embedding)

    return model
