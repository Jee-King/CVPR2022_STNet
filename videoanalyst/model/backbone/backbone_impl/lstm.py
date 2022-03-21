# -*- coding: utf-8 -*

import torch.nn as nn
import torch

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase


