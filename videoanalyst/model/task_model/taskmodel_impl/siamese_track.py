# -*- coding: utf-8 -*

from loguru import logger

import torch, time

from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
from torch.cuda.amp import autocast as autocast
# import seaborn as sns
import numpy as np
import cv2
import matplotlib.pyplot as plt

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class SiamTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                neck_conv_bias=[True, True, True, True],
                                corr_fea_output=False,
                                trt_mode=False,
                                trt_fea_model_path="",
                                trt_track_model_path="",
                                amp=False)

    support_phases = ["train", "feature", "track", "freeze_track_fea"]

    def __init__(self, backbone, transfor, atten, head, loss=None):
        super(SiamTrack, self).__init__()
        self.basemodel = backbone
        self.transfor = transfor
        self.atten = atten
        self.head = head
        self.loss = loss
        self.trt_fea_model = None
        self.trt_track_model = None
        self._phase = "train"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def train_forward(self, training_data):
        # for snn init state
        cfg_cnn = [(3, 64, 2, 0, 11),
                   (64, 128, 2, 0, 9),
                   (128, 256, 2, 0, 5),
                   (64, 128, 1, 1, 3),
                   (128, 256, 1, 1, 3)]
        cfg_kernel = [147, 70, 33, 31, 31]
        cfg_kernel_first = [59, 26, 11, 15, 15]
        batch_size = training_data['box_gt'].shape[0]
        c1_mem_first = c1_spike_first = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel_first[0], cfg_kernel_first[0])
        c2_mem_first = c2_spike_first = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel_first[1], cfg_kernel_first[1])
        c3_mem_first = c3_spike_first = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel_first[2], cfg_kernel_first[2])
        snn_state_first = [c1_mem_first.cuda(), c1_spike_first.cuda(), c2_mem_first.cuda(), c2_spike_first.cuda(),
                           c3_mem_first.cuda(), c3_spike_first.cuda()]
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0])
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1])
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2])
        snn_state = [c1_mem.cuda(), c1_spike.cuda(), c2_mem.cuda(), c2_spike.cuda(), c3_mem.cuda(), c3_spike.cuda()]

        target_img_pos = training_data["im_z_pos"]
        target_img_neg = training_data['im_z_neg']
        search_img_pos = training_data["im_x_pos"]
        search_img_neg = training_data["im_x_neg"]

        # with autocast():
        # for init transformer
        # trans_z_sig, trans_z_lowres = self.transfor(target_img_pos, target_img_neg)
        # trans_z_sig, trans_z_lowres = None, None
        trans_z_sig, trans_z_lowres = self.transfor(target_img_pos, target_img_neg)
        trans_x_sig, trans_x_lowers = self.transfor(search_img_pos, search_img_neg)

        # backbone feature
        tem_fea_z, spa_fea_z, _ = self.basemodel(target_img_pos, target_img_neg, snn_state_first, trans_z_sig, trans_z_lowres, first_seq=True)
        tem_fea_x, spa_fea_x, _ = self.basemodel(search_img_pos, search_img_neg, snn_state, trans_x_sig, trans_x_lowers, first_seq=False)

        f_z = self.atten(tem_fea_z, spa_fea_z)
        f_x = self.atten(tem_fea_x, spa_fea_x)

        # feature adjustment
        c_z_k = self.c_z_k(f_z)
        r_z_k = self.r_z_k(f_z)
        c_x = self.c_x(f_x)
        r_x = self.r_x(f_x)
        # feature matching
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)
        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
            c_out, r_out)
        predict_data = dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
        )
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] = corr_fea
        return predict_data

    def instance(self, img):
        f_z = self.basemodel(img)
        # template as kernel
        c_x = self.c_x(f_z)
        self.cf = c_x

    def forward(self, *args, phase=None):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        # used for template feature extraction (normal mode)
        elif phase == 'feature':
            target_img_pos, target_img_neg, snn_state_first = args
            if self._hyper_params["trt_mode"]:
                # extract feature with trt model
                out_list = self.trt_fea_model(target_img_pos)
            else:

                trans_z_sig, trans_z_lowres = self.transfor(target_img_pos, target_img_neg)
                # trans_z_sig, trans_z_lowres = self.transfor(tem_pos, tem_neg)
                tem_fea_z, spa_fea_z, _ = self.basemodel(target_img_pos, target_img_neg, snn_state_first, trans_z_sig, trans_z_lowres, first_seq=True)
                f_z = self.atten(tem_fea_z, spa_fea_z)
                # template as kernel
                c_z_k = self.c_z_k(f_z)
                r_z_k = self.r_z_k(f_z)
                # output
                out_list = [c_z_k, r_z_k]
        # used for template feature extraction (trt mode)
        elif phase == "freeze_track_fea":
            search_img, = args
            # backbone feature
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # head
            return [c_x, r_x]
        # [Broken] used for template feature extraction (trt mode)

        elif phase == "freeze_track_head":
            c_out, r_out = args
            # head
            outputs = self.head(c_out, r_out, 0, True)
            return outputs
        # used for tracking one frame during test
        elif phase == 'track':
            if len(args) == 3+2:
                search_img_pos, search_img_neg, snn_state, c_z_k, r_z_k = args
                if self._hyper_params["trt_mode"]:
                    c_x, r_x = self.trt_track_model(search_img_pos)
                else:
                    # backbone feature
                    trans_x_sig, trans_x_lowers = self.transfor(search_img_pos, search_img_neg)
                    tem_fea_x, spa_fea_x, snn_state = self.basemodel(search_img_pos, search_img_neg, snn_state, trans_x_sig, trans_x_lowers, first_seq=False)
                    f_x = self.atten(tem_fea_x, spa_fea_x)
                    c_x = self.c_x(f_x)
                    r_x = self.r_x(f_x)
            elif len(args) == 4+2:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out, search_img_pos[-1].size(-1))
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            # register extra output
            extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
            self.cf = c_x
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra, snn_state
        else:
            raise ValueError("Phase non-implemented.")

        return  out_list
        # if phase == 'feature':
        #     return  out_list
        # if phase == 'track':
        #     return out_list, snn_state

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        self._initialize_conv()
        super().update_params()
        if self._hyper_params["trt_mode"]:
            logger.info("trt mode enable")
            from torch2trt import TRTModule
            self.trt_fea_model = TRTModule()
            self.trt_fea_model.load_state_dict(
                torch.load(self._hyper_params["trt_fea_model_path"]))
            self.trt_track_model = TRTModule()
            self.trt_track_model.load_state_dict(
                torch.load(self._hyper_params["trt_track_model_path"]))
            logger.info("loading trt model succefully")

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
