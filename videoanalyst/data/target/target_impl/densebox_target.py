# -*- coding: utf-8 -*-

from typing import Dict

from ..target_base import TRACK_TARGETS, TargetBase
from .utils import make_densebox_target


@TRACK_TARGETS.register
class DenseboxTarget(TargetBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        z_size=127,
        x_size=303,
        score_size=17,
        score_offset=87,
        total_stride=8,
        num_conv3x3=3,
    )

    def __init__(self) -> None:
        super().__init__()

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def __call__(self, sampled_data: Dict) -> Dict:
        data_z_pos = sampled_data["data1_pos"]
        im_z_pos, bbox_z = data_z_pos["image"], data_z_pos["anno"]
        data_z_neg = sampled_data["data1_neg"]
        im_z_neg = data_z_neg["image"]

        data_x_pos = sampled_data["data2_pos"]
        im_x_pos, bbox_x = data_x_pos["image"], data_x_pos["anno"]
        data_x_neg = sampled_data["data2_neg"]
        im_x_neg = data_x_neg["image"]

        is_negative_pair = sampled_data["is_negative_pair"]

        # input tensor
        im_z_pos = [im.transpose(2, 0, 1) for im in im_z_pos]
        im_z_neg = [im.transpose(2, 0, 1) for im in im_z_neg]
        im_x_pos = [im.transpose(2, 0, 1) for im in im_x_pos]
        im_x_neg = [im.transpose(2, 0, 1) for im in im_x_neg]

        # training target
        cls_label, ctr_label, box_label = make_densebox_target(
            bbox_x.reshape(1, 4), self._hyper_params)
        if is_negative_pair:
            cls_label[cls_label == 0] = -1
            cls_label[cls_label == 1] = 0

        training_data = dict(
            im_z_pos=im_z_pos,
            im_z_neg=im_z_neg,
            im_x_pos=im_x_pos,
            im_x_neg=im_x_neg,
            bbox_z=bbox_z,
            bbox_x=bbox_x,
            cls_gt=cls_label,
            ctr_gt=ctr_label,
            box_gt=box_label,
            is_negative_pair=int(is_negative_pair),
        )
        #training_data = super().__call__(training_data)

        return training_data
