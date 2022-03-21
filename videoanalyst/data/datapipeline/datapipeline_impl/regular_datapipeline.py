# -*- coding: utf-8 -*-
from typing import Dict, List

from videoanalyst.utils import convert_numpy_to_tensor

from ...sampler.sampler_base import SamplerBase
from ..datapipeline_base import (TRACK_DATAPIPELINES, VOS_DATAPIPELINES,
                                 DatapipelineBase)


@TRACK_DATAPIPELINES.register
@VOS_DATAPIPELINES.register
class RegularDatapipeline(DatapipelineBase):
    r"""
    Tracking datapipeline. Integrate sampler togethor with a list of processes

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict()

    def __init__(
            self,
            sampler: SamplerBase,
            pipeline: List = [],
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.pipeline = pipeline

    def __getitem__(self, item) -> Dict:
        r"""
        An interface to load batch data
        """
        sampled_data = self.sampler[item]

        for proc in self.pipeline:      # 1. crop dataset  2. get label: cls_res, center_res, gt_boxes_res
            sampled_data = proc(sampled_data)
        sampled_data = convert_numpy_to_tensor(sampled_data)
        return sampled_data
