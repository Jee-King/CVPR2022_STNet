# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from videoanalyst.data.dataset.dataset_base import DatasetBase
from videoanalyst.utils import load_image

from ..sampler_base import TRACK_SAMPLERS, VOS_SAMPLERS, SamplerBase


@TRACK_SAMPLERS.register
@VOS_SAMPLERS.register
class TrackPairSampler(SamplerBase):
    r"""
    Tracking data sampler
    Sample procedure:
    __getitem__
    │
    ├── _sample_track_pair
    │   ├── _sample_dataset
    │   ├── _sample_sequence_from_dataset
    │   ├── _sample_track_frame_from_static_image
    │   └── _sample_track_frame_from_sequence
    │
    └── _sample_track_frame
        ├── _sample_dataset
        ├── _sample_sequence_from_dataset
        ├── _sample_track_frame_from_static_image (x2)
        └── _sample_track_pair_from_sequence
            └── _sample_pair_idx_pair_within_max_diff
    Hyper-parameters
    ----------------
    negative_pair_ratio: float
        the ratio of negative pairs
    target_type: str
        "mask" or "bbox"
    """
    default_hyper_params = dict(negative_pair_ratio=0.0, target_type="bbox")

    def __init__(self,
                 datasets: List[DatasetBase] = [],
                 seed: int = 0,
                 data_filter=None) -> None:
        super().__init__(datasets, seed=seed)
        if data_filter is None:
            self.data_filter = [lambda x: False]
        else:
            self.data_filter = data_filter

        self._state["ratios"] = [
            d._hyper_params["ratio"] for d in self.datasets
        ]
        sum_ratios = sum(self._state["ratios"])
        self._state["ratios"] = [d / sum_ratios for d in self._state["ratios"]]
        self._state["max_diffs"] = [
            # max_diffs, or -1 (invalid value for video, but not used for static image dataset)
            d._hyper_params.get("max_diff", -1) for d in self.datasets
        ]

    def __getitem__(self, item) -> dict:
        is_negative_pair = (self._state["rng"].rand() <
                            self._hyper_params["negative_pair_ratio"])
        data1_pos = data2_pos = None
        tempos1 = []
        tempos2 = []
        temneg1 = []
        temneg2 = []
        sample_try_num = 0
        while self.data_filter(data1_pos) or self.data_filter(data2_pos):
            if is_negative_pair:
                data1_pos, data1_neg = self._sample_track_frame()
                data2_pos, data2_neg = self._sample_track_frame()
            else:
                data1_pos, data1_neg, data2_pos, data2_neg = self._sample_track_pair()  # get image 1 pair

            for i in range(1,6):
                tempos1.append(load_image(data1_pos["image_1"].split('.')[0].replace('img_120_split', 'img_120_5_split') + '_{}.jpg'.format(i)))
                temneg1.append(load_image(data1_neg["image_2"].split('.')[0].replace('img_120_split', 'img_120_5_split') + '_{}.jpg'.format(i)))
                tempos2.append(load_image(data2_pos["image_1"].split('.')[0].replace('img_120_split', 'img_120_5_split') + '_{}.jpg'.format(i)))
                temneg2.append(load_image(data2_neg["image_2"].split('.')[0].replace('img_120_split', 'img_120_5_split') + '_{}.jpg'.format(i)))
            data1_pos["image"] = tempos1
            data1_neg["image"] = temneg1
            data2_pos["image"] = tempos2
            data2_neg["image"] = temneg2
            sample_try_num += 1
        sampled_data = dict(
            data1_pos=data1_pos,
            data1_neg=data1_neg,
            data2_pos=data2_pos,
            data2_neg=data2_neg,
            is_negative_pair=is_negative_pair,
        )

        return sampled_data

    def _get_len_seq(self, seq_data) -> int:
        return len(seq_data["image_1"])
        # return len(seq_data["image"][0])


    def _sample_track_pair(self) -> Tuple[Dict, Dict]:
        dataset_idx, dataset = self._sample_dataset()
        sequence_data_pos, sequence_data_neg  = self._sample_sequence_from_dataset(dataset)
        len_seq = self._get_len_seq(sequence_data_pos)
        if len_seq == 1 and not isinstance(sequence_data_pos["anno"][0], list):
            # static image dataset
            data1 = self._sample_track_frame_from_static_image(sequence_data_pos)
            data2 = deepcopy(data1)
        else:
            # video dataset
            data1_pos, data1_neg, data2_pos, data2_neg = self._sample_track_pair_from_sequence(
                sequence_data_pos, sequence_data_neg, self._state["max_diffs"][dataset_idx])

        return data1_pos, data1_neg, data2_pos, data2_neg

    def _sample_track_frame(self) -> Dict:
        _, dataset = self._sample_dataset()
        sequence_data_pos, sequence_data_neg = self._sample_sequence_from_dataset(dataset)
        len_seq = self._get_len_seq(sequence_data_pos)
        if len_seq == 1:
            # static image dataset
            data_frame = self._sample_track_frame_from_static_image(
                sequence_data_pos)
        else:
            # video dataset
            data_frame_pos, data_frame_neg = self._sample_track_frame_from_sequence(sequence_data_pos, sequence_data_neg)

        return data_frame_pos, data_frame_neg

    def _sample_dataset(self):
        r"""
        Returns
        -------
        int
            sampled dataset index
        DatasetBase
            sampled dataset
        """
        dataset_ratios = self._state["ratios"]
        rng = self._state["rng"]
        dataset_idx = rng.choice(len(self.datasets), p=dataset_ratios)
        dataset = self.datasets[dataset_idx]

        return dataset_idx, dataset

    def _sample_sequence_from_dataset(self, dataset: DatasetBase) -> Dict:
        r"""
        """
        rng = self._state["rng"]
        len_dataset = len(dataset)
        idx = rng.choice(len_dataset)

        sequence_data_pos, sequence_data_neg  = dataset[idx]

        return sequence_data_pos, sequence_data_neg

    def _generate_mask_for_vos(self, anno):
        mask = Image.open(anno[0])
        mask = np.array(mask, dtype=np.uint8)
        obj_id = anno[1]
        mask[mask != obj_id] = 0
        mask[mask == obj_id] = 1
        return mask

    def _sample_track_frame_from_sequence(self, sequence_data_pos, sequence_data_neg) -> Dict:
        rng = self._state["rng"]
        len_seq = self._get_len_seq(sequence_data_pos)
        idx = rng.choice(len_seq)
        data_frame_pos = {k: v[idx] for k, v in sequence_data_pos.items()}
        data_frame_neg = {k: v[idx] for k, v in sequence_data_neg.items()}
        # convert mask path to mask, specical for youtubevos and davis
        if self._hyper_params["target_type"] == "mask":
            if isinstance(data_frame_pos["anno"], list):
                mask = self._generate_mask_for_vos(data_frame_pos["anno"])
                data_frame_pos["anno"] = mask
        return data_frame_pos, data_frame_neg

    def _sample_track_pair_from_sequence(self, sequence_data_pos: Dict, sequence_data_neg: Dict,
                                         max_diff: int) -> Tuple[Dict, Dict]:
        """sample a pair of frames within max_diff distance
        
        Parameters
        ----------
        sequence_data : List
            sequence data: image= , anno=
        max_diff : int
            maximum difference of indexes between two frames in the  pair
        
        Returns
        -------
        Tuple[Dict, Dict]
            track pair data
            data: image= , anno=
        """
        len_seq = self._get_len_seq(sequence_data_pos)
        idx1, idx2 = self._sample_pair_idx_pair_within_max_diff(
            len_seq, max_diff)
        data1_pos = {k: v[idx1] for k, v in sequence_data_pos.items()}
        data1_neg = {k: v[idx1] for k, v in sequence_data_neg.items()}
        # data1_1 = {k[1]: v[idx1] for k, v in sequence_data.items()}
        data2_pos = {k: v[idx2] for k, v in sequence_data_pos.items()}
        data2_neg = {k: v[idx2] for k, v in sequence_data_neg.items()}
        # data2_1 = {k[1]: v[idx1] for k, v in sequence_data.items()}
        if isinstance(data1_pos["anno"],
                      list) and self._hyper_params["target_type"] == "mask":
            # convert mask path to mask, specical for youtubevos
            data1_pos["anno"] = self._generate_mask_for_vos(data1_pos["anno"])
            data2_pos["anno"] = self._generate_mask_for_vos(data2_pos["anno"])
        return data1_pos, data1_neg, data2_pos, data2_neg

    def _sample_pair_idx_pair_within_max_diff(self, L, max_diff):
        r"""
        Draw a pair of index in range(L) within a given maximum difference
        Arguments
        ---------
        L: int
            difference
        max_diff: int
            difference
        """
        rng = self._state["rng"]
        idx1 = rng.choice(L)
        idx2_choices = list(range(idx1-max_diff, L)) + \
                    list(range(L+1, idx1+max_diff+1))
        idx2_choices = list(set(idx2_choices).intersection(set(range(L))))
        idx2 = rng.choice(idx2_choices)
        return int(idx1), int(idx2)

    def _sample_track_frame_from_static_image(self, sequence_data):
        rng = self._state["rng"]
        num_anno = len(sequence_data['anno'])
        if num_anno > 0:
            idx = rng.choice(num_anno)
            anno = sequence_data["anno"][idx]
        else:
            # no anno, assign a dummy one
            if self._hyper_params["target_type"] == "bbox":
                anno = [-1, -1, -1, -1]
            elif self._hyper_params["target_type"] == "mask":
                anno = np.zeros((sequence_data["image"][0].shape[:2]))
            else:
                logger.error("target type {} is not supported".format(
                    self._hyper_params["target_type"]))
                exit(0)
        data = dict(
            image=sequence_data["image"][0],
            anno=anno,
        )

        return data
