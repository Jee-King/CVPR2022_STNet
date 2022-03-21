# -*- coding: utf-8 -*
import time
from typing import List

# from PIL import Image
import cv2
import numpy as np
import torch

from videoanalyst.evaluation.got_benchmark.utils.viz import show_frame
from videoanalyst.pipeline.pipeline_base import PipelineBase


class PipelineTracker(object):
    def __init__(self,
                 name: str,
                 pipeline: PipelineBase,
                 is_deterministic: bool = True):
        """Helper tracker for comptability with 
        
        Parameters
        ----------
        name : str
            [description]
        pipeline : PipelineBase
            [description]
        is_deterministic : bool, optional
            [description], by default False
        """
        self.name = name
        self.is_deterministic = is_deterministic
        self.pipeline = pipeline

    def init(self, image_pos: np.array, image_neg, box, snn_state_first):
        """Initialize pipeline tracker
        
        Parameters
        ----------
        image : np.array
            image of the first frame
        box : np.array or List
            tracking bbox on the first frame
            formate: (x, y, w, h)
        """
        self.pipeline.init(image_pos, image_neg, box, snn_state_first)

    def update(self, image_pos: np.array, image_neg, snn_state):
        """Perform tracking
        
        Parameters
        ----------
        image : np.array
            image of the current frame
        
        Returns
        -------
        np.array
            tracking bbox
            formate: (x, y, w, h)
        """
        return self.pipeline.update(image_pos, image_neg, snn_state)

    def track(self, img_files_pos: List, img_files_neg, box, visualize: bool = False):
        """Perform tracking on a given video sequence
        
        Parameters
        ----------
        img_files : List
            list of image file paths of the sequence
        box : np.array or List
            box of the first frame
        visualize : bool, optional
            Visualize or not on each frame, by default False
        
        Returns
        -------
        [type]
            [description]
        """
        frame_num = len(img_files_pos)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        cfg_cnn = [(3, 64, 2, 0, 11),
                   (64, 128, 2, 0, 9),
                   (128, 256, 2, 0, 5),
                   (64, 128, 1, 1, 3),
                   (128, 256, 1, 1, 3)]
        cfg_kernel = [147, 70, 33, 31, 31]
        cfg_kernel_first = [59, 26, 11, 15, 15]
        batch_size = 1
        c1_mem_first = c1_spike_first = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel_first[0],
                                                    cfg_kernel_first[0])
        c2_mem_first = c2_spike_first = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel_first[1],
                                                    cfg_kernel_first[1])
        c3_mem_first = c3_spike_first = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel_first[2],
                                                    cfg_kernel_first[2])
        snn_state_first = [c1_mem_first.cuda(), c1_spike_first.cuda(), c2_mem_first.cuda(), c2_spike_first.cuda(),
                           c3_mem_first.cuda(), c3_spike_first.cuda()]
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0])
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1])
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2])
        snn_state = [c1_mem.cuda(), c1_spike.cuda(), c2_mem.cuda(), c2_spike.cuda(), c3_mem.cuda(), c3_spike.cuda()]

        for f, img_file in enumerate(img_files_pos):
            # image = Image.open(img_file)
            # if not image.mode == 'RGB':
            #     image = image.convert('RGB')
            image_pos = []
            image_neg = []
            for i in range(1, 6):
                image_pos.append(
                    cv2.imread(img_file.split('.')[0].replace('img_120_split', 'img_120_5_split') + '_{}.jpg'.format(i),
                               cv2.IMREAD_COLOR))
 
                image_neg.append(cv2.imread(
                    img_files_neg[f].split('.')[0].replace('img_120_split', 'img_120_5_split') + '_{}.jpg'.format(i),
                    cv2.IMREAD_COLOR))
            start_time = time.time()
            if f == 0:
                self.init(image_pos, image_neg, box, snn_state_first)
            else:
                boxes[f, :], snn_state = self.update(image_pos, image_neg, snn_state)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image_pos, boxes[f, :])
        # print(np.mean(times))
        return boxes, times
