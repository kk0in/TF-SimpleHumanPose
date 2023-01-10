from email.mime import image
from operator import gt
import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import math

import tensorflow as tf

from tfflat.base import Tester
from tfflat.utils import mem_info
from model import Model

# from gen_batch import generate_batch
from dataset import Dataset
from nms.nms import oks_nms

from itertools import islice

from latency import Latency 
from deburger import Deburger
from gen_batch import GenCrop
from crop_estimation import CropEstimation

class BatchEstimation:
    def __init__(self, config=None):
        self.config = config

        self.latency = Latency(self.config)
        self.deburger = Deburger(self.config)
        self.gen_batch = GenCrop(self.config) 
        self.crop_estimation = CropEstimation(self.config)

    def batch_estimation(self, cropped_data, tester, kps_result, area_save):
        for batch_id in range(0, len(cropped_data), self.config.test_batch_size):
            start_id = batch_id
            end_id = min(len(cropped_data), batch_id + self.config.test_batch_size) 
            # end_id = batch_id + self.config.test_batch_size
            imgs = []
            crop_infos = []
            for i in range(start_id, end_id):
                # cropped_data[i] structure: {'image_id', 'imgpath', 'bbox', 'joints', 'score'} (dict)
                img, crop_info = self.gen_batch.generate_batch(cropped_data[i], self.config.pid+i, stage='test') # detection part, img: 256*192*3, crop_info: 4 
                #print(img.shape)
                imgs.append(img)
                crop_infos.append(crop_info)
            # np array formatting
            imgs = np.array(imgs)
            # print(imgs.shape)
            crop_infos = np.array(crop_infos)
            # print(crop_infos.shape)


            # latency measure
            if self.config.ltc_each:
                self.latency.measure_each(imgs, tester)
            if self.config.ltc_total:
                self.latency.measure_total(imgs, tester)

            # pose estimation (inference)
            # keypoint에 대한 confidence map (64*48)
            # 256*192로 crop된 이미지에 대해서 heatmap 생성  -> 가로 세로 4 픽셀 씩 묶어서 해당 영역에 keypoint에 대한 confidence 값이 적힘
            heatmap = tester.predict_one([imgs])[0] # heatmap: (동일한 image_id를 가지는 이미지 개수)*64*48*17
            
            if self.config.flip_test:
                flip_imgs = imgs[:, :, ::-1, :]
                flip_heatmap = tester.predict_one([flip_imgs])[0]
                
                flip_heatmap = flip_heatmap[:, :, ::-1, :]
                for (q, w) in self.config.kps_symmetry:
                    flip_heatmap_w, flip_heatmap_q = flip_heatmap[:,:,:,w].copy(), flip_heatmap[:,:,:,q].copy()
                    flip_heatmap[:,:,:,q], flip_heatmap[:,:,:,w] = flip_heatmap_w, flip_heatmap_q
                flip_heatmap[:,:,1:,:] = flip_heatmap.copy()[:,:,0:-1,:]
                heatmap += flip_heatmap
                heatmap /= 2

            if self.config.d1:
                self.deburger.d_one(cropped_data, batch_id, imgs, crop_infos, heatmap)
            
            # for each human detection from clustered batch
            # img_id = 0
            kps_result, area_save = self.crop_estimation.crop_estimation(heatmap, imgs, start_id, end_id, kps_result, crop_infos, area_save, cropped_data)

        return kps_result, area_save
    