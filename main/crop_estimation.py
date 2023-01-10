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

from visualization import Visualization
from data_preprocessing import DataPreprocessing
from deburger import Deburger

class CropEstimation:
    def __init__(self, config=None):
        self.config = config

        self.visualization = Visualization(self.config)
        self.data_preprocessing = DataPreprocessing(self.config)
        self.deburger = Deburger(self.config)

    def crop_estimation(self, heatmap, imgs, start_id, end_id, kps_result, crop_infos, area_save, cropped_data):
        for image_id in range(start_id, end_id): # 같은 image_id를 갖는 범위
            if self.config.makepatch:
                self.data_preprocessing.make_patch(imgs)
                
            for j in range(self.config.num_kps):
                hm_j = heatmap[image_id - start_id, :, :, j] # shape: 64*48 -> keypoint마다 64*48*1의 feature map이 생성되며, 각 픽셀은 해당 픽셀에 해당 keypoint에 대한 confidence 값에 따라서 최대 255까지 적힘
                idx = hm_j.argmax() # confidence 값이 가장 큰 위치의 좌표인데 픽셀의 좌표를 1차원으로 바꾼 후 위치를 반환함 -> 다시 2차원으로 바꿔주어야 함
        
                y, x = np.unravel_index(idx, hm_j.shape) # (64*48 범위에서) 위에서 찾은 위치의 2차원 좌표 (해당 keypoint에 대한 confidence 값이 가장 높은 위치)
                
                #  floor(버림) -> 반올림해서 예상되는 keypoint의 픽셀 상 위치 구하기 
                px = int(math.floor(x + 0.5))
                py = int(math.floor(y + 0.5))

                # output 범위 내에 들어가 있는지 확인 (output_shape = (64, 48))
                if 1 < px < self.config.output_shape[1]-1 and 1 < py < self.config.output_shape[0]-1:
                    diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                                        hm_j[py+1][px]-hm_j[py-1][px]])
                    diff = np.sign(diff)
                    x += diff[0] * .25
                    y += diff[1] * .25
                
                if self.config.d3:
                    self.deburger.d_three(self, hm_j, image_id, y, x, py, px, diff)

                ## kps_result의 1,2 열은 조정된 x와 y 좌표가 들어가고, 3열에는 해당 위치 해당 keypoint에 대한 confidence 값이 들어감 
                # input_shape/ output_shape == 4
                kps_result[image_id, j, :2] = (x * self.config.input_shape[1] / self.config.output_shape[1], y * self.config.input_shape[0] / self.config.output_shape[0])
                # confidence score 
                kps_result[image_id, j, 2] = hm_j.max() / 255 

            
            crop_info = crop_infos[image_id - start_id,:]
            # crop_info == (xl,yl,xr,yr)
            area = (crop_info[2] - crop_info[0]) * (crop_info[3] - crop_info[1]) # (xr-xl)*(yr-yl) == bbox 면적

            if self.config.d4:
                self.deburger.d_four(kps_result, image_id)
                
            # img_id = 0
            # confidence 값이 0.9 이상인 위치가 한 군데라도 있으면 + cropped area가 기준보다 크면 visualization 실행 
            # 현 상태에서는 np.any(kps_result[image_id,:,2])이 True(1)여서 무조건 1>0.9를 만족하게 됨 (문경식 님에게 메일로 문의 결과 오타 맞음 -> np.any(kps_result[:,:,2] > 0.9) 로 수정함  )
            # area는 bbox (x,y,w,h) 정보 이용해서 (1.25w)*(1.25h)와 같이 계산
            if self.config.vis_each and np.any(kps_result[image_id,:,2] > 0.1) and area > 96**2: 
            # if vis:
                self.visualization.vis_each(imgs, kps_result, cropped_data, image_id, start_id,)

            # map back to original imagesim
            for j in range(self.config.num_kps):
                # x_new = x_l + 세로길이*(원래x/해상도 세로)
                kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / self.config.input_shape[1] * (\
                crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + crop_infos[image_id - start_id][0]
                # y_new = y_l + 가로길이*(원래y/해상도 가로)
                kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / self.config.input_shape[0] * (\
                crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + crop_infos[image_id - start_id][1]
            
            area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])
        
        return kps_result, area_save
