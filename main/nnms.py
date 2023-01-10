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

class NMS:
    def __init__(self, config=None):
        self.config = config

    def nms(self, score_result, kps_result, area_save, cropped_data):
        if self.config.dataset == 'COCO' or self.config.dataset == 'Soccer':
            rescored_score = np.zeros((len(score_result))) # score_result 길이 == cropped data 개수 (동일한 이미지 id 개수) -> 1차원 리스트
            for i in range(len(score_result)):
                score_mask = score_result[i] > self.config.score_thr # 17개 keypoint 항목에 대하여 각각 score_thr == 0.2 보다 크면 1, 작으면 0으로 기록됨 
                if np.sum(score_mask) > 0: # 0.2보다 큰 게 하나라도 있으면, 이 조건 충족함 
                    # 17개 데이터 중 0.2를 넘는 confidence 값들만 가지고 평균을 내서 rescored_score[i]에 저장 
                    rescored_score[i] = np.mean(score_result[i][score_mask]) * cropped_data[i]['score'] # (normally) cropped_data[i]['score'] == 1
            
            # print("score_result:", score_result)
            # print("rescored_score:", rescored_score)
            # exit()

            # score_result.shape: (#cropped_image, ) -> 각 요소는 모든 keypoint에 대한 confidence 값들을 평균낸 값
            score_result = rescored_score
            # print("score_result", score_result)
            # exit()

            # NMS: 대다수의 object detection algorithm은 object가 존재하는 위치 주변에 여러 개의 keypoint set을 만듦 -> 이 중 하나의 keypoint set을 선택해야 하는데, 이때 적용하는 기법이 NMS; 즉, detector가 예측한 keypoint set 중에서 정확한 set을 선택하도록 하는 기법
            keep = oks_nms(kps_result, score_result, area_save, self.config.oks_nms_thr) # NMS를 적용한 후 선택된 keypoint set에 대한 정보를 가지고 있는 변수
            # print("keep:", keep)

            # gt 파일을 보면 대부분 한 사람 당 하나의 keypoint set이 있기 때문에 NMS 과정이 크게 의미는 없음 (실제로 아래 print 결과의 before, after가 거의 비슷)
            if len(keep) > 0 :
                # print("before")
                # print("keep_result.shape:", kps_result.shape)
                # print("score_result.shape:", score_result.shape)
                # print("area_save.shape:", area_save.shape)

                # 위에서 구한 NMS 결과 적용
                kps_result = kps_result[keep,:]
                score_result = score_result[keep]
                area_save = area_save[keep]

                # print("after")
                # print("keep_result.shape:", kps_result.shape)
                # print("score_result.shape:", score_result.shape)
                # print("area_save.shape:", area_save.shape)

                # print("keep_result:", kps_result)

                # exit()
                
        elif self.config.dataset == 'PoseTrack':
            keep = oks_nms(kps_result, np.mean(score_result,axis=1), area_save, self.config.oks_nms_thr)
            if len(keep) > 0 :
                kps_result = kps_result[keep,:]
                score_result = score_result[keep,:]
                area_save = area_save[keep]

        return kps_result, score_result, area_save
        