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

class Latency:
    def __init__(self, config=None):
        self.config = config

        self.latency = []

    def measure_each(self, imgs, tester):
        for i in imgs:
            i = np.array(i).reshape(1,256,192,3)
            start_latency = time.time()
            heatmap = tester.predict_one([i])[0] # heatmap: (동일한 image_id를 가지는 이미지 개수)*64*48*17
            end_latency = time.time()
            self.latency.append(end_latency-start_latency)
            print("mean_latency:", sum(self.latency)/len(self.latency))
            print("len_latency:", len(self.latency))
        
    def measure_total(self, imgs, tester):
        start_latency = time.time()
        heatmap = tester.predict_one([imgs])[0] # heatmap: (동일한 image_id를 가지는 이미지 개수)*64*48*17
        end_latency = time.time()
        self.latency.append((end_latency-start_latency))
        # 1. batch size - 한 crop image를 처리하는데 걸리는 시간 (batch size가 커짐에 따라 시간이 줄어듦)
        print("total_latency:", sum(self.latency)/11004)
        # 2. batch size - 한 prediction에 걸리는 시간 (batch size가 커짐에 따라 시간이 증가함)
        print("crop_latency:", sum(self.latency)/len(self.latency))
        print("len_latency:", len(self.latency))
