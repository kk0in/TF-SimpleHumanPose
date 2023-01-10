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

class Save:
    def __init__(self, config=None):
        self.config = config

    def save(self, kps_result, dump_results, im_info, score_result):
        for i in range(len(kps_result)): # cropped_image 개수만큼 for문 순환 
            if self.config.dataset == 'COCO' or self.config.dataset == 'Soccer':
                result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(score_result[i], 4)),
                            keypoints=kps_result[i].round(3).tolist(), human_id=i, new_id=self.config.pid2)
                self.config.pid2 += 1
            elif self.config.dataset == 'PoseTrack':
                result = dict(image_id=im_info['image_id'], category_id=1, track_id=0, scores=score_result[i].round(4).tolist(),
                            keypoints=kps_result[i].round(3).tolist())
            elif self.config.dataset == 'MPII':
                result = dict(image_id=im_info['image_id'], scores=score_result[i].round(4).tolist(),
                            keypoints=kps_result[i].round(3).tolist())

            dump_results.append(result)
        
        return dump_results

    