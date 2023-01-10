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

class Visualization:
    def __init__(self, config=None):
        self.config = config
        
        self.img_id = 0
        
    def vis_each(self, imgs, kps_result, cropped_data, image_id, start_id):
        tmpimg = imgs[image_id-start_id].copy()
        tmpimg = self.config.denormalize_input(tmpimg)
        # tmpimg = cv2.imread(os.path.join(self.config.img_path, str(self.config.pid) + '.jpg'))
        tmpimg = tmpimg.astype('uint8')
        tmpkps = np.zeros((3, self.config.num_kps))
        # tmpkps: 3*17 -> 1행: x, 2행: y, 3행: confidence 값
        tmpkps[:2,:] = kps_result[image_id,:,:2].transpose(1,0) 
        tmpkps[2,:] = kps_result[image_id,:,2] 
        _tmpimg = tmpimg.copy()
        _tmpimg = self.config.vis_keypoints(_tmpimg, tmpkps)
        cv2.imwrite(osp.join(self.config.vis_dir, cropped_data[0]['imgpath'][-16:-4] + '_' + str(self.img_id) + '.jpg'), _tmpimg)
        # self.config.pid += 1
        self.img_id += 1

    def vis_total(self, kps_result, cropped_data):
        tmpimg = cv2.imread(os.path.join(self.config.img_path, cropped_data[0]['imgpath']))
        tmpimg = tmpimg.astype('uint8')
        for i in range(len(kps_result)):
            # tmpkps: 3*17 -> 1행: x, 2행: y, 3행: confidence
            tmpkps = np.zeros((3, self.config.num_kps))
            tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
            tmpkps[2,:] = kps_result[i, :, 2]
            tmpimg = self.config.vis_keypoints(tmpimg, tmpkps) 
        cv2.imwrite(osp.join(self.config.vis_dir, cropped_data[0]['imgpath'][-16:-4] + '.jpg'), tmpimg)
