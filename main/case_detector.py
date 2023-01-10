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

class CaseDetector:
    def __init__(self, config=None):
        self.config = config
        
        self.xtemp = []
        self.ytemp = []
        self.target = []

    def case_one_detect(self, cropped_data):
        ik = 0
        for data in cropped_data:
            _, _, wtemp, htemp = data['bbox']
            num_keypoints = data['num_keypoints']
            # print(wtemp, ',', htemp)
            # print(num_keypoints)
            self.xtemp.append(num_keypoints)
            self.ytemp.append(wtemp*htemp)
            if wtemp*htemp <= 500:
                print("image_id, human_id: %s, %s" %(data['image_id'], str(ik)))
            ik += 1
        # print('')
        # if (cnt/len(cropped_data)) > 0.7:
        #     target.append(int(cropped_data[0]['image_id']))
        #     print(cropped_data)
        #     print(cnt/len(cropped_data))
        #     print(len(cropped_data))
        #     print("target:", target)
        # print("xtemp: ", xtemp)
        # print("ytemp: ", ytemp)
        # exit()

    # case 2 detection
    def case_two_detect(self, cropped_data):
        for data in cropped_data:
            cnt = 0
            keypoints = np.array(data['joints'])
            xg = keypoints[0::3]; yg = (-1) * keypoints[1::3]; vg = keypoints[2::3]
            
            k1 = np.count_nonzero(vg > 0)
            if k1 == 0:
                continue

            l_shoulder = np.array([xg[5], yg[5], vg[5]]); r_shoulder = np.array([xg[6], yg[6], vg[6]]);
            l_elbow = np.array([xg[7], yg[7], vg[7]]); r_elbow = np.array([xg[8], yg[8], vg[8]])
            l_wrist = np.array([xg[9], yg[9], vg[9]]); r_wrist = np.array([xg[10], yg[10], vg[10]])
            l_hip = np.array([xg[11], yg[11], vg[11]]); r_hip = np.array([xg[12], yg[12], vg[12]])
            l_knee = np.array([xg[13], yg[13], vg[13]]); r_knee = np.array([xg[14], yg[14], vg[14]])
            l_ankle = np.array([xg[15], yg[15], vg[15]]); r_ankle = np.array([xg[16], yg[16], vg[16]])

            # 두 elbow-shoulder 사이 각도가 120도 이상 벌어질 때
            if l_shoulder[2]>0 and r_shoulder[2]>0 and l_elbow[2]>0 and r_elbow[2]>0:
                l_upper_arm = np.array([l_elbow[0]-l_shoulder[0], l_elbow[1]-l_shoulder[1]])
                r_upper_arm = np.array([r_elbow[0]-r_shoulder[0], r_elbow[1]-r_shoulder[1]])
                if np.dot(l_upper_arm, r_upper_arm)/(np.linalg.norm(l_upper_arm)*np.linalg.norm(r_upper_arm)) <= -0.5:
                    print("두 elbow-shoulder 사이 각도가 120도 이상 벌어질 때")
                    cnt += 1

            # # 두 wrist-elbow 사이 각도가 120도 이상 벌어질 때
            # if l_elbow[2]>0 and r_elbow[2]>0 and l_wrist[2]>0 and r_wrist[2]>0:
            #     l_lower_arm = np.array([l_wrist[0]-l_elbow[0], l_wrist[1]-l_elbow[1]])
            #     r_lower_arm = np.array([r_wrist[0]-r_elbow[0], r_wrist[1]-r_elbow[1]])
            #     if np.dot(l_lower_arm, r_lower_arm)/(np.linalg.norm(l_lower_arm)*np.linalg.norm(r_lower_arm)) <= -0.5:
            #         print("두 wrist-elbow 사이 각도가 120도 이상 벌어질 때")
            #         cnt += 1

            # 두 knee-hip 사이 각도가 90도 이상 벌어질 때
            if l_hip[2]>0 and r_hip[2]>0 and l_knee[2]>0 and r_knee[2]>0:
                l_upper_leg = np.array([l_knee[0]-l_hip[0], l_knee[1]-l_hip[1]])
                r_upper_leg = np.array([r_knee[0]-r_hip[0], r_knee[1]-r_hip[1]])
                if np.dot(l_upper_leg, r_upper_leg)/(np.linalg.norm(l_upper_leg)*np.linalg.norm(r_upper_leg)) <= 0:
                    cnt += 1

            # 두 ankle-knee 사이 각도가 90도 이상 벌어질 때
            if l_knee[2]>0 and r_knee[2]>0 and l_ankle[2]>0 and r_ankle[2]>0:
                l_lower_leg = np.array([l_ankle[0]-l_knee[0], l_ankle[1]-l_knee[1]])
                r_lower_leg = np.array([r_ankle[0]-r_knee[0], r_ankle[1]-r_knee[1]])
                if np.dot(l_lower_leg, r_lower_leg)/(np.linalg.norm(l_lower_leg)*np.linalg.norm(r_lower_leg)) <= 0:
                    print("두 ankle-knee 사이 각도가 90도 이상 벌어질 때")
                    cnt += 1

            # # 한 elbow가 평균 shoulder보다 높은 경우
            # if l_shoulder[2]>0 and r_shoulder[2]>0:
            #     mid_shoulder = np.array([(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2])
            #     if l_elbow[2]>0:
            #         if l_elbow[1] > mid_shoulder[1]:
            #             print("한 elbow가 평균 shoulder보다 높은 경우")
            #             cnt += 1
            #     if r_elbow[2]>0:
            #         if r_elbow[1] > mid_shoulder[1]:
            #             print("한 elbow가 평균 shoulder보다 높은 경우")
            #             cnt += 1

            # # 한 wrist가 평균 elbow보다 높은 경우
            # if l_elbow[2]>0 and r_elbow[2]>0:
            #     mid_elbow = np.array([(l_elbow[0]+r_elbow[0])/2, (l_elbow[1]+r_elbow[1])/2])
            #     if l_wrist[2]>0:
            #         if l_wrist[1] > mid_elbow[1]:
            #             print("한 wrist가 평균 elbow보다 높은 경우")
            #             cnt += 1
            #     if r_wrist[2]>0:
            #         if r_wrist[1] > mid_elbow[1]:
            #             print("한 wrist가 평균 elbow보다 높은 경우")
            #             cnt += 1

            # # 한 knee가 평균 hip보다 높은 경우
            # if l_hip[2]>0 and r_hip[2]>0:
            #     mid_hip = np.array([(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2])
            #     if l_knee[2]>0:
            #         if l_knee[1] > mid_hip[1]:
            #             print("한 knee가 평균 hip보다 높은 경우")
            #             cnt += 1
            #     if r_knee[2]>0:
            #         if r_knee[1] > mid_hip[1]:
            #             print("한 knee가 평균 hip보다 높은 경우")
            #             cnt += 1

            # # 한 ankle이 평균 knee보다 높은 경우 
            # if l_knee[2]>0 and r_knee[2]>0:
            #     mid_knee = np.array([(l_knee[0]+r_knee[0])/2, (l_knee[1]+r_knee[1])/2])
            #     if l_ankle[2]>0:
            #         if l_ankle[1] > mid_knee[1]:
            #             print("한 ankle이 평균 knee보다 높은 경우")
            #             cnt += 1
            #     if r_ankle[2]>0:
            #         if r_ankle[1] > mid_knee[1]:
            #             print("한 ankle이 평균 knee보다 높은 경우")
            #             cnt += 1

            if cnt > 0:
                self.target.append(int(cropped_data[0]['image_id']))
        print("target:", self.target)

    # none general pose
    def case_three_detect(self, cropped_data):
        ik = 0
        for data in cropped_data:
            keypoints = np.array(data['joints'])
            xg = keypoints[0::3]; yg = (-1) * keypoints[1::3]; vg = keypoints[2::3]
            
            k1 = np.count_nonzero(vg > 0)
            if k1 == 0:
                continue

            l_shoulder = np.array([xg[5], yg[5], vg[5]]); r_shoulder = np.array([xg[6], yg[6], vg[6]]);
            l_elbow = np.array([xg[7], yg[7], vg[7]]); r_elbow = np.array([xg[8], yg[8], vg[8]])
            l_wrist = np.array([xg[9], yg[9], vg[9]]); r_wrist = np.array([xg[10], yg[10], vg[10]])
            l_hip = np.array([xg[11], yg[11], vg[11]]); r_hip = np.array([xg[12], yg[12], vg[12]])
            l_knee = np.array([xg[13], yg[13], vg[13]]); r_knee = np.array([xg[14], yg[14], vg[14]])
            l_ankle = np.array([xg[15], yg[15], vg[15]]); r_ankle = np.array([xg[16], yg[16], vg[16]])
            
            if l_shoulder[2]>0 and r_shoulder[2]>0 and l_hip[2]>0 and r_hip[2]>0 and l_ankle[2]>0 and r_ankle[2]>0:
                mid_shoulder = np.array([(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2])
                mid_hip = np.array([(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2])
                mid_ankle = np.array([(l_ankle[0]+r_ankle[0])/2, (l_ankle[1]+r_ankle[1])/2])

                upper_body = np.array([mid_shoulder[0]-mid_hip[0], mid_shoulder[1]-mid_hip[1]])
                lower_body = np.array([mid_ankle[0]-mid_hip[0], mid_ankle[1]-mid_hip[1]])

                if not ((np.dot(upper_body, lower_body)/(np.linalg.norm(upper_body)*np.linalg.norm(lower_body)) <= -0.5) and (np.linalg.norm(upper_body)/np.linalg.norm(lower_body) < 11)):
                    print("image_id, human_id: %s, %s" %(data['image_id'], str(ik)))
            ik += 1
