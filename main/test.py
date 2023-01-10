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

from case_detector import CaseDetector
from gen_batch import GenCrop
from batch_estimation import BatchEstimation
from crop_estimation import CropEstimation
from latency import Latency
from nnms import NMS
from visualization import Visualization
from save import Save
from deburger import Deburger
from data_preprocessing import DataPreprocessing

class Test:
    def __init__(self, config=None):
        self.config = config

        self.case_detector = CaseDetector(self.config)
        self.gen_batch = GenCrop(self.config)
        self.latency = Latency(self.config)
        self.batch_estimation = BatchEstimation(self.config)
        self.crop_estimation = CropEstimation(self.config)
        self.visualization = Visualization(self.config)
        self.nms = NMS(self.config)
        self.save = Save(self.config)
        self.debuger = Deburger(self.config)
        self.data_preprocessing = DataPreprocessing(self.config)
        
    def test_net(self, tester, dets, det_range, gpu_id):
        dump_results = []
        img_start = det_range[0]
        # img_id = 0
        # img_id3 = 0
        pbar = tqdm(total=det_range[1] - img_start - 1, position=gpu_id)
        pbar.set_description("GPU %s" % str(gpu_id))
        while img_start < det_range[1]:
            img_end = img_start + 1
            im_info = dets[img_start]
            # dets 내 같은 image_id를 갖는 범위 찾기 
            while img_end < det_range[1] and dets[img_end]['image_id'] == im_info['image_id']:
                img_end += 1
            
            # all human detection results of a certain image 
            cropped_data = dets[img_start:img_end] # 같은 image 내 다른 사람들을 모아 놓은 list
            # cropped_data = dets
            # img_id = 0
            
            # case detector
            if self.config.c1:
                self.case_detector.case_one_detect(cropped_data)
            if self.config.c2:
                self.case_detector.case_two_detect(cropped_data)
            if self.config.c3:
                self.case_detector.case_three_detect(cropped_data)

            pbar.update(img_end - img_start)
            img_start = img_end

            # kps: keypoints (self.config.num_kps == 17)
            kps_result = np.zeros((len(cropped_data), self.config.num_kps, 3)) # (#cropped_data)*17*3 
            area_save = np.zeros(len(cropped_data)) # 같은 이미지에서 crop된 이미지들의 크기를 저장하는 list

            # cluster human detection results with test_batch_size
            kps_result, area_save = self.batch_estimation.batch_estimation(cropped_data, tester, kps_result, area_save)
            
            if self.config.d2:
                self.debuger.d_two(kps_result)
            
            # score_result.shape == kps_result[:, :, 2].shape == (#cropped_data)*17
            score_result = np.copy(kps_result[:, :, 2]) 
            kps_result[:, :, 2] = 1
            kps_result = kps_result.reshape(-1, self.config.num_kps*3) # cropped image 개수 * 17 * 3 의 총 개수를 가지고 (#cropped_image, 51)로 재구성 => x,y,1 

            # print("kps_result", kps_result)
            # print("kps_result.shape", kps_result.shape)
            # exit()
            # print("kps_result.shape", kps_result.shape)
            # print("score_result.shape", score_result.shape)
            
            #rescoring and oks nms
            kps_result, score_result, area_save = self.nms.nms(score_result, kps_result, area_save, cropped_data)

            print("kps_result.shape", kps_result.shape)
            print("score_result.shape", score_result.shape)
            exit()

            #total image visualize part
            if self.config.vis_total and np.any(kps_result[:,:,2] > 0.1): # 0.9 때문에 저화질의 경우 vis 파일이 안 생기는 경우가 있음 -> 해당 keypoint에 대한 confidence 값이 0.9 이상인 위치가 한 군데라도 있으면 visualization 실행
                self.visualization.vis_total(kps_result, cropped_data)
            
            # save result
            dump_results = self.save.save(kps_result, dump_results, im_info, score_result)

        return dump_results


    def test(self, test_model):
        
        # annotation load
        d = Dataset()
        annot = d.load_annot(self.config.testset) # if testset==val: annot = person_keypoints_val2017.json 
        gt_img_id = d.load_imgid(annot) # id 순으로 정렬된 image들 (image_info_test-dev2017.json)
        # print("#gt_img_id: ", len(gt_img_id))
        # sys.stdout = open('output.txt','w')

        # human bbox load
        if self.config.useGTbbox and self.config.testset in ['train', 'val']:
            if self.config.testset == 'train':
                dets = d.load_train_data(score=True)
            elif self.config.testset == 'val':
                dets = d.load_val_data_with_annot()
            dets.sort(key=lambda x: (x['image_id']))
        else:
            with open(self.config.human_det_path, 'r') as f: # human_detection.json
                dets = json.load(f)
            # dets = [i for i in dets if i['image_id'] in gt_img_id]    
            dets = [i for i in dets if i['image_id'] in dict(islice(gt_img_id.items(),1))] # human_detection.json 내 id가 image_info_test-dev2017.json에도 있는 경우
            dets = [i for i in dets if i['category_id'] == 1] # 그 중에서 category id가 1인 경우만 선정
            dets = [i for i in dets if i['score'] > 0] # 그 중에서 score가 양수인 경우만 선정
            print("#dets: ", len(dets))
            # sys.stdout.close()
            # exit()
            dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True) # 역순 정렬
        
            img_id = []
            for i in dets:
                img_id.append(i['image_id']) # 위 순서로 image id만 추출
            imgname = d.imgid_to_imgname(annot, img_id, self.config.testset) # 해당 id에 맞는 image 이름 추출
            for i in range(len(dets)):
                dets[i]['imgpath'] = imgname[i] # dets 원소 위치에 맞게 이름 넣어줌

        # job assign (multi-gpu)
        from tfflat.mp_utils import MultiProc
        img_start = 0
        ranges = [0]
        img_num = len(np.unique([i['image_id'] for i in dets]))
        images_per_gpu = int(img_num / len(args.gpu_ids.split(','))) + 1
        # print(len(dets))
        # exit()
        # 같은 image_id를 갖는 애들이 많으면 img_end가 커짐, 누적되는 img_end를 ranges에 저장
        for run_img in range(img_num):
            img_end = img_start + 1
            while img_end < len(dets) and dets[img_end]['image_id'] == dets[img_start]['image_id']: # image_id가 연속으로 같은 경우에 img_end가 커짐 (image_id로 한번 정렬해서 이것이 가능)
                img_end += 1
            if (run_img + 1) % images_per_gpu == 0 or (run_img + 1) == img_num: # gpu가 여러 개이면 앞 조건에서 걸림, 한 개이면 뒤 조건에서 걸림
                ranges.append(img_end) # gpu가 여러 개인 경우, 첫번째 조건을 만족할 때마다 그 당시 img_end가 기록된다.
            img_start = img_end

        def func(gpu_id):
            self.config.set_args(args.gpu_ids.split(',')[gpu_id])
            tester = Tester(Model(), self.config)
            tester.load_weights(test_model)
            range = [ranges[gpu_id], ranges[gpu_id + 1]]
            return self.test_net(tester, dets, range, gpu_id)

        MultiGPUFunc = MultiProc(len(args.gpu_ids.split(',')), func)
        result = MultiGPUFunc.work()

        # evaluation
        d.evaluation(result, annot, self.config.result_dir, self.config.testset) # result == 실험값, annot == ground truth

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, dest='gpu_ids')
        parser.add_argument('--test_epoch', type=str, dest='test_epoch')
        args = parser.parse_args()

        # test gpus
        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        
        assert args.test_epoch, 'Test epoch is required.'
        return args

    global args
    args = parse_args()
    inference = Test(cfg)
    inference.test(int(args.test_epoch))

