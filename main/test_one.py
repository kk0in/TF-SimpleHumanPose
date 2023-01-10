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

from gen_batch import generate_batch
from dataset import Dataset
from nms.nms import oks_nms

from itertools import islice

soccer_img_path = osp.join('..', 'data', 'soccer', 'images', 'sb5.0')

def test_net(tester, dets, det_range, gpu_id):

    dump_results = []

    start_time = time.time()

    img_start = det_range[0]
    img_end = det_range[1] - 1
    img_id = 0
    img_id2 = 0
    pbar = tqdm(total=img_end - img_start - 1, position=gpu_id)
    pbar.set_description("GPU %s" % str(gpu_id))
        
    # all human detection results of a certain image 
    cropped_data = dets[img_start:img_end] # 같은 image를 가리키고 있음

    pbar.update(img_end - img_start)
    # img_start = img_end

    # kps: keypoints
    kps_result = np.zeros((len(cropped_data), cfg.num_kps, 3)) # cfg.num_kps == 17
    area_save = np.zeros(len(cropped_data))

    # cluster human detection results with test_batch_size
    start_id = img_start
    end_id = img_end

    imgs = []
    crop_infos = []
    for i in range(start_id, end_id):
        img, crop_info = generate_batch(cropped_data[i], stage='soccer') # detection part, img: 256*192*3, crop_info: 4 
        imgs.append(img)
        crop_infos.append(crop_info)
    imgs = np.array(imgs)
    crop_infos = np.array(crop_infos)
    
    # forward (pose estimation)
    heatmap = tester.predict_one([imgs])[0] # heatmap: 2*64*48*17

    if cfg.flip_test:
        flip_imgs = imgs[:, :, ::-1, :]
        flip_heatmap = tester.predict_one([flip_imgs])[0]
        
        flip_heatmap = flip_heatmap[:, :, ::-1, :]
        for (q, w) in cfg.kps_symmetry:
            flip_heatmap_w, flip_heatmap_q = flip_heatmap[:,:,:,w].copy(), flip_heatmap[:,:,:,q].copy()
            flip_heatmap[:,:,:,q], flip_heatmap[:,:,:,w] = flip_heatmap_w, flip_heatmap_q
        flip_heatmap[:,:,1:,:] = flip_heatmap.copy()[:,:,0:-1,:]
        heatmap += flip_heatmap
        heatmap /= 2
    
    # for each human detection from clustered batch
    for image_id in range(start_id, end_id):
        
        for j in range(cfg.num_kps):
            hm_j = heatmap[image_id - start_id, :, :, j] # 명암 값이 적힌 2차원 map, 사진의 모양과 동일
            idx = hm_j.argmax() # 명암 값이 가장 큰 위치의 좌표인데 픽셀의 좌표를 1차원으로 바꾼 후 위치를 반환함 -> 다시 2차원으로 바꿔주어야 함
    
            y, x = np.unravel_index(idx, hm_j.shape) # 위에서 찾은 위치의 2차원 좌표 
 
            #  floor(버림) -> 반올림해서 픽셀 위치 구하기 
            px = int(math.floor(x + 0.5))
            py = int(math.floor(y + 0.5))

            # output 범위 내에 들어가 있는지 확인 (output_shape = (64, 48))
            if 1 < px < cfg.output_shape[1]-1 and 1 < py < cfg.output_shape[0]-1:
                diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                                    hm_j[py+1][px]-hm_j[py-1][px]])
                diff = np.sign(diff)
                x += diff[0] * .25
                y += diff[1] * .25

            # 0,1
            kps_result[image_id, j, :2] = (x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
            # 2
            kps_result[image_id, j, 2] = hm_j.max() / 255 

        vis=True
        crop_info = crop_infos[image_id - start_id,:]
        area = (crop_info[2] - crop_info[0]) * (crop_info[3] - crop_info[1])
        if vis and np.any(kps_result[image_id,:,2]) > 0.9 and area > 96**2:
            tmpimg = imgs[image_id-start_id].copy()
            tmpimg = cfg.denormalize_input(tmpimg)
            tmpimg = tmpimg.astype('uint8')
            tmpkps = np.zeros((3,cfg.num_kps))
            tmpkps[:2,:] = kps_result[image_id,:,:2].transpose(1,0)
            tmpkps[2,:] = kps_result[image_id,:,2]
            _tmpimg = tmpimg.copy()
            _tmpimg = cfg.vis_keypoints(_tmpimg, tmpkps)
            cv2.imwrite(osp.join(cfg.soccer_dir, str(img_id+1) + '_output.jpg'), _tmpimg)
            img_id += 1

        # map back to original images
        for j in range(cfg.num_kps):
            kps_result[image_id, j, 0] = kps_result[image_id, j, 0] / cfg.input_shape[1] * (\
            crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + crop_infos[image_id - start_id][0]
            kps_result[image_id, j, 1] = kps_result[image_id, j, 1] / cfg.input_shape[0] * (\
            crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + crop_infos[image_id - start_id][1]
        
        area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])
            
    #vis
    vis = False
    if vis and np.any(kps_result[:,:,2] > 0.9):
        tmpimg = cv2.imread(os.path.join(cropped_data[i]))
        tmpimg = tmpimg.astype('uint8')
        for i in range(len(kps_result)):
            tmpkps = np.zeros((3,cfg.num_kps))
            tmpkps[:2,:] = kps_result[i, :, :2].transpose(1,0)
            tmpkps[2,:] = kps_result[i, :, 2]
            tmpimg = cfg.vis_keypoints(tmpimg, tmpkps) #tmpimg: image file, tmpkps: keypoints file
        cv2.imwrite(osp.join(cfg.soccer_dir, str(img_id2+1) + '.jpg'), tmpimg)
        img_id2 += 1
    
    score_result = np.copy(kps_result[:, :, 2]) 
    kps_result[:, :, 2] = 1
    kps_result = kps_result.reshape(-1,cfg.num_kps*3)
    
    # # rescoring and oks nms
    # if cfg.dataset == 'COCO':
    #     rescored_score = np.zeros((len(score_result))) # score_result 길이 == image 개수 
    #     for i in range(len(score_result)):
    #         score_mask = score_result[i] > cfg.score_thr
    #         if np.sum(score_mask) > 0:
    #             rescored_score[i] = np.mean(score_result[i][score_mask]) * cropped_data[i]['score']
    #     score_result = rescored_score
    #     keep = oks_nms(kps_result, score_result, area_save, cfg.oks_nms_thr) # keep: oks가 thr보다 작은 경우만 해당
    #     if len(keep) > 0 :
    #         kps_result = kps_result[keep,:]
    #         score_result = score_result[keep]
    #         area_save = area_save[keep]
    # elif cfg.dataset == 'PoseTrack':
    #     keep = oks_nms(kps_result, np.mean(score_result,axis=1), area_save, cfg.oks_nms_thr)
    #     if len(keep) > 0 :
    #         kps_result = kps_result[keep,:]
    #         score_result = score_result[keep,:]
    #         area_save = area_save[keep]
    
    # # save result
    # for i in range(len(kps_result)):
    #     if cfg.dataset == 'COCO':
    #         result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(score_result[i], 4)),
    #                         keypoints=kps_result[i].round(3).tolist())
    #     elif cfg.dataset == 'PoseTrack':
    #         result = dict(image_id=im_info['image_id'], category_id=1, track_id=0, scores=score_result[i].round(4).tolist(),
    #                         keypoints=kps_result[i].round(3).tolist())
    #     elif cfg.dataset == 'MPII':
    #         result = dict(image_id=im_info['image_id'], scores=score_result[i].round(4).tolist(),
    #                         keypoints=kps_result[i].round(3).tolist())

    #     dump_results.append(result)

    return dump_results

def test(test_model):
    
    dets = []

    for i in range(1,542):
        dets.append(soccer_img_path+'/'+str(i)+'.jpg')

    # job assign (multi-gpu)
    from tfflat.mp_utils import MultiProc
    img_start = 1
    ranges = [0, 542]
    img_num = len(dets) + 1 
    # images_per_gpu = int(img_num / len(args.gpu_ids.split(','))) + 1
    
    def func(gpu_id):
        cfg.set_args(args.gpu_ids.split(',')[gpu_id])
        tester = Tester(Model(), cfg)
        tester.load_weights(test_model)
        range = [ranges[gpu_id], ranges[gpu_id + 1]]
        return test_net(tester, dets, range, gpu_id)

    MultiGPUFunc = MultiProc(len(args.gpu_ids.split(',')), func)
    result = MultiGPUFunc.work()

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
    test(int(args.test_epoch))
