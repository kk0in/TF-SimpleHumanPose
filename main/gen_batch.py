import os
import os.path as osp
import numpy as np
import cv2
# from config import cfg
import random
import time
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class GenCrop:
    def __init__(self, config=None):
        self.config = config

    def get_affine_transform(self, center,
                            scale,
                            rot,
                            output_size,
                            shift=np.array([0, 0], dtype=np.float32),
                            inv=0):

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def generate_batch(self, d, pid, stage='train'):
        # print("img path:", d['imgpath'][:-4])
        # exit()

        if self.config.patch:
            img = cv2.imread(os.path.join(self.config.img_path, str(pid) + '.jpg'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            img = cv2.imread(os.path.join(self.config.img_path, d['imgpath'][:-4] + '.jpg'), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if img is None:
            print('cannot read ' + os.path.join(self.config.img_path, d['imgpath']))
            assert 0

        bbox = np.array(d['bbox']).astype(np.float32)
        
        x, y, w, h = bbox # x,y는 이미지 왼쪽 하단의 위치
        aspect_ratio = self.config.input_shape[1]/self.config.input_shape[0] # input_shape = (256, 192)
        center = np.array([x + w * 0.5, y + h * 0.5])
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w,h]) * 1.25 
        rotation = 0

        if stage == 'train':

            joints = np.array(d['joints']).reshape(self.config.num_kps, 3).astype(np.float32)
            
            # data augmentation(확대)
            scale = scale * np.clip(np.random.randn()*self.config.scale_factor + 1, 1-self.config.scale_factor, 1+self.config.scale_factor)
            rotation = np.clip(np.random.randn()*self.config.rotation_factor, -self.config.rotation_factor*2, self.config.rotation_factor*2)\
                    if random.random() <= 0.6 else 0
            if random.random() <= 0.5:
                img = img[:, ::-1, :]
                center[0] = img.shape[1] - 1 - center[0]
                joints[:,0] = img.shape[1] - 1 - joints[:,0]
                #대칭성 이용
                for (q, w) in self.config.kps_symmetry:
                    joints_q, joints_w = joints[q,:].copy(), joints[w,:].copy()
                    joints[w,:], joints[q,:] = joints_q, joints_w

            trans = self.get_affine_transform(center, scale, rotation, (self.config.input_shape[1], self.config.input_shape[0]))
            cropped_img = cv2.warpAffine(img, trans, (self.config.input_shape[1], self.config.input_shape[0]), flags=cv2.INTER_LINEAR)
            # cropped_img = cropped_img[:,:, ::-1]
            cropped_img = self.config.normalize_input(cropped_img)
            
            for i in range(self.config.num_kps):
                if joints[i,2] > 0:
                    joints[i,:2] = affine_transform(joints[i,:2], trans)
                    joints[i,2] *= ((joints[i,0] >= 0) & (joints[i,0] < self.config.input_shape[1]) & (joints[i,1] >= 0) & (joints[i,1] < self.config.input_shape[0]))
            target_coord = joints[:,:2]
            target_valid = joints[:,2]
            
            # for debug
            vis = False
            if vis:
                filename = str(random.randrange(1,500))
                tmpimg = cropped_img.astype(np.float32).copy()
                tmpimg = self.config.denormalize_input(tmpimg)
                tmpimg = tmpimg.astype(np.uint8).copy()
                tmpkps = np.zeros((3,self.config.num_kps))
                tmpkps[:2,:] = target_coord.transpose(1,0)
                tmpkps[2,:] = target_valid
                tmpimg = self.config.vis_keypoints(tmpimg, tmpkps)
                cv2.imwrite(osp.join(self.config.vis_dir, filename + '_gt.jpg'), tmpimg)
            
            return [cropped_img,
                    target_coord, 
                    (target_valid > 0)]

        elif stage == 'test':
            # input_shape: (256, 192)
            trans = self.get_affine_transform(center, scale, rotation, (self.config.input_shape[1], self.config.input_shape[0])) # 아핀 변환 (카메라 기하학)
            cropped_img = cv2.warpAffine(img, trans, (self.config.input_shape[1], self.config.input_shape[0]), flags=cv2.INTER_LINEAR)
            #cropped_img = cropped_img[:,:, ::-1]
            cropped_img = self.config.normalize_input(cropped_img)
            # bbox에서 x,y,w,h 추출해서 center와 scale 계산 후 crop된 부분의 (왼쪽 아래 좌표, 오른쪽 위 좌표)로 crop_info를 return함
            #crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5])
            crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5])

            # cropped_img: 256*192*3, crop_info: 4 
            return [cropped_img, crop_info]

        else:
            # crop_info = np.asarray([0,0,w,h])
            crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5])
            return [img, crop_info]


