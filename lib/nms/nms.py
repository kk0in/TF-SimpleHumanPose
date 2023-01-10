# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .cpu_nms import cpu_nms
from .gpu_nms import gpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None): # g: 정답 좌표, d: 추정 좌표
    # None 2번 연속이라서 아래 값이 들어감
    
    if not isinstance(sigmas, np.ndarray):
        # keypoint 마다 존재하는 상수
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2

    xg = g[0::3] # index 0부터 3씩 증가시키면서 리스트의 마지막 요소까지 가져옴 -> keypoint  x 추출 
    yg = g[1::3] # keypoint y 추출
    vg = g[2::3] # 1 추출

    # print("g", g)
    # print("d", d)
    # exit()

    # d.shape = 나머지 cropped image 개수 (동일한 이미지 내) * 51
    # d가 []인 경우 -> d.shape = (0, )
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None: # in_vis_thre가 None이라서 여긴 들어가지 않음
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def oks_nms(kpts, scores, areas, thresh, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks (oks가 thresh보다 클 때는 제외됨. 나머지 위치는 keep에 들어감)
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    # len(kpts) == len(cropped data)
    if len(kpts) == 0:
        return []

    # scores = np.array([1, 2, 3, 4]) => order == [3 2 1 0]
    order = scores.argsort()[::-1] # scores를 내림차순으로 정렬하기 위한 인덱스 (score 개수만큼 있음)

    # print("order_before:", order) # scores==[0.69292538 0.54283401] => order==[0 1]
    # print("scores:", scores)
    # exit()

    # (***important***) nms process
    # 1. cropped_image 들을 17개 관절 평균 confidence score 기준으로 모두 내림차순 정렬  
    # 2. 맨 앞에 있는 keypoint set 하나를 기준으로 잡고, 다른 keypoint set들과 OKS 값을 구함 -> OKS가 threshold 이상인 keypoint set들은 제거 (keypoint set끼리 OKS가 높을수록, 즉 많이 겹칠수록 같은 사람을 검출하고 있다고 판단하기 때문)
    # 3. 해당 과정을 순차적으로 시행 (confident threshold가 높을수록, OKS threshold가 낮을수록 더 많은 keypoint set이 제거됨)
    keep = []
    while order.size > 0:
        i = order[0] # i = 0 -> 1
        keep.append(i) # keep = [0] -> [0 1]

        # print(kpts[order[1:]])
        # print(areas[i])
        # exit()

        # 하나를 기준으로 잡고 나머지 keypoint set과 OKS 계산
        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        # print("oks_ovr:", oks_ovr) 
        # OKS가 threshold 이상인 keypoint set들은 같은 대상의 keypoint를 나타내는 것으로 간주되지만, score가 더 낮으므로 자동으로 고려대상에서 빠짐
        inds = np.where(oks_ovr <= thresh)[0] # where -> 해당 조건이 만족하는 원소의 index를 [[index1, index2, ...]]로 반환, 그래서 [0]을 붙임
        # print("inds:", inds)
        order = order[inds + 1] # order[0]는 처음에 진행했으므로 1씩 더해줘서 index 0이 들어가는 것을 막음
        # print("order:", order)
        # exit()
    return keep

