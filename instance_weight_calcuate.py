# coding = utf-8

''''
计算weight
'''

import torch
import cv2
import numpy as np

'''
根据单张图片的肿瘤个数进行加权
'''
def calcute_tumor_weight(label):
    label = label.cpu().numpy()
    weight_list = []
    for i in range(label.shape[0]):
        data = label[i]
        sum_data = data.sum()
        if sum_data == 0:
            weight_list.append(np.ones(data.shape))
            continue
        label_copy = data * 255
        label_copy = label_copy.astype(np.uint8)
        contours, _ = cv2.findContours(label_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tumor_number = len(contours)
        weight = np.ones(data.shape)
        weight = weight * (1 - data)
        for counter in contours:
            data_list = []
            for t in range(counter.shape[0]):
                j = counter[t][0]
                data_list.append(j)
            rect = cv2.minAreaRect(np.array([data_list], np.int32))
            box =  cv2.boxPoints(rect)   # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)
            temp = np.zeros(label_copy.shape).astype(np.uint8)
            cv2.fillPoly(temp, [box], 1)

            sum_temp = (temp * data).sum()
            avg_weight = sum_data / (sum_temp * tumor_number)
            weight = weight + (avg_weight * (temp*data))
        weight_list.append(weight)

    weight_list = np.array(weight_list)
    weight_list = torch.from_numpy(weight_list)
    return weight_list

'''
对背景和肿瘤同步进行加权
'''
def calcute_tumor_weight_v2(label):
    label = label.cpu().numpy()
    weight_list = []
    for i in range(label.shape[0]):
        data = label[i]
        sum_data = data.sum()
        back_data = (1 - data).sum()
        total_data = back_data + sum_data
        avg_data = float(total_data) / 2
        if sum_data == 0:
            weight_list.append(np.ones(data.shape))
            continue
        label_copy = data * 255
        label_copy = label_copy.astype(np.uint8)
        contours, _ = cv2.findContours(label_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tumor_number = len(contours)
        weight = np.ones(data.shape)
        weight = weight * (1 - data) * (avg_data / back_data)
        for counter in contours:
            data_list = []
            for t in range(counter.shape[0]):
                j = counter[t][0]
                data_list.append(j)
            rect = cv2.minAreaRect(np.array([data_list], np.int32))
            box =  cv2.boxPoints(rect)   # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)
            temp = np.zeros(label_copy.shape).astype(np.uint8)
            cv2.fillPoly(temp, [box], 1)

            sum_temp = (temp * data).sum()
            avg_weight = avg_data / (sum_temp * tumor_number)
            weight = weight + (avg_weight * (temp*data))
        weight_list.append(weight)

    weight_list = np.array(weight_list)
    weight_list = torch.from_numpy(weight_list)
    return weight_list

'''
先对肿瘤进行整体加权，然后根据单张图片的肿瘤个数进行加权
'''
def calcute_tumor_weight_v3(label):
    label = label.cpu().numpy()
    weight_list = []
    for i in range(label.shape[0]):
        data = label[i]
        sum_data = data.sum()
        if sum_data == 0:
            weight_list.append(np.ones(data.shape))
            continue
        label_copy = data * 255
        label_copy = label_copy.astype(np.uint8)
        contours, _ = cv2.findContours(label_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tumor_number = len(contours)
        weight = np.ones(data.shape)
        weight = weight * (1 - data)
        for counter in contours:
            data_list = []
            for t in range(counter.shape[0]):
                j = counter[t][0]
                data_list.append(j)
            rect = cv2.minAreaRect(np.array([data_list], np.int32))
            box =  cv2.boxPoints(rect)   # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)
            temp = np.zeros(label_copy.shape).astype(np.uint8)
            cv2.fillPoly(temp, [box], 1)

            sum_temp = (temp * data).sum()
            avg_weight = sum_data / (sum_temp * tumor_number)
            weight = weight + (avg_weight * (temp*data) * 10)
        weight_list.append(weight)

    weight_list = np.array(weight_list)
    weight_list = torch.from_numpy(weight_list)
    return weight_list