# coding = utf-8

'''
获取所有的ROI区域
'''

import json

import click
import cv2
#import nibabel as nib
import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19
from dataset.transform import MedicalTransform
from network import ResUNet
from utils.vis import imshow
from network import DenseUNet


def calc(seg, idx):
    bincount = np.bincount(seg.flatten())
    area = int(bincount[idx])

    value = []
    for i in range(seg.shape[0]):
        value.append(seg[i].max())
    value = np.array(value)

    slice_ = np.where(value > idx - 1)[0]
    num_slice = len(slice_)
    min_z = int(slice_.min())
    max_z = int(slice_.max()) + 1
    min_x = min_y = 10000
    max_x = max_y = -1
    for i in range(min_z, max_z):
        if seg[i].max() > idx - 1:
            mask = np.ma.masked_where(seg[i] > idx - 1, seg[i]).mask
            rect = cv2.boundingRect(mask.astype(np.uint8))
            min_x = min(min_x, rect[0])
            min_y = min(min_y, rect[1])
            max_x = max(max_x, rect[0] + rect[2])
            max_y = max(max_y, rect[1] + rect[3])

    roi = {'min_x': min_x, 'min_y': min_y, 'min_z': min_z,
           'max_x': max_x, 'max_y': max_y, 'max_z': max_z,
           'area': area, 'slice': num_slice}

    return roi


def get_roi_from_munet():

    roi_file = "/datasets/DongbeiDaxue/chengkunv2/roi.json"
    #with open(roi_file, 'r') as f:
    #    rois = json.load(f)
    rois = {}

    dataset = KiTS19("/datasets/DongbeiDaxue/chengkunv2", stack_num=3, spec_classes=[0, 1, 2], img_size=(512, 512),
                     use_roi=False, roi_file=None, roi_error_range=5,
                     train_transform=None, valid_transform=None)

    model_01 = DenseUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
    data = {'net': model_01}
    cp_file = Path(
        "/home/diaozhaoshuo/log/BeliefFunctionNN/chengkung/dongbeidaxue/munet/checkpoint_denseunet_all/best.pth")
    cp.load_params(data, cp_file, device='cpu')
    model_01 = model_01.cuda()
    model_01.eval()

    torch.set_grad_enabled(False)
    subset = dataset.train_dataset
    case_slice_indices = dataset.train_case_slice_indices
    type = "train"
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=1, sampler=sampler,
                         num_workers=1, pin_memory=True)

    vol_output = []
    vol_images = []

    case = 0
    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, labels, idx = data['image'].cuda(), data['label'], data['index']
            outputs = model_01(imgs)
            outputs = outputs['output']
            outputs = outputs.argmax(dim=1)
            outputs[outputs == 2] = 1

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            imgs = imgs.cpu().detach().numpy()
            idx = idx.numpy()

            vol_output.append(outputs)


            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]

                vol = vol_output[:vol_num_slice]
                kidney = calc(vol, idx=1)
                case_roi = {'kidney': kidney, 'vol':{'total_x':vol.shape[1], 'total_y':vol.shape[2], 'total_z':vol.shape[0]}}
                case_id = dataset.case_idx_to_case_id(case, 'train')
                rois[f'case_{case_id:05d}'] = case_roi
                with open(roi_file, 'w') as f:
                    json.dump(rois, f, indent=4, separators=(',', ': '))

                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)

    torch.set_grad_enabled(False)
    subset = dataset.valid_dataset
    case_slice_indices = dataset.valid_case_slice_indices
    type = "valid"
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=1, sampler=sampler,
                             num_workers=1, pin_memory=True)

    vol_label = []
    vol_output = []
    vol_images = []

    case = 0
    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, labels, idx = data['image'].cuda(), data['label'], data['index']
            outputs = model_01(imgs)
            outputs = outputs['output']
            outputs = outputs.argmax(dim=1)
            outputs[outputs == 2] = 1

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            imgs = imgs.cpu().detach().numpy()
            idx = idx.numpy()

            vol_output.append(outputs)

            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]

                vol = vol_output[:vol_num_slice]
                kidney = calc(vol, idx=1)
                case_roi = {'kidney': kidney,
                            'vol': {'total_x': vol.shape[1], 'total_y': vol.shape[2], 'total_z': vol.shape[0]}}
                case_id = dataset.case_idx_to_case_id(case, 'valid')
                rois[f'case_{case_id:05d}'] = case_roi
                with open(roi_file, 'w') as f:
                    json.dump(rois, f, indent=4, separators=(',', ': '))

                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)

def get_roi_from_gt():

    roi_file = "/datasets/DongbeiDaxue/chengkunv2/roi_gt.json"
    #with open(roi_file, 'r') as f:
    #    rois = json.load(f)
    rois = {}

    dataset = KiTS19("/datasets/DongbeiDaxue/chengkunv2", stack_num=3, spec_classes=[0, 1, 1], img_size=(512, 512),
                     use_roi=False, roi_file=None, roi_error_range=5,
                     train_transform=None, valid_transform=None)




    subset = dataset.valid_dataset
    case_slice_indices = dataset.valid_case_slice_indices
    type = "valid"
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=1, sampler=sampler,
                         num_workers=1, pin_memory=True)

    vol_label = []
    vol_images = []

    case = 0
    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, labels, idx = data['image'].cuda(), data['label'], data['index']


            labels = labels.cpu().detach().numpy()

            imgs = imgs.cpu().detach().numpy()
            idx = idx.numpy()

            vol_label.append(labels)
            vol_images.append(imgs)

            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_label = np.concatenate(vol_label, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]

                vol = vol_label[:vol_num_slice]
                kidney = calc(vol, idx=1)
                case_roi = {'kidney': kidney}
                case_id = dataset.case_idx_to_case_id(case, 'valid')
                rois[f'case_{case_id:05d}'] = case_roi
                with open(roi_file, 'w') as f:
                    json.dump(rois, f, indent=4, separators=(',', ': '))

                vol_label = [vol_label[vol_num_slice:]]
                case += 1
                pbar.update(1)

def diff_roi():
    gt_roi_file = "/datasets/DongbeiDaxue/chengkunv2/roi_gt.json"
    roi_file = "/datasets/DongbeiDaxue/chengkunv2/roi.json"

    with open(roi_file, 'r') as f:
        rois = json.load(f)

    with open(gt_roi_file, 'r') as f:
        gt_rois = json.load(f)


    for key in rois.keys():
       predict = rois[key]["kidney"]
       gt = gt_rois[key]["kidney"]
       print(key,
             gt['min_x']-predict['min_x'], gt['min_y']-predict['min_y'], gt['min_z']-predict['min_z'],
             predict['max_x']-gt['max_x'], predict['max_y']-gt['max_y'], predict['max_z']-gt['max_z'])


if __name__ == '__main__':
    get_roi_from_munet()