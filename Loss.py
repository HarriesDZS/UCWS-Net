# -*-coding:utf-8-*-
"""
所有的损失函数
"""

import sys

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

device_id = 1


class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.loss_ce_v2 = torch.nn.CrossEntropyLoss(reduction='none').cuda(device=device_id)
        self.loss_ce = torch.nn.CrossEntropyLoss().cuda(device=device_id)
        self.back_ratio = 1


    def forward(self, preds1, cams1, preds1_back, preds2, cams2, y, index=0):

        cur_loss = 0

        current_predict = preds1[index].argmax(dim=1)
        current_prob = torch.softmax(preds1[index], dim=1)
        flag_predict = preds1[1 - index].detach()
        flag_predict = flag_predict.argmax(dim=1)

        for i in range(y.shape[0]):
          zero = torch.zeros(y[i].shape)
          zero = zero.type(torch.LongTensor)
          zero = zero.cuda(device=device_id)

          ce_loss_1 = self.loss_ce(preds1[index][i].unsqueeze(0), y[i].unsqueeze(0))
          ce_loss_2 = self.loss_ce(preds2[index][i].unsqueeze(0), y[i].unsqueeze(0))
          er_loss = torch.mean(
            torch.pow(cams1[index][i, 1:, :, :] - cams2[index][i, 1:, :, :], 2) * y[i].unsqueeze(-1).unsqueeze(-1))
          ce_back_loss = 0.5 * self.loss_ce_v2(preds1_back[index][i].unsqueeze(0), zero.unsqueeze(0))
          ce_back_loss = self.back_ratio * torch.mean(ce_back_loss * y[i])
          ce_loss = 0.5 * (ce_loss_1 + ce_loss_2)
          if current_predict[i] != flag_predict[i] and current_predict[i] == 0 and y[i] == 1:
              cur_loss +=  current_prob[i][1] * (ce_loss + er_loss + ce_back_loss)
              #cur_loss += (ce_loss + er_loss + ce_back_loss)

          else:
              cur_loss += (ce_loss + er_loss + ce_back_loss)



        same = (current_predict == flag_predict).float()
        target = cams1[1-index].detach()
        same_loss = torch.mean(
            torch.pow(cams1[index][:, 1:, :, :] - target[:, 1:, :, :], 2) * y.unsqueeze(-1).unsqueeze(
                -1).unsqueeze(
                -1)* same.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


        return cur_loss/y.shape[0] + same_loss


class CombineLossYiDaYi(nn.Module):
    def __init__(self):
        super(CombineLossYiDaYi, self).__init__()
        self.loss_ce_v2 = torch.nn.CrossEntropyLoss(reduction='none').cuda(device=device_id)
        self.loss_ce = torch.nn.CrossEntropyLoss().cuda(device=device_id)
        self.back_ratio = 1


    def forward(self, preds1, cams1, preds1_back, preds2, cams2, y, index=0):

        cur_loss = 0

        current_predict = preds1[index].argmax(dim=1)
        current_prob = torch.softmax(preds1[index], dim=1)
        flag_predict = preds1[1 - index].detach()
        flag_predict = flag_predict.argmax(dim=1)

        for i in range(y.shape[0]):
          zero = torch.zeros(y[i].shape)
          zero = zero.type(torch.LongTensor)
          zero = zero.cuda(device=device_id)

          ce_loss_1 = self.loss_ce(preds1[index][i].unsqueeze(0), y[i].unsqueeze(0))
          ce_loss_2 = self.loss_ce(preds2[index][i].unsqueeze(0), y[i].unsqueeze(0))
          er_loss = torch.mean(
            torch.pow(cams1[index][i, 1:, :, :] - cams2[index][i, 1:, :, :], 2) * y[i].unsqueeze(-1).unsqueeze(-1))
          ce_back_loss = 0.5 * self.loss_ce_v2(preds1_back[index][i].unsqueeze(0), zero.unsqueeze(0))
          ce_back_loss = self.back_ratio * torch.mean(ce_back_loss * y[i])
          ce_loss = 0.5 * (ce_loss_1 + ce_loss_2)
          if current_predict[i] != flag_predict[i] and current_predict[i] == 0 and y[i] == 1:
              #cur_loss +=  current_prob[i][1].detach() * (ce_loss + er_loss + ce_back_loss)
              cur_loss += current_prob[i][1].detach() * (ce_loss + er_loss + ce_back_loss)

          else:
              cur_loss += (ce_loss + er_loss + ce_back_loss)
              #cur_loss += (ce_loss + er_loss)



        same = (current_predict == flag_predict).float()
        target = cams1[1-index].detach()
        same_loss = torch.mean(
            torch.pow(cams1[index][:, 1:, :, :] - target[:, 1:, :, :], 2) * y.unsqueeze(-1).unsqueeze(
                -1).unsqueeze(
                -1)* same.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


        return cur_loss/y.shape[0] + same_loss


