# coding = utf-8

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import os
from torch.utils import data
import numpy as np

from utils import get_absolute_project_dir


"""
should be revised according to your dataset
"""

class STSDataLoader(data.Dataset):
    def __init__(self, type="train", transform=None):
        """
        :param type: 数据类别：包括train和valid
        :param data_aug: 是否采用数据增强方法
        """
        self.root_dir = None
        self.project_dir = None

        (train_ids, valid_ids) = self.fetch_train_valid_ids()

        if type == "train":
             (self.data_map, self.idx_map) = self.fetch_data_map(ids=train_ids)
        else:
            (self.data_map, self.idx_map) = self.fetch_data_map(ids=valid_ids)

        self.pet_min = 0
        self.pet_max = 255
        self.ct_min = -260
        self.ct_max = 340

        self.transform = transform



    def __getitem__(self, item):
        (case_id,file_index) = self.calcuate_file_from_index(index=item)
        pet_file_name = os.path.join(self.root_dir, "STS_{}_PET.npy".format(case_id))
        pet_file = np.load(pet_file_name)
        pet_data = pet_file[file_index]
        pet_data = (pet_data-self.pet_min)/(self.pet_max-self.pet_min)
        #pet_data = (pet_data - np.min(pet_data)) / (np.max(pet_data) - np.min(pet_data))

        ct_file_name = os.path.join(self.root_dir, "STS_{}_CT.npy".format(case_id))
        ct_file = np.load(ct_file_name)
        ct_data = ct_file[file_index]
        ct_data[ct_data >= self.ct_max] = self.ct_max
        ct_data[ct_data <= self.ct_min] = self.ct_min
        ct_data = (ct_data-self.ct_min)/(self.ct_max-self.ct_min)
        #ct_data = (ct_data - np.min(ct_data)) / (np.max(ct_data) - np.min(ct_data))

        label_file_name = os.path.join(self.root_dir, "STS_{}_Label.npy".format(case_id))
        label_file = np.load(label_file_name)
        label_data = label_file[file_index]

        image = [ct_data, pet_data]
        image = np.stack(image, axis=2)
        image = image.astype("float32")

        data = {"image":image, "label":label_data}
        if self.transform is not None:
            data = self.transform(data)

        image = data["image"]
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))

        pet_data = image[1]
        ct_data = image[0]
        label_data = data["label"]

        pet_data = pet_data.reshape(1, pet_data.shape[0], pet_data.shape[1])
        ct_data = ct_data.reshape(1, ct_data.shape[0], ct_data.shape[1])

        mask = torch.tensor([1])


        return (case_id, pet_data, ct_data, label_data, mask)



    def __len__(self):
        max_length = 0
        for key in self.idx_map.keys():
            max_length = max(max_length, self.idx_map[key][1])
        return max_length


    def fetch_train_valid_ids(self):
        train_ids = []
        valid_ids = []
        with open(os.path.join(self.project_dir, "STS_train"), "r") as file:
            for line in file:
                train_ids.append(line.strip())
        with open(os.path.join(self.project_dir, "STS_valid"), "r") as file:
            for line in file:
                valid_ids.append(line.strip())
        return (train_ids, valid_ids)

    def fetch_data_map(self, ids):
        """
        获取两个map,data_map{key:case_id,valud:切片},idx_map{key:case_id, value:[起止切片]}
        :param ids:
        :return:
        """
        data_map = {}
        idx_map = {}
        begin_init = 0
        for id in ids:
            file_name = os.path.join(self.root_dir, "STS_{}_Label.npy".format(str(id)))
            label = np.load(file_name)
            data_map[id] = []
            for i in range(label.shape[0]):
                if label[i].sum() <= 0:
                    continue
                data_map[id].append(i)
            idx_map[id] = [begin_init, begin_init+len(data_map[id])]
            begin_init += len(data_map[id])
        return (data_map, idx_map)

    def calcuate_file_from_index(self, index):
        case_id = -1
        for key in self.idx_map.keys():
            if self.idx_map[key][0] <= index and self.idx_map[key][1] > index:
                case_id = key
                break
        file_index = self.data_map[case_id][index - self.idx_map[case_id][0]]
        return (case_id, file_index)


    def get_data_map(self):
        return self.data_map

    def get_idx_map(self):
        return  self.idx_map


