# -*-coding:utf-8-*-

import os
import glob

import numpy as np
from PIL import Image
import kornia.augmentation as F

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import subprocess
import random
import kornia as K

from PIL import Image, ImageOps
from random import random, randint, uniform, choice

class Rotate90(F.AugmentationBase2D):

	def __init__(self, return_transform = False):
		super(Rotate90, self).__init__()

	def generate_parameters(self, input_shape):

		angles_rad = torch.randint(1, 4, (input_shape[0],)) * K.pi/2.
		angles_deg = K.rad2deg(angles_rad)
		return dict(angles=angles_deg)

	def compute_transformation(self, input, params):
	  # compute transformation
		B, C ,H ,W = input.shape
		angles = params['angles'].type_as(input)
		center = torch.tensor([[W / 2, H / 2]]*B).type_as(input)
		transform = K.get_rotation_matrix2d(
		center, angles, torch.ones_like(center))
		return transform

	def apply_transform(self, input, params):

		# compute transformation
		B, C, H, W = input.shape
		transform = self.compute_transformation(input, params)
		# apply transformation and return
		output = K.warp_affine(input, transform, (H, W))
		return output


class Flip(nn.Module):
	def __init__(self):
		super(Flip, self).__init__()
		self.transform_hflip = F.RandomHorizontalFlip(p=1.0)
		self.transform_vflip = F.RandomVerticalFlip(p=1.0)

	def forward(self, input_tensor, other_tensor=None):
		k = random()
		if k >= 0.5:
			self.transform = self.transform_hflip
			transformation_param = self.transform_hflip.generate_parameters(input_tensor.shape)
		else:
			self.transform = self.transform_vflip
			transformation_param = self.transform_vflip.generate_parameters(input_tensor.shape)

		input_tensor = self.transform(input_tensor, transformation_param)
		if other_tensor is not None:
			if isinstance(other_tensor, list):
				for i in range(len(other_tensor)):
					other_tensor[i] = self.transform(other_tensor[i], transformation_param)
			else:
				other_tensor = self.transform(other_tensor, transformation_param)
			return input_tensor, other_tensor
		return input_tensor

class Rotate(nn.Module):
	def __init__(self):
		super(Rotate, self).__init__()
		self.transform = Rotate90()

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape)
		input_tensor = self.transform(input_tensor, transformation_param)
		if other_tensor is not None:
			if isinstance(other_tensor, list):
				for i in range(len(other_tensor)):
					other_tensor[i] = self.transform(other_tensor[i], transformation_param)
			else:
				other_tensor = self.transform(other_tensor, transformation_param)
			return input_tensor, other_tensor
		return input_tensor

class Translate(nn.Module):
	def __init__(self):
		super(Translate, self).__init__()
		self.transform = F.RandomAffine(degrees=0, translate=(0.3,0.3), align_corners=True)

	def forward(self, input_tensor, other_tensor=None):

		transformation_param = self.transform.generate_parameters(input_tensor.shape)
		input_tensor = self.transform(input_tensor, transformation_param)
		if other_tensor is not None:
			if isinstance(other_tensor, list):
				for i in range(len(other_tensor)):
					other_tensor[i] = self.transform(other_tensor[i], transformation_param)
			else:
				other_tensor = self.transform(other_tensor, transformation_param)
			return input_tensor, other_tensor

		return input_tensor


class Resize(nn.Module):
	def __init__(self):
		super(Resize, self).__init__()
		self.sizes = [i for i in range(192,296, 8)]

	def forward(self, input_tensor, other_tensor=None):
		output_size  = choice(self.sizes)
		input_tensor = nn.functional.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners= True)
		if other_tensor is not None:
			other_tensor = nn.functional.interpolate(other_tensor, size=output_size, mode='bilinear', align_corners= True)
			return input_tensor, other_tensor

		return input_tensor


class All(nn.Module):
	def __init__(self):
		super(All, self).__init__()
		self.transform1 = Flip()
		self.transform2 = Rotate()
		self.transform3 = Translate()
		self.transform4 = Resize()
		self.transforms = [self.transform1, self.transform3]
		#self.transforms = [self.transform1, self.transform2, self.transform3, self.transform4]



	def forward(self, input_tensor, other_tensor=None):
		apply_transform = choice(self.transforms)
		if other_tensor is not None:
			input_tensor, other_tensor = apply_transform(input_tensor, other_tensor)
			return input_tensor, other_tensor
		return apply_transform(input_tensor)