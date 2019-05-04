import os
import re
from abc import ABC, abstractmethod

import numpy as np
import imageio


supported_datasets = ['smallnorb', 'dsprites', 'cars3d']


def get_dataset(dataset_id, path):
	if dataset_id == 'smallnorb':
		return SmallNorb(path)

	if dataset_id == 'dsprites':
		return DSprites(path)

	if dataset_id == 'cars3d':
		return Cars3D(path)

	raise Exception('unsupported dataset: %s' % dataset_id)


class DataSet(ABC):

	def __init__(self, base_dir):
		super().__init__()
		self._base_dir = base_dir

	@abstractmethod
	def read_images(self):
		pass


class SmallNorb(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_image_paths(self):
		img_paths = dict()

		regex = re.compile('(\d+)_(\w+)_(\d+)_azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
		for file_name in os.listdir(self._base_dir):
			img_path = os.path.join(self._base_dir, file_name)
			img_id, category, instance, azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()

			object_id = '_'.join((category, instance, elevation, lighting, lt_rt))
			if object_id not in img_paths:
				img_paths[object_id] = list()

			img_paths[object_id].append(img_path)

		return img_paths

	def read_images(self):
		imgs = dict()

		for object_id, object_img_paths in self.__list_image_paths().items():
			imgs[object_id] = np.stack([imageio.imread(path)[..., np.newaxis] for path in object_img_paths], axis=0)

		return imgs


class DSprites(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

	def read_images(self):
		data = np.load(self.__data_path)
		data_imgs = data['imgs']
		data_classes = data['latents_classes']

		imgs = dict()

		for shape in range(3):
			imgs[str(shape)] = (data_imgs[data_classes[:, 1] == shape] * 255)[..., np.newaxis]

		return imgs


class Cars3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_image_paths(self):
		img_paths = dict()

		regex = re.compile('elevation(\d+)_azimuth(\d+)_object(\d+).png')
		for file_name in os.listdir(self._base_dir):
			img_path = os.path.join(self._base_dir, file_name)
			elevation, azimuth, object_id = regex.match(file_name).groups()

			if object_id not in img_paths:
				img_paths[object_id] = list()

			img_paths[object_id].append(img_path)

		return img_paths

	def read_images(self):
		imgs = dict()

		for object_id, object_img_paths in self.__list_image_paths().items():
			imgs[object_id] = np.stack([imageio.imread(path) for path in object_img_paths], axis=0)

		return imgs
