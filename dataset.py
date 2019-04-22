import os
import re
from abc import ABC, abstractmethod


supported_datasets = ['smallnorb', 'celeba', 'vggface2']


def get_dataset(dataset_id, path):
	if dataset_id == 'smallnorb':
		return SmallNorb(path)

	if dataset_id == 'celeba':
		return CelebA(path)

	if dataset_id == 'vggface2':
		return VggFace2(path)

	raise Exception('unsupported dataset: %s' % dataset_id)


class DataSet(ABC):

	def __init__(self, base_dir):
		super().__init__()
		self._base_dir = base_dir

	@abstractmethod
	def get_identity_map(self):
		pass


class SmallNorb(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def get_identity_map(self):
		identity_map = dict()

		regex = re.compile('(\d+)_(\w+)_(\d+)_azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
		for file_name in os.listdir(self._base_dir):
			img_path = os.path.join(self._base_dir, file_name)
			img_id, category, instance, azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()

			object_id = '%s_%s' % (category, instance)
			if object_id not in identity_map:
				identity_map[object_id] = list()

			identity_map[object_id].append(img_path)

		return identity_map


class CelebA(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__imgs_dir = os.path.join(self._base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self._base_dir, 'Anno', 'identity_CelebA.txt')

	def get_identity_map(self):
		with open(self.__identity_map_path, 'r') as fd:
			lines = fd.read().splitlines()

		identity_map = dict()
		for line in lines:
			img_name, identity = line.split(' ')
			img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')

			if identity not in identity_map:
				identity_map[identity] = list()

			identity_map[identity].append(img_path)

		return identity_map


class VggFace2(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__imgs_dir = os.path.join(self._base_dir, 'train.cropped')

	def get_identity_map(self):
		identity_ids = os.listdir(self.__imgs_dir)

		identity_map = dict()
		for identity in identity_ids:
			identity_map[identity] = [
				os.path.join(self.__imgs_dir, identity, f)
				for f in os.listdir(os.path.join(self.__imgs_dir, identity))
			]

		return identity_map
