import os
import re
import glob
from abc import ABC, abstractmethod

import numpy as np
import imageio
import cv2
import h5py

from keras.datasets import mnist
from scipy.ndimage.filters import gaussian_filter


supported_datasets = [
	'mnist',
	'smallnorb',
	'cars3d',
	'shapes3d',
	'celeba',
	'kth',
	'edges2shoes'
]


def get_dataset(dataset_id, path=None):
	if dataset_id == 'mnist':
		return Mnist()

	if dataset_id == 'smallnorb':
		return SmallNorb(path)

	if dataset_id == 'cars3d':
		return Cars3D(path)

	if dataset_id == 'shapes3d':
		return Shapes3D(path)

	if dataset_id == 'celeba':
		return CelebA(path)

	if dataset_id == 'kth':
		return KTH(path)

	if dataset_id == 'edges2shoes':
		return Edges2Shoes(path)

	raise Exception('unsupported dataset: %s' % dataset_id)


class DataSet(ABC):

	def __init__(self, base_dir=None):
		super().__init__()
		self._base_dir = base_dir

	@abstractmethod
	def read_images(self):
		pass


class Mnist(DataSet):

	def __init__(self):
		super().__init__()

	def read_images(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		x = np.concatenate((x_train, x_test), axis=0)
		y = np.concatenate((y_train, y_test), axis=0)

		imgs = np.stack([cv2.resize(x[i], dsize=(64, 64)) for i in range(x.shape[0])], axis=0)
		imgs = np.expand_dims(imgs, axis=-1)

		identities = y
		poses = np.empty(shape=(x.shape[0], ), dtype=np.uint32)

		return imgs, identities, poses


class SmallNorb(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def __list_imgs(self):
		img_paths = []
		identity_ids = []
		pose_ids = []

		regex = re.compile('azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
		for category in os.listdir(self._base_dir):
			for instance in os.listdir(os.path.join(self._base_dir, category)):
				for file_name in os.listdir(os.path.join(self._base_dir, category, instance)):
					img_path = os.path.join(self._base_dir, category, instance, file_name)
					azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()

					identity_id = '_'.join((category, instance, elevation, lighting, lt_rt))
					pose_id = azimuth

					img_paths.append(img_path)
					identity_ids.append(identity_id)
					pose_ids.append(pose_id)

		return img_paths, identity_ids, pose_ids

	def read_images(self):
		img_paths, identity_ids, pose_ids = self.__list_imgs()

		unique_identity_ids = list(set(identity_ids))
		unique_pose_ids = list(set(pose_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 1), dtype=np.uint8)
		identities = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		poses = np.empty(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])
			imgs[i, :, :, 0] = cv2.resize(img, dsize=(64, 64))

			identities[i] = unique_identity_ids.index(identity_ids[i])
			poses[i] = unique_pose_ids.index(pose_ids[i])

		return imgs, identities, poses


class Cars3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, 'cars3d.npz')

	def read_images(self):
		imgs = np.load(self.__data_path)['imgs']
		identities = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)
		poses = np.empty(shape=(imgs.shape[0], ), dtype=np.uint32)

		for elevation in range(4):
			for azimuth in range(24):
				for object_id in range(183):
					img_idx = elevation * 24 * 183 + azimuth * 183 + object_id

					identities[img_idx] = object_id
					poses[img_idx] = elevation * 24 + azimuth

		return imgs, identities, poses


class Shapes3D(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__data_path = os.path.join(base_dir, '3dshapes.h5')

	def __img_index(self, floor_hue, wall_hue, object_hue, scale, shape, orientation):
		return (
			floor_hue * 10 * 10 * 8 * 4 * 15
			+ wall_hue * 10 * 8 * 4 * 15
			+ object_hue * 8 * 4 * 15
			+ scale * 4 * 15
			+ shape * 15
			+ orientation
		)

	def read_images(self):
		with h5py.File(self.__data_path, 'r') as data:
			imgs = data['images'][:]
			identities = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
			pose_ids = dict()

			for floor_hue in range(10):
				for wall_hue in range(10):
					for object_hue in range(10):
						for scale in range(8):
							for shape in range(4):
								for orientation in range(15):
									img_idx = self.__img_index(floor_hue, wall_hue, object_hue, scale, shape, orientation)
									pose_id = '_'.join((str(floor_hue), str(wall_hue), str(object_hue), str(scale), str(orientation)))

									identities[img_idx] = shape
									pose_ids[img_idx] = pose_id

			unique_pose_ids = list(set(pose_ids.values()))
			poses = np.empty(shape=(imgs.shape[0],), dtype=np.uint32)
			for img_idx, pose_id in pose_ids.items():
				poses[img_idx] = unique_pose_ids.index(pose_id)

			return imgs, identities, poses


class CelebA(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__imgs_dir = os.path.join(self._base_dir, 'Img', 'img_align_celeba_png.7z', 'img_align_celeba_png')
		self.__identity_map_path = os.path.join(self._base_dir, 'Anno', 'identity_CelebA.txt')

	def __list_imgs(self):
		with open(self.__identity_map_path, 'r') as fd:
			lines = fd.read().splitlines()

		img_paths = []
		identity_ids = []

		for line in lines:
			img_name, identity_id = line.split(' ')
			img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.png')

			img_paths.append(img_path)
			identity_ids.append(identity_id)

		return img_paths, identity_ids

	def read_images(self, crop_size=(128, 128), target_size=(64, 64)):
		img_paths, identity_ids = self.__list_imgs()

		unique_identity_ids = list(set(identity_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
		identities = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		poses = np.zeros(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])

			if crop_size:
				img = img[
					(img.shape[0] // 2 - crop_size[0] // 2):(img.shape[0] // 2 + crop_size[0] // 2),
					(img.shape[1] // 2 - crop_size[1] // 2):(img.shape[1] // 2 + crop_size[1] // 2)
				]

			if target_size:
				img = cv2.resize(img, dsize=target_size)

			imgs[i] = img
			identities[i] = unique_identity_ids.index(identity_ids[i])

		return imgs, identities, poses


class KTH(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

		self.__action_dir = os.path.join(self._base_dir, 'handwaving')
		self.__condition = 'd4'

	def __list_imgs(self):
		img_paths = []
		identity_ids = []

		for identity in os.listdir(self.__action_dir):
			for f in os.listdir(os.path.join(self.__action_dir, identity, self.__condition)):
				img_paths.append(os.path.join(self.__action_dir, identity, self.__condition, f))
				identity_ids.append(identity)

		return img_paths, identity_ids

	def read_images(self):
		img_paths, identity_ids = self.__list_imgs()

		unique_identity_ids = list(set(identity_ids))

		imgs = np.empty(shape=(len(img_paths), 64, 64, 1), dtype=np.uint8)
		identities = np.empty(shape=(len(img_paths), ), dtype=np.uint32)
		poses = np.zeros(shape=(len(img_paths), ), dtype=np.uint32)

		for i in range(len(img_paths)):
			imgs[i, :, :, 0] = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2GRAY)
			identities[i] = unique_identity_ids.index(identity_ids[i])

		return imgs, identities, poses


class Edges2Shoes(DataSet):

	def __init__(self, base_dir):
		super().__init__(base_dir)

	def read_images(self):
		img_paths = glob.glob(os.path.join(self._base_dir, '*', '*.jpg'))

		edge_imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
		shoe_imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)

		for i in range(len(img_paths)):
			img = imageio.imread(img_paths[i])
			img = gaussian_filter(img, sigma=1)

			img = cv2.resize(img, dsize=(128, 64))

			edge_imgs[i] = img[:, :64, :]
			shoe_imgs[i] = img[:, 64:, :]

		imgs = np.concatenate((edge_imgs, shoe_imgs), axis=0)
		identities = np.concatenate((
			np.zeros(shape=(edge_imgs.shape[0], ), dtype=np.uint8),
			np.ones(shape=(shoe_imgs.shape[0], ), dtype=np.uint8)
		), axis=0)

		poses = np.zeros_like(identities)

		return imgs, identities, poses
