import os
import re
from abc import ABC, abstractmethod

import numpy as np
import imageio
import cv2
import PIL
import h5py

from keras.datasets import mnist


supported_datasets = [
	# 'mnist',
	# 'smallnorb',
	# 'dsprites',
	# 'noisy-dsprites',
	# 'color-dsprites',
	# 'scream-dsprites',
	'cars3d',
	# 'shapes3d'
]


def get_dataset(dataset_id, path=None):
	# if dataset_id == 'mnist':
	# 	return Mnist()

	# if dataset_id == 'smallnorb':
	# 	return SmallNorb(path)

	# if dataset_id == 'dsprites':
	# 	return DSprites(path)
	#
	# if dataset_id == 'noisy-dsprites':
	# 	return NoisyDSprites(path)
	#
	# if dataset_id == 'color-dsprites':
	# 	return ColorDSprites(path)
	#
	# if dataset_id == 'scream-dsprites':
	# 	return ScreamDSprites(path)

	if dataset_id == 'cars3d':
		return Cars3D(path)

	# if dataset_id == 'shapes3d':
	# 	return Shapes3D(path)

	raise Exception('unsupported dataset: %s' % dataset_id)


class DataSet(ABC):

	def __init__(self, base_dir=None):
		super().__init__()
		self._base_dir = base_dir

	@abstractmethod
	def read_images(self):
		pass


# class Mnist(DataSet):
#
# 	def __init__(self):
# 		super().__init__()
#
# 	def read_images(self):
# 		(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# 		imgs = dict()
#
# 		for digit in range(10):
# 			digit_imgs = x_train[y_train == digit]
# 			digit_imgs = np.stack([cv2.resize(digit_imgs[i], dsize=(64, 64)) for i in range(digit_imgs.shape[0])], axis=0)
# 			imgs[digit] = np.expand_dims(digit_imgs, axis=-1)
#
# 		return imgs


# class SmallNorb(DataSet):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
#
# 	def __list_image_paths(self):
# 		img_paths = dict()
#
# 		regex = re.compile('(\d+)_(\w+)_(\d+)_azimuth(\d+)_elevation(\d+)_lighting(\d+)_(\w+).jpg')
# 		for file_name in os.listdir(self._base_dir):
# 			img_path = os.path.join(self._base_dir, file_name)
# 			img_id, category, instance, azimuth, elevation, lighting, lt_rt = regex.match(file_name).groups()
#
# 			object_id = '_'.join((category, instance, elevation, lighting, lt_rt))
# 			if object_id not in img_paths:
# 				img_paths[object_id] = list()
#
# 			img_paths[object_id].append(img_path)
#
# 		return img_paths
#
# 	def read_images(self):
# 		imgs = dict()
#
# 		for object_id, object_img_paths in self.__list_image_paths().items():
# 			object_imgs = [imageio.imread(path) for path in object_img_paths]
# 			object_imgs = [cv2.resize(img, dsize=(64, 64)) for img in object_imgs]
# 			object_imgs = np.stack(object_imgs, axis=0)
#
# 			imgs[object_id] = np.expand_dims(object_imgs, axis=-1)
#
# 		return imgs


# class DSprites(DataSet):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
#
# 		self.__data_path = os.path.join(base_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
#
# 	def read_images(self):
# 		data = np.load(self.__data_path)
# 		data_imgs = data['imgs']
# 		data_classes = data['latents_classes']
#
# 		imgs = dict()
#
# 		for shape in range(3):
# 			imgs[str(shape)] = (data_imgs[data_classes[:, 1] == shape] * 255)[..., np.newaxis]
#
# 		return imgs
#
#
# class NoisyDSprites(DSprites):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
#
# 	def read_images(self):
# 		imgs = super().read_images()
#
# 		for shape, shape_imgs in imgs.items():
# 			noise = np.random.uniform(0, 1, size=(shape_imgs.shape[0], 64, 64, 3))
# 			imgs[shape] = (np.minimum(shape_imgs / 255.0 + noise, 1.) * 255).astype(np.uint8)
#
# 		return imgs
#
#
# class ColorDSprites(DSprites):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
#
# 	def read_images(self):
# 		imgs = super().read_images()
#
# 		for shape, shape_imgs in imgs.items():
# 			color = np.random.uniform(0.5, 1, size=(shape_imgs.shape[0], 1, 1, 3))
# 			color = np.tile(color, reps=(1, 64, 64, 1))
#
# 			imgs[shape] = (shape_imgs * color).astype(np.uint8)
#
# 		return imgs
#
#
# class ScreamDSprites(DSprites):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
#
# 		self.__scream_path = os.path.join(base_dir, 'scream.jpg')
#
# 		scream_img = PIL.Image.open(self.__scream_path)
# 		scream_img.thumbnail((350, 274, 3))
#
# 		self.__scream_img = np.array(scream_img) / 255.0
#
# 	def __apply_background(self, img):
# 		x_crop = np.random.randint(0, self.__scream_img.shape[0] - 64)
# 		y_crop = np.random.randint(0, self.__scream_img.shape[1] - 64)
#
# 		background = (self.__scream_img[x_crop:x_crop + 64, y_crop:y_crop + 64] + np.random.uniform(0, 1, size=3)) / 2
#
# 		mask = (img == 255).squeeze()
# 		background[mask] = 1 - background[mask]
#
# 		return (background * 255).astype(np.uint8)
#
# 	def read_images(self):
# 		imgs = super().read_images()
#
# 		for shape, shape_imgs in imgs.items():
# 			imgs[shape] = np.stack([self.__apply_background(shape_imgs[i]) for i in range(shape_imgs.shape[0])], axis=0)
#
# 		return imgs


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


# class Shapes3D(DataSet):
#
# 	def __init__(self, base_dir):
# 		super().__init__(base_dir)
#
# 		self.__data_path = os.path.join(base_dir, '3dshapes.h5')
#
# 	def __img_index(self, floor_hue, wall_hue, object_hue, scale, shape, orientation):
# 		return (
# 			floor_hue * 10 * 10 * 8 * 4 * 15
# 			+ wall_hue * 10 * 8 * 4 * 15
# 			+ object_hue * 8 * 4 * 15
# 			+ scale * 4 * 15
# 			+ shape * 15
# 			+ orientation
# 		)
#
# 	def read_images(self):
# 		with h5py.File(self.__data_path, 'r') as data:
# 			data_imgs = data['images']
#
# 			img_idxs = dict()
# 			for floor_hue in range(10):
# 				for wall_hue in range(10):
# 					for object_hue in range(10):
# 						for scale in range(8):
# 							for shape in range(4):
# 								for orientation in range(15):
# 									img_idx = self.__img_index(floor_hue, wall_hue, object_hue, scale, shape, orientation)
#
# 									if shape not in img_idxs:
# 										img_idxs[shape] = list()
#
# 									img_idxs[shape].append(img_idx)
#
# 			imgs = dict()
# 			for shape, shape_img_idxs in img_idxs.items():
# 				imgs[shape] = data_imgs[shape_img_idxs]
#
# 			return imgs
