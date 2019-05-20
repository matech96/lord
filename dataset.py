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
	'mnist',
	'smallnorb',
	# 'dsprites',
	# 'noisy-dsprites',
	# 'color-dsprites',
	# 'scream-dsprites',
	'cars3d',
	'shapes3d',
	'celeba',
	'kth'
]


def get_dataset(dataset_id, path=None):
	if dataset_id == 'mnist':
		return Mnist()

	if dataset_id == 'smallnorb':
		return SmallNorb(path)

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

	if dataset_id == 'shapes3d':
		return Shapes3D(path)

	if dataset_id == 'celeba':
		return CelebA(path)

	if dataset_id == 'kth':
		return KTH(path)

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
