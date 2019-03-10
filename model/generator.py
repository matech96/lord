import random

import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):

	def __init__(self, imgs, batch_size, n_batches_per_epoch):
		self.__imgs = imgs
		self.__batch_size = batch_size
		self.__n_batches_per_epoch = n_batches_per_epoch

		self.__identity_ids = list(self.__imgs.keys())
		self.__img_shape = self.__imgs[self.__identity_ids[0]]['imgs'][0].shape

	def __len__(self):
		return self.__n_batches_per_epoch

	def __getitem__(self, item):
		source_imgs = np.empty(shape=(self.__batch_size, *self.__img_shape), dtype=np.float64)
		target_imgs = np.empty(shape=(self.__batch_size, *self.__img_shape), dtype=np.float64)

		identity_ids = random.choices(self.__identity_ids, k=self.__batch_size)

		for i, identity_id in enumerate(identity_ids):
			idx = np.random.randint(0, self.__imgs[identity_id]['imgs'].shape[0], size=2)

			source_imgs[i] = self.__imgs[identity_id]['imgs'][idx[0]] * self.__imgs[identity_id]['masks'][idx[0]][..., np.newaxis]
			target_imgs[i] = self.__imgs[identity_id]['imgs'][idx[1]] * self.__imgs[identity_id]['masks'][idx[1]][..., np.newaxis]

		x = [source_imgs, target_imgs]
		y = source_imgs

		return x, y
