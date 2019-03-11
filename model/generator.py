import random

import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):

	def __init__(self, vgg, imgs, batch_size, n_batches_per_epoch):
		self.__vgg = vgg
		self.__imgs = imgs
		self.__batch_size = batch_size
		self.__n_batches_per_epoch = n_batches_per_epoch

		self.__identity_ids = list(self.__imgs.keys())
		self.__img_shape = self.__imgs[self.__identity_ids[0]][0].shape

	def __len__(self):
		return self.__n_batches_per_epoch

	def __getitem__(self, item):
		source_imgs = np.empty(shape=(self.__batch_size, *self.__img_shape), dtype=np.float64)
		target_imgs = np.empty(shape=(self.__batch_size, *self.__img_shape), dtype=np.float64)

		identity_ids = random.choices(self.__identity_ids, k=self.__batch_size)

		for i, identity_id in enumerate(identity_ids):
			idx = np.random.randint(0, self.__imgs[identity_id].shape[0], size=2)

			source_imgs[i] = self.__imgs[identity_id][idx[0]]
			target_imgs[i] = self.__imgs[identity_id][idx[1]]

		x = [source_imgs, target_imgs]
		y = self.__vgg.predict(source_imgs)

		return x, y
