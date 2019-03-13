import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):

	def __init__(self, imgs, batch_size, n_batches_per_epoch):
		self.__imgs = imgs
		self.__batch_size = batch_size
		self.__n_batches_per_epoch = n_batches_per_epoch

	def __len__(self):
		return self.__n_batches_per_epoch

	def __getitem__(self, item):
		idx = np.random.choice(self.__imgs.shape[0], size=self.__batch_size)

		imgs = self.__imgs[idx]
		imgs = imgs.astype(np.float64) / 255

		x = [imgs, imgs]
		y = imgs

		return x, y
