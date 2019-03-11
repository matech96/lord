import numpy as np
import imageio
import cv2


class FacePreprocessor:

	def __init__(self, crop_size=None, target_size=None):
		self.__crop_size = tuple(crop_size) if crop_size else None
		self.__target_size = tuple(target_size) if target_size else None

	def preprocess_img(self, path):
		img = imageio.imread(path)

		if self.__crop_size:
			img = img[
				(img.shape[0] // 2 - self.__crop_size[0] // 2):(img.shape[0] // 2 + self.__crop_size[0] // 2),
				(img.shape[1] // 2 - self.__crop_size[1] // 2):(img.shape[1] // 2 + self.__crop_size[1] // 2)
			]

		if self.__target_size:
			img = cv2.resize(img, dsize=self.__target_size)

		return img

	def preprocess_imgs(self, paths):
		imgs = []

		for path in paths:
			try:
				img = self.preprocess_img(path)
				imgs.append(img)

			except Exception as e:
				print('failed to preprocess img: %s (%s)' % (path, e))

		return np.stack(imgs, axis=0)
