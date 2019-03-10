import numpy as np
import imageio
import cv2

from keras.models import load_model


class FacePreprocessor:

	def __init__(self, segmentation_model_path, crop_size=None, target_size=None):
		self.__segmentation_model = load_model(segmentation_model_path)
		self.__crop_size = crop_size
		self.__target_size = target_size

	def preprocess_img(self, path):
		img = imageio.imread(path)

		if self.__crop_size:
			img = img[
				(img.shape[0] // 2 - self.__crop_size[0] // 2):(img.shape[0] // 2 + self.__crop_size[0] // 2),
				(img.shape[1] // 2 - self.__crop_size[1] // 2):(img.shape[1] // 2 + self.__crop_size[1] // 2)
			]

		if self.__target_size:
			img = cv2.resize(img, dsize=self.__target_size)

		mask = self.segment_face(img)

		img = img.astype(np.float64) / 255

		return img, mask

	def segment_face(self, img):
		img_shape = img.shape[0], img.shape[1]

		img = cv2.resize(img, dsize=(500, 500))
		img = img.astype(np.float64)

		img = img[..., ::-1]
		img -= np.array((104.00698793, 116.66876762, 122.67891434))

		mask = self.__segmentation_model.predict(np.expand_dims(img, axis=0))[0]

		mask = cv2.resize(mask, dsize=(img_shape[1], img_shape[0]))
		mask = np.clip(mask.argmax(axis=2), 0, 1).astype(np.float64)

		mask = cv2.GaussianBlur(mask, ksize=(7, 7), sigmaX=6)

		return mask

	def preprocess_imgs(self, paths):
		imgs = []
		masks = []

		for path in paths:
			try:
				img, mask = self.preprocess_img(path)

				imgs.append(img)
				masks.append(mask)

			except Exception as e:
				print('failed to preprocess img: %s (%s)' % (path, e))

		return np.stack(imgs, axis=0), np.stack(masks, axis=0)
