import numpy as np
import imageio
import cv2


def preprocess_image(path, crop_size=None, target_size=None):
	img = imageio.imread(path)

	if crop_size:
		img = img[
			(img.shape[0] // 2 - crop_size[0] // 2):(img.shape[0] // 2 + crop_size[0] // 2),
			(img.shape[1] // 2 - crop_size[1] // 2):(img.shape[1] // 2 + crop_size[1] // 2)
		]

	if target_size:
		img = cv2.resize(img, dsize=target_size)

	return img


def preprocess_images(paths, crop_size=None, target_size=None):
	imgs = []

	for path in paths:
		imgs.append(preprocess_image(path, crop_size, target_size))

	return np.stack(imgs, axis=0)
