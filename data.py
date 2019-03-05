from imageio import imread
import numpy as np
import cv2
from tqdm import tqdm


def preprocess_image(path, crop_size, target_size):
	img = imread(path)

	if crop_size:
		cropped_img = img[
			(img.shape[0] // 2 - crop_size[0] // 2):(img.shape[0] // 2 + crop_size[0] // 2),
			(img.shape[1] // 2 - crop_size[1] // 2):(img.shape[1] // 2 + crop_size[1] // 2)
		]
	else:
		cropped_img = img

	resized_img = cv2.resize(cropped_img, dsize=target_size)

	return resized_img


def preprocess_images(paths, crop_size, target_size):
	imgs = []

	for path in tqdm(paths):
		imgs.append(preprocess_image(path, crop_size, target_size))

	return np.stack(imgs, axis=0)
