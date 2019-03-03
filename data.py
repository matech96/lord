import os
from uuid import uuid4

from multiprocessing.pool import ThreadPool
import functools

from imageio import imread, imwrite
import numpy as np
import cv2
from tqdm import tqdm

from facedetection.face_detection import FaceDetector
from mediaio.video_io import VideoFileReader


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


def preprocess_video(video_path, out_dir, saving_ratio=0.05):
	face_detector = FaceDetector()
	video_name = os.path.splitext(os.path.basename(video_path))[0]

	with VideoFileReader(video_path) as reader:
		while True:
			try:
				frame = reader.read_next_frame(convert_to_gray_scale=False)

				if np.random.rand() > saving_ratio:
					continue

				bounding_box = face_detector.detect_face(frame)

				s = max(bounding_box.get_height(), bounding_box.get_width())
				bounding_box.resize_equally(height=s, width=s)

				face = frame[
					bounding_box.top: bounding_box.top + bounding_box.get_height(),
					bounding_box.left: bounding_box.left + bounding_box.get_width()
				]

				imwrite(os.path.join(out_dir, video_name + '_%s.png' % uuid4()), face)

			except IndexError:
				return

			except Exception as e:
				print('failed to process current frame (%s). skipping' % e)
				continue


def preprocess_videos(video_dir, out_dir, n_processes):
	video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]

	with ThreadPool(processes=n_processes) as pool:
		pool.map(functools.partial(preprocess_video, out_dir=out_dir), video_paths)

