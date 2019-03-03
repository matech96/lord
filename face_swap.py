import argparse
import pickle
import os

import numpy as np

from dataset import CelebA
import data

from assets import AssetManager
from config import img_config

from model import FaceConverter


def preprocess_multi_identity(args):
	assets = AssetManager(args.base_dir)

	dataset = CelebA(args.dataset_path)
	identity_map = dataset.get_identity_map()

	imgs = dict()
	for i, identity in enumerate(identity_map.keys()):
		print('\rpreprocessing identity: #%d' % i, end='')
		imgs[identity] = data.preprocess_images(identity_map[identity], img_config['crop_size'], img_config['target_size'])

	with open(assets.get_multi_identity_preprocess_file_path(args.data_name), 'wb') as fd:
		pickle.dump(imgs, fd)


def preprocess_single_identity(args):
	assets = AssetManager(args.base_dir)

	img_paths = [os.path.join(args.dataset_path, f) for f in os.listdir(args.dataset_path)]
	imgs = data.preprocess_images(img_paths, crop_size=None, target_size=img_config['target_size'])

	np.savez(assets.get_single_identity_preprocess_file_path(args.data_name), imgs=imgs)


def extract_face_images(args):
	data.preprocess_videos(args.video_dir, args.out_dir, args.num_of_processes)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model)
	checkpoints_dir = assets.recreate_model_checkpoint_dir(args.model)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model)

	with open(assets.get_multi_identity_preprocess_file_path(args.train_data_name), 'rb') as fd:
		train_images = pickle.load(fd)

		for k in train_images.keys():
			train_images[k] = (train_images[k] / 255) * 2 - 1

	with np.load(assets.get_single_identity_preprocess_file_path(args.anchor_data_name)) as d:
		anchor_images = d['imgs']

		anchor_images = (anchor_images / 255) * 2 - 1

	# with np.load(assets.get_single_identity_preprocess_file_path(args.validation_data_name_a)) as d:
	# 	validation_images_a = d['imgs']
	# 	validation_images_a = (validation_images_a / 255) * 2 - 1
	#
	# with np.load(assets.get_single_identity_preprocess_file_path(args.validation_data_name_b)) as d:
	# 	validation_images_b = d['imgs']
	# 	validation_images_b = (validation_images_b / 255) * 2 - 1

	face_converter = FaceConverter.build(
		img_shape=(64, 64, 3),
		identity_dim=32,
		n_adain_layers=3,
		adain_dim=256,
		batch_size=16
	)

	face_converter.train(
		images=train_images,
		anchor_images=anchor_images,

		# validation_images_a=validation_images_a,
		# validation_images_b=validation_images_b,

		n_total_iterations=1000000,
		n_checkpoint_iterations=1000,
		n_log_iterations=100,

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir,
	)

	face_converter.save(model_dir)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_multi_identity_parser = action_parsers.add_parser('preprocess-multi-identity')
	preprocess_multi_identity_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_multi_identity_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_multi_identity_parser.set_defaults(func=preprocess_multi_identity)

	preprocess_single_identity_parser = action_parsers.add_parser('preprocess-single-identity')
	preprocess_single_identity_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_single_identity_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_single_identity_parser.set_defaults(func=preprocess_single_identity)

	extract_face_images_parser = action_parsers.add_parser('extract-face-images')
	extract_face_images_parser.add_argument('-vd', '--video-dir', type=str, required=True)
	extract_face_images_parser.add_argument('-od', '--out-dir', type=str, required=True)
	extract_face_images_parser.add_argument('-np', '--num-of-processes', type=int, default=8)
	extract_face_images_parser.set_defaults(func=extract_face_images)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-mn', '--model', type=str, required=True)
	train_parser.add_argument('-tdn', '--train-data-name', type=str, required=True)
	train_parser.add_argument('-adn', '--anchor-data-name', type=str, required=True)
	# train_parser.add_argument('-vdna', '--validation-data-name-a', type=str, required=True)
	# train_parser.add_argument('-vdnb', '--validation-data-name-b', type=str, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
