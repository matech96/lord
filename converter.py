import argparse
import os
import imageio
import pickle

import numpy as np

import dataset
from assets import AssetManager
from model.network import FaceConverter
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	identity_map = img_dataset.get_identity_map()

	imgs = dict()
	for object_id in identity_map.keys():
		imgs[object_id] = np.stack([imageio.imread(path) for path in identity_map[object_id]], axis=0)

	with open(assets.get_preprocess_file_path(args.data_name), 'wb') as fd:
		pickle.dump(imgs, fd)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	with open(assets.get_preprocess_file_path(args.data_name), 'rb') as fd:
		imgs = pickle.load(fd)

	face_converter = FaceConverter.build(
		img_shape=default_config['img_shape'],

		content_dim=args.content_dim,

		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim']
	)

	face_converter.train(
		imgs=imgs,
		batch_size=default_config['batch_size'] * args.gpus,

		n_epochs=default_config['n_epochs'],
		n_iterations_per_epoch=default_config['n_iterations_per_epoch'],
		n_epochs_per_checkpoint=default_config['n_epochs_per_checkpoint'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	face_converter.save(model_dir)


def convert(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	prediction_dir = assets.create_prediction_dir(args.model_name)

	face_converter = FaceConverter.load(model_dir)

	target_img = imageio.imread(args.target_img_path)
	target_img = target_img.astype(np.float64) / 255

	for source_img_path in args.source_img_paths:
		source_img = imageio.imread(source_img_path)
		source_img = source_img.astype(np.float64) / 255

		converted_img = face_converter.converter.predict([
			source_img[np.newaxis, ..., np.newaxis], target_img[np.newaxis, ..., np.newaxis]
		])[0, ..., 0]

		merged_img = np.concatenate((source_img, target_img, converted_img), axis=1)
		imageio.imwrite(os.path.join(prediction_dir, os.path.basename(source_img_path)), merged_img)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-cd', '--content-dim', type=int, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	convert_parser = action_parsers.add_parser('convert')
	convert_parser.add_argument('-sp', '--source-img-paths', type=str, nargs='+', required=True)
	convert_parser.add_argument('-tp', '--target-img-path', type=str, required=True)
	convert_parser.add_argument('-mn', '--model-name', type=str, required=True)
	convert_parser.set_defaults(func=convert)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()