import argparse
import os
import imageio

import numpy as np

import dataset
from data import FacePreprocessor

from assets import AssetManager
from model.network import FaceConverter
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	face_set = dataset.get_dataset(args.dataset, args.dataset_path)
	identity_map = face_set.get_identity_map()

	face_preprocessor = FacePreprocessor(args.crop_size, args.target_size)
	preprocessed_path = assets.get_preprocess_file_path(args.data_name)

	identity_ids = list(identity_map.keys())[args.head_identity_index:args.tail_identity_index]
	imgs = [face_preprocessor.preprocess_imgs(identity_map[identity]) for identity in identity_ids]

	np.savez(preprocessed_path, imgs=np.concatenate(imgs, axis=0))


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	imgs = np.load(assets.get_preprocess_file_path(args.data_name))['imgs'][:args.max_images]

	face_converter = FaceConverter.build(
		img_shape=default_config['img_shape'],

		content_dim=args.content_dim,
		identity_dim=args.identity_dim,

		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim'],

		use_vgg_face=(args.vgg_type == 'vgg-face')
	)

	face_converter.train(
		imgs=imgs,
		batch_size=default_config['batch_size'] * args.gpus,
		n_gpus=args.gpus,

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

	imgs = np.load(assets.get_preprocess_file_path(args.data_name))['imgs']

	for i in range(args.num_of_samples):
		idx = np.random.choice(imgs.shape[0], size=2, replace=False)
		source_img = imgs[idx[0]].astype(np.float64) / 255
		target_img = imgs[idx[1]].astype(np.float64) / 255

		converted_img = face_converter.converter.predict([source_img[np.newaxis, ...], target_img[np.newaxis, ...]])[0]
		merged_img = np.concatenate((source_img, target_img, converted_img), axis=1)

		imageio.imwrite(os.path.join(prediction_dir, '%d.png' % i), merged_img)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-ds', '--dataset', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.add_argument('-hi', '--head-identity-index', type=int, required=True)
	preprocess_parser.add_argument('-ti', '--tail-identity-index', type=int, required=True)
	preprocess_parser.add_argument('-cs', '--crop-size', type=int, nargs=2, required=False)
	preprocess_parser.add_argument('-ts', '--target-size', type=int, nargs=2, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-cd', '--content-dim', type=int, required=True)
	train_parser.add_argument('-id', '--identity-dim', type=int, required=True)
	train_parser.add_argument('-mi', '--max-images', type=int, default=1000000)
	train_parser.add_argument('-vgg', '--vgg-type', type=str, choices=('vgg-face', 'vgg'), required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	convert_parser = action_parsers.add_parser('convert')
	convert_parser.add_argument('-dn', '--data-name', type=str, required=True)
	convert_parser.add_argument('-mn', '--model-name', type=str, required=True)
	convert_parser.add_argument('-ns', '--num-of-samples', type=int, required=True)
	convert_parser.set_defaults(func=convert)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
