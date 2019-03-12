import argparse
import random
import os

import numpy as np
import imageio

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
	preprocessed_dir = assets.get_preprocess_dir(args.data_name)

	identity_ids = list(identity_map.keys())[args.head_identity_index:args.tail_identity_index]

	for i, identity in enumerate(identity_ids):
		print('processing identity: %s' % identity)

		identity_imgs = face_preprocessor.preprocess_imgs(identity_map[identity])
		np.savez(os.path.join(preprocessed_dir, identity + '.npz'), imgs=identity_imgs)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	preprocessed_dir = assets.get_preprocess_dir(args.data_name)

	imgs = dict()
	for identity_file_name in os.listdir(preprocessed_dir)[:args.max_identities]:
		identity_id = os.path.splitext(identity_file_name)[0]
		identity_imgs = np.load(os.path.join(preprocessed_dir, identity_file_name))['imgs'][:args.max_files_per_identity]
		imgs[identity_id] = identity_imgs.astype(np.float64) / 255

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
		batch_size=default_config['batch_size'],

		n_epochs=default_config['n_epochs'],
		n_iterations_per_epoch=default_config['n_iterations_per_epoch'],
		n_epochs_per_checkpoint=default_config['n_epochs_per_checkpoint'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	face_converter.save(model_dir)

#
# def convert(args):
# 	assets = AssetManager(args.base_dir)
# 	model_dir = assets.get_model_dir(args.model_name)
# 	prediction_dir = assets.create_prediction_dir(args.model_name)
#
# 	with open(assets.get_preprocess_file_path(args.data_name), 'rb') as fd:
# 		train_images = pickle.load(fd)
#
# 		for k in train_images.keys():
# 			train_images[k] = (train_images[k] / 255) * 2 - 1
#
# 	face_converter = FaceConverter.load(model_dir)
#
# 	for i in range(args.num_of_samples):
# 		source_identity_id = random.choice(list(train_images.keys()))
# 		target_identity_id = random.choice(list(train_images.keys()))
#
# 		source_img = train_images[source_identity_id][0]
# 		target_img = train_images[target_identity_id][0]
#
# 		converted_img = face_converter.converter.predict([
# 			np.expand_dims(source_img, axis=0), np.expand_dims(target_img, axis=0)
# 		])[0]
#
# 		merged_img = np.concatenate((source_img, target_img, converted_img), axis=1)
# 		imageio.imwrite(os.path.join(prediction_dir, '%d.png' % i), merged_img)


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
	train_parser.add_argument('-mf', '--max-files-per-identity', type=int, default=50)
	train_parser.add_argument('-mi', '--max-identities', type=int, default=10000)
	train_parser.add_argument('-vgg', '--vgg-type', type=str, choices=('vgg-face', 'vgg'), required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	# convert_parser = action_parsers.add_parser('convert')
	# convert_parser.add_argument('-dn', '--data-name', type=str, required=True)
	# convert_parser.add_argument('-mn', '--model-name', type=str, required=True)
	# convert_parser.add_argument('-ns', '--num-of-samples', type=int, required=True)
	# convert_parser.set_defaults(func=convert)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
