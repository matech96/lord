import argparse
import pickle

import numpy as np

import dataset
from assets import AssetManager
from model.network import Converter
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	imgs = img_dataset.read_images()

	with open(assets.get_preprocess_file_path(args.data_name), 'wb') as fd:
		pickle.dump(imgs, fd)


def load_data(path, max_identities, max_images_per_identity):
	with open(path, 'rb') as fd:
		data = pickle.load(fd)

	imgs = []
	img_identities = []

	identities = list(data.keys())[:max_identities]
	for i, identity in enumerate(identities):
		identity_imgs = data[identity][:max_images_per_identity]

		imgs.append(identity_imgs)
		img_identities.append(np.full(shape=(identity_imgs.shape[0], ), fill_value=i))

	imgs = np.concatenate(imgs, axis=0) / 255.0
	img_identities = np.concatenate(img_identities, axis=0)

	return imgs, img_identities


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	imgs, img_identities = load_data(
		path=assets.get_preprocess_file_path(args.data_name),
		max_identities=args.max_identities,
		max_images_per_identity=args.max_images_per_identity
	)

	converter = Converter.build(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_identities=img_identities.max() + 1,

		pose_dim=args.pose_dim,
		identity_dim=args.identity_dim,

		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim']
	)

	converter.train(
		imgs=imgs,
		identities=img_identities,

		batch_size=default_config['batch_size'],
		n_epochs=default_config['n_epochs'],

		n_epochs_per_decay=default_config['n_epochs_per_decay'],
		n_epochs_per_checkpoint=default_config['n_epochs_per_checkpoint'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	converter.save(model_dir)


def test(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	prediction_dir = assets.create_prediction_dir(args.model_name)

	imgs, img_identities = load_data(
		path=assets.get_preprocess_file_path(args.data_name),
		max_identities=args.max_identities,
		max_images_per_identity=args.max_images_per_identity
	)

	converter = Converter.load(model_dir)

	converter.test(
		imgs=imgs,

		batch_size=default_config['batch_size'],
		n_epochs=default_config['n_epochs'],

		n_epochs_per_decay=default_config['n_epochs_per_decay'],
		n_epochs_per_checkpoint=default_config['n_epochs_per_checkpoint'],

		prediction_dir=prediction_dir
	)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-pd', '--pose-dim', type=int, required=True)
	train_parser.add_argument('-id', '--identity-dim', type=int, required=True)
	train_parser.add_argument('-mi', '--max-identities', type=int, required=True)
	train_parser.add_argument('-mipi', '--max-images-per-identity', type=int, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	test_parser = action_parsers.add_parser('test')
	test_parser.add_argument('-dn', '--data-name', type=str, required=True)
	test_parser.add_argument('-mn', '--model-name', type=str, required=True)
	test_parser.add_argument('-mi', '--max-identities', type=int, required=True)
	test_parser.add_argument('-mipi', '--max-images-per-identity', type=int, required=True)
	test_parser.set_defaults(func=test)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
