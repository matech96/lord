import argparse
import numpy as np

import dataset
from assets import AssetManager
from model.network import Converter
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	imgs, identities, poses = img_dataset.read_images()

	np.savez(assets.get_preprocess_file_path(args.data_name), imgs=imgs, identities=identities, poses=poses)


def split_identities(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, identities, poses = data['imgs'], data['identities'], data['poses']

	n_identities = np.unique(identities).size
	test_identities = np.random.choice(n_identities, size=args.num_test_identities, replace=False)

	test_idx = np.isin(identities, test_identities)
	train_idx = ~np.isin(identities, test_identities)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], identities=identities[test_idx], poses=poses[test_idx], n_identities=n_identities
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], identities=identities[train_idx], poses=poses[train_idx], n_identities=n_identities
	)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, identities, poses, n_identities = data['imgs'], data['identities'], data['poses'], data['n_identities']

	imgs = imgs.astype(np.float32) / 255.0

	converter = Converter.build(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_identities=n_identities,

		pose_dim=args.pose_dim,
		identity_dim=args.identity_dim,

		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim']
	)

	converter.train(
		imgs=imgs,
		identities=identities,

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

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs']

	imgs = imgs.astype(np.float32) / 255.0

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

	split_identities_parser = action_parsers.add_parser('split-identities')
	split_identities_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_identities_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_identities_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_identities_parser.add_argument('-ntsi', '--num-test-identities', type=int, required=True)
	split_identities_parser.set_defaults(func=split_identities)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-pd', '--pose-dim', type=int, required=True)
	train_parser.add_argument('-id', '--identity-dim', type=int, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	test_parser = action_parsers.add_parser('test')
	test_parser.add_argument('-dn', '--data-name', type=str, required=True)
	test_parser.add_argument('-mn', '--model-name', type=str, required=True)
	test_parser.set_defaults(func=test)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
