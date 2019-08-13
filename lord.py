import argparse
import os

import numpy as np

import dataset
from assets import AssetManager
from model.network import Converter
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	imgs, identities, poses = img_dataset.read_images()
	n_identities = np.unique(identities).size

	np.savez(
		file=assets.get_preprocess_file_path(args.data_name),
		imgs=imgs, identities=identities, poses=poses, n_identities=n_identities
	)


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


def split_samples(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, identities, poses = data['imgs'], data['identities'], data['poses']

	n_identities = np.unique(identities).size
	n_samples = imgs.shape[0]

	n_test_samples = int(n_samples * args.test_split)

	test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
	train_idx = ~np.isin(np.arange(n_samples), test_idx)

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

		pose_std=default_config['pose_std'],
		n_adain_layers=default_config['n_adain_layers'],
		adain_dim=default_config['adain_dim'],

		perceptual_loss_layers=default_config['perceptual_loss']['layers'],
		perceptual_loss_weights=default_config['perceptual_loss']['weights']
	)

	converter.train(
		imgs=imgs,
		identities=identities,

		batch_size=default_config['train']['batch_size'],
		n_epochs=default_config['train']['n_epochs'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	converter.save(model_dir)


def train_encoders(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, identities, poses, n_identities = data['imgs'], data['identities'], data['poses'], data['n_identities']
	imgs = imgs.astype(np.float32) / 255.0

	converter = Converter.load(model_dir, include_encoders=False)

	glo_backup_dir = os.path.join(model_dir, args.glo_dir)
	if not os.path.exists(glo_backup_dir):
		os.mkdir(glo_backup_dir)
		converter.save(glo_backup_dir)

	converter.train_pose_encoder(
		imgs=imgs,

		batch_size=default_config['train_encoders']['batch_size'],
		n_epochs=default_config['train_encoders']['n_epochs'],

		model_dir=model_dir
	)

	converter.train_identity_encoder(
		imgs=imgs,
		identities=identities,

		batch_size=default_config['train_encoders']['batch_size'],
		n_epochs=default_config['train_encoders']['n_epochs'],

		model_dir=model_dir
	)

	converter.save(model_dir)


def test(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	prediction_dir = assets.create_prediction_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs']
	imgs = imgs.astype(np.float32) / 255.0

	converter = Converter.load(model_dir, include_encoders=True)
	converter.test(imgs=imgs, prediction_dir=prediction_dir, n_samples=args.num_samples)


def encode(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	prediction_dir = assets.create_prediction_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, identities = data['imgs'], data['identities']
	imgs = imgs.astype(np.float32) / 255.0

	converter = Converter.load(model_dir, include_encoders=True)

	pose_codes = converter.pose_encoder.predict(imgs)
	identity_codes = converter.identity_encoder.predict(imgs)

	np.savez(file=os.path.join(prediction_dir, 'pose.npz'), codes=pose_codes, identities=identities)
	np.savez(file=os.path.join(prediction_dir, 'identity.npz'), codes=identity_codes, identities=identities)


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

	split_samples_parser = action_parsers.add_parser('split-samples')
	split_samples_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_samples_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_samples_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_samples_parser.add_argument('-ts', '--test-split', type=float, required=True)
	split_samples_parser.set_defaults(func=split_samples)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-pd', '--pose-dim', type=int, required=True)
	train_parser.add_argument('-id', '--identity-dim', type=int, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	train_encoders_parser = action_parsers.add_parser('train-encoders')
	train_encoders_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_encoders_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_encoders_parser.add_argument('-gd', '--glo-dir', type=str, default='glo')
	train_encoders_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_encoders_parser.set_defaults(func=train_encoders)

	test_parser = action_parsers.add_parser('test')
	test_parser.add_argument('-dn', '--data-name', type=str, required=True)
	test_parser.add_argument('-mn', '--model-name', type=str, required=True)
	test_parser.add_argument('-ns', '--num-samples', type=int, default=10)
	test_parser.add_argument('-g', '--gpus', type=int, default=1)
	test_parser.set_defaults(func=test)

	encode_parser = action_parsers.add_parser('encode')
	encode_parser.add_argument('-dn', '--data-name', type=str, required=True)
	encode_parser.add_argument('-mn', '--model-name', type=str, required=True)
	encode_parser.add_argument('-g', '--gpus', type=int, default=1)
	encode_parser.set_defaults(func=encode)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
