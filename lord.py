import argparse
import os

import numpy as np
import wandb

import dataset
from assets import AssetManager
from content_classification import LORDContentClassifier
from lighning_plot import ligning_plot
from model.network import Converter
from config import default_config


def preprocess(args):
	assets = AssetManager(args.base_dir)

	img_dataset = dataset.get_dataset(args.dataset_id, args.dataset_path)
	imgs, classes, contents = img_dataset.read_images()
	n_classes = np.unique(classes).size

	np.savez(
		file=assets.get_preprocess_file_path(args.data_name),
		imgs=imgs, classes=classes, contents=contents, n_classes=n_classes
	)


def split_classes(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, contents = data['imgs'], data['classes'], data['contents']

	n_classes = np.unique(classes).size
	test_classes = np.random.choice(n_classes, size=args.num_test_classes, replace=False)

	test_idx = np.isin(classes, test_classes)
	train_idx = ~np.isin(classes, test_classes)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], classes=classes[test_idx], contents=contents[test_idx], n_classes=n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], classes=classes[train_idx], contents=contents[train_idx], n_classes=n_classes
	)


def split_samples(args):
	assets = AssetManager(args.base_dir)

	data = np.load(assets.get_preprocess_file_path(args.input_data_name))
	imgs, classes, contents = data['imgs'], data['classes'], data['contents']

	n_classes = np.unique(classes).size
	n_samples = imgs.shape[0]

	n_test_samples = int(n_samples * args.test_split)

	test_idx = np.random.choice(n_samples, size=n_test_samples, replace=False)
	train_idx = ~np.isin(np.arange(n_samples), test_idx)

	np.savez(
		file=assets.get_preprocess_file_path(args.test_data_name),
		imgs=imgs[test_idx], classes=classes[test_idx], contents=contents[test_idx], n_classes=n_classes
	)

	np.savez(
		file=assets.get_preprocess_file_path(args.train_data_name),
		imgs=imgs[train_idx], classes=classes[train_idx], contents=contents[train_idx], n_classes=n_classes
	)


def train(args):
	wandb.config.update(default_config)
	args_dict = vars(args)
	args_dict.pop('func')
	wandb.config.update(args_dict)
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, classes, contents, n_classes = data['imgs'], data['classes'], data['contents'], data['n_classes']
	imgs = imgs.astype(np.float32) / 255.0

	converter = Converter.build(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		n_classes=n_classes,

		content_dim=args.content_dim,
		class_dim=args.class_dim,

		content_std=args.content_std,
		content_decay=args.content_decay,

		n_adain_layers=default_config['n_adain_layers'],
		adain_enabled=args.adain,
		adain_dim=default_config['adain_dim'],
		adain_normalize=args.adain_normalize,

		perceptual_loss_layers=default_config['perceptual_loss']['layers'],
		perceptual_loss_weights=default_config['perceptual_loss']['weights'],
		perceptual_loss_scales=default_config['perceptual_loss']['scales'],
	)

	converter.train(
		imgs=imgs,
		classes=classes,

		batch_size=default_config['train']['batch_size'],
		n_epochs=default_config['train']['n_epochs'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	converter.save(model_dir)


def train_encoders(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.get_model_dir(args.model_name)
	tensorboard_dir = assets.get_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs, classes, contents, n_classes = data['imgs'], data['classes'], data['contents'], data['n_classes']
	imgs = imgs.astype(np.float32) / 255.0

	converter = Converter.load(model_dir, include_encoders=False)

	glo_backup_dir = os.path.join(model_dir, args.glo_dir)
	if not os.path.exists(glo_backup_dir):
		os.mkdir(glo_backup_dir)
		converter.save(glo_backup_dir)

	converter.train_encoders(
		imgs=imgs,
		classes=classes,

		batch_size=default_config['train_encoders']['batch_size'],
		n_epochs=default_config['train_encoders']['n_epochs'],

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	converter.save(model_dir)


def end2end(args):
	wandb.init(name=args.model_name, project=args.project_name, sync_tensorboard=True)
	train(args)
	train_encoders(args)
	cc_train = LORDContentClassifier(model_name=args.model_name, data_name=args.data_name, base_dir=args.base_dir)
	cc_train.train_content_classifier(1000)
	cc_train.train_class_classifier(1000)
	cc_test = LORDContentClassifier(model_name=args.model_name, data_name=args.test_data_name, base_dir=args.base_dir)
	cc_test.train_content_classifier(1000)
	cc_test.train_class_classifier(1000)
	ligning_plot(args.model_name, args.base_dir, args.adain)


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

	split_classes_parser = action_parsers.add_parser('split-classes')
	split_classes_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_classes_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_classes_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_classes_parser.add_argument('-ntsi', '--num-test-classes', type=int, required=True)
	split_classes_parser.set_defaults(func=split_classes)

	split_samples_parser = action_parsers.add_parser('split-samples')
	split_samples_parser.add_argument('-idn', '--input-data-name', type=str, required=True)
	split_samples_parser.add_argument('-trdn', '--train-data-name', type=str, required=True)
	split_samples_parser.add_argument('-tsdn', '--test-data-name', type=str, required=True)
	split_samples_parser.add_argument('-ts', '--test-split', type=float, required=True)
	split_samples_parser.set_defaults(func=split_samples)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-cd', '--content-dim', type=int, required=True)
	train_parser.add_argument('-yd', '--class-dim', type=int, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	train_encoders_parser = action_parsers.add_parser('train-encoders')
	train_encoders_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_encoders_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_encoders_parser.add_argument('-gd', '--glo-dir', type=str, default='glo')
	train_encoders_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_encoders_parser.set_defaults(func=train_encoders)

	end2end_parser = action_parsers.add_parser('end2end')
	end2end_parser.add_argument('-pn', '--project-name', type=str, required=True)
	end2end_parser.add_argument('-dn', '--data-name', type=str, required=True)
	end2end_parser.add_argument('-tdn', '--test-data-name', type=str, required=True)
	end2end_parser.add_argument('-mn', '--model-name', type=str, required=True)
	end2end_parser.add_argument('-cd', '--content-dim', type=int, required=True)
	end2end_parser.add_argument('-yd', '--class-dim', type=int, required=True)
	end2end_parser.add_argument('-cde', '--content-decay', type=float, default=0.001)
	end2end_parser.add_argument('-cstd', '--content-std', type=float, default=1)
	end2end_parser.add_argument('-gd', '--glo-dir', type=str, default='glo')
	end2end_parser.add_argument('-g', '--gpus', type=int, default=1)
	end2end_parser.add_argument('--adain', dest='adain', action='store_true')
	end2end_parser.add_argument('--no-adain', dest='adain', action='store_false')
	end2end_parser.add_argument('--adain-normalize', dest='adain_normalize', action='store_true')
	end2end_parser.add_argument('--no-adain-normalize', dest='adain_normalize', action='store_false')
	end2end_parser.set_defaults(func=end2end)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
