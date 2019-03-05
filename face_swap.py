import argparse
import pickle

import data
from dataset import CelebA

from assets import AssetManager
from models.converter import FaceConverter


def preprocess(args):
	assets = AssetManager(args.base_dir)

	dataset = CelebA(args.dataset_path)
	identity_map = dataset.get_identity_map()

	imgs = dict()
	for i, identity in enumerate(identity_map.keys()):
		print('preprocessing identity: #%d' % i)
		imgs[identity] = data.preprocess_images(identity_map[identity], crop_size=(128, 128), target_size=(64, 64))

	with open(assets.get_preprocess_file_path(args.data_name), 'wb') as fd:
		pickle.dump(imgs, fd)


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	with open(assets.get_preprocess_file_path(args.data_name), 'rb') as fd:
		train_images = pickle.load(fd)

		for k in train_images.keys():
			train_images[k] = (train_images[k] / 255) * 2 - 1

	face_converter = FaceConverter.build(
		img_shape=(64, 64, 3),

		content_dim=32,
		identity_dim=512,

		n_adain_layers=4,
		adain_dim=256
	)

	face_converter.train(
		images=train_images,
		batch_size=64,

		n_total_iterations=1000000,
		n_checkpoint_iterations=1000,
		n_log_iterations=100,

		model_dir=model_dir,
		tensorboard_dir=tensorboard_dir
	)

	face_converter.save(model_dir)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	args = parser.parse_args()
	args.func(args)


if __name__ == '__main__':
	main()
