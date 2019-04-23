import os
import pickle

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Dense, UpSampling2D, BatchNormalization, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Flatten, Concatenate, Embedding
from keras.models import Model, load_model
from keras.applications import vgg16

from model.evaluation import EvaluationCallback


class Converter:

	class Config:

		def __init__(self, img_shape, n_imgs, n_identities, pose_dim, n_adain_layers, adain_dim):
			self.img_shape = img_shape
			self.n_imgs = n_imgs
			self.n_identities = n_identities
			self.pose_dim = pose_dim
			self.n_adain_layers = n_adain_layers
			self.adain_dim = adain_dim

	@classmethod
	def build(cls, img_shape, n_imgs, n_identities, pose_dim, n_adain_layers, adain_dim):
		config = Converter.Config(img_shape, n_imgs, n_identities, pose_dim, n_adain_layers, adain_dim)

		pose_embedding = cls.__build_pose_embedding(n_imgs, pose_dim)
		identity_embedding = cls.__build_identity_embedding(n_identities, n_adain_layers, adain_dim)
		generator = cls.__build_generator(pose_dim, n_adain_layers, adain_dim)

		return Converter(config, pose_embedding, identity_embedding, generator)

	@classmethod
	def load(cls, model_dir):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		pose_embedding = load_model(os.path.join(model_dir, 'pose_embedding.h5py'))
		identity_embedding = load_model(os.path.join(model_dir, 'identity_embedding.h5py'))
		generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		return Converter(config, pose_embedding, identity_embedding, generator)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.pose_embedding.save(os.path.join(model_dir, 'pose_embedding.h5py'))
		self.identity_embedding.save(os.path.join(model_dir, 'identity_embedding.h5py'))
		self.generator.save(os.path.join(model_dir, 'generator.h5py'))

	def __init__(self, config, pose_embedding, identity_embedding, generator):
		self.config = config

		self.pose_embedding = pose_embedding
		self.identity_embedding = identity_embedding
		self.generator = generator

		self.vgg = self.__build_vgg()

	def train(self, imgs, identities,
			  batch_size, n_epochs, n_epochs_per_checkpoint,
			  model_dir, tensorboard_dir):

		img_id = K.placeholder(shape=(batch_size, 1))
		identity = K.placeholder(shape=(batch_size, 1))
		target_img = K.placeholder(shape=(batch_size, *self.config.img_shape))

		pose_code = self.pose_embedding(img_id)
		identity_code = self.identity_embedding(identity)
		generated_img = self.generator([pose_code, identity_code])

		target_perceptual_codes = self.vgg(target_img)
		generated_perceptual_codes = self.vgg(generated_img)

		loss = K.mean(K.abs(generated_perceptual_codes - target_perceptual_codes))  # + gamma * K.mean(K.abs(pose_code))

		pose_optimizer = optimizers.Adam(lr=1e-3, beta_1=0.5, beta_2=0.999)
		identity_optimizer = optimizers.Adam(lr=5e-4, beta_1=0.5, beta_2=0.999)
		generator_optimizer = optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

		train_function = K.function(
			inputs=[img_id, identity, target_img], outputs=[loss],
			updates=(
				pose_optimizer.get_updates(loss, self.pose_embedding.trainable_weights)
				+ identity_optimizer.get_updates(loss, self.identity_embedding.trainable_weights)
				+ generator_optimizer.get_updates(loss, self.generator.trainable_weights)
			)
		)

		evaluation_callback = EvaluationCallback(
			imgs, identities, self.pose_embedding, self.identity_embedding, self.generator, tensorboard_dir
		)

		n_samples = imgs.shape[0]
		for e in range(1, n_epochs + 1):
			epoch_idx = np.random.permutation(n_samples)
			for i in np.arange(start=0, stop=(n_samples - batch_size + 1), step=batch_size):
				batch_idx = epoch_idx[i:(i + batch_size)]
				loss_val = train_function([batch_idx, identities[batch_idx], imgs[batch_idx]])

			if e % 100 == 0:
				K.set_value(pose_optimizer.lr, K.get_value(pose_optimizer.lr) * 0.5)
				K.set_value(identity_optimizer.lr, K.get_value(identity_optimizer.lr) * 0.5)
				K.set_value(generator_optimizer.lr, K.get_value(generator_optimizer.lr) * 0.5)

			evaluation_callback.on_epoch_end(epoch=e, logs={
				'loss': loss_val[0],
				'pose_lr': K.get_value(pose_optimizer.lr),
				'identity_lr': K.get_value(identity_optimizer.lr),
				'generator_lr': K.get_value(generator_optimizer.lr)
			})

			if e % n_epochs_per_checkpoint == 0:
				self.save(model_dir)

		evaluation_callback.on_train_end(None)

	# def normalize_pose_embeddings(self):
	# 	pose_embeddings = self.pose_embedding.get_weights()[0]
	#
	# 	norm = np.sqrt(np.sum(pose_embeddings ** 2, axis=1, keepdims=True))
	# 	pose_embeddings = pose_embeddings / norm
	#
	# 	self.pose_embedding.set_weights([pose_embeddings])

	@classmethod
	def __build_generator(cls, pose_dim, n_adain_layers, adain_dim):
		pose_code = Input(shape=(pose_dim,))
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		x = Dense(units=128)(pose_code)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Dense(units=256)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Dense(units=6*6*256)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Reshape(target_shape=(6, 6, 256))(x)

		for i in range(n_adain_layers):
			x = UpSampling2D(size=(2, 2))(x)
			x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
			x = LeakyReLU()(x)

			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])

		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=1, kernel_size=(7, 7), padding='same')(x)
		target_img = Activation('sigmoid')(x)

		model = Model(inputs=[pose_code, identity_adain_params], outputs=target_img, name='generator')

		print('decoder arch:')
		model.summary()

		return model

	@classmethod
	def __build_pose_embedding(cls, n_imgs, pose_dim):
		img_id = Input(shape=(1, ))
		pose_embedding = Embedding(input_dim=n_imgs, output_dim=pose_dim)(img_id)
		pose_embedding = Reshape(target_shape=(pose_dim, ))(pose_embedding)
		pose_embedding = Activation('softmax')(pose_embedding)

		model = Model(inputs=img_id, outputs=pose_embedding, name='pose-embedding')

		print('pose embedding:')
		model.summary()

		return model

	@classmethod
	def __build_identity_embedding(cls, n_identities, n_adain_layers, adain_dim):
		identity = Input(shape=(1, ))
		identity_embedding = Embedding(input_dim=n_identities, output_dim=(n_adain_layers * adain_dim * 2))(identity)
		identity_embedding = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(identity_embedding)

		model = Model(inputs=identity, outputs=identity_embedding, name='identity-embedding')

		print('identity embedding:')
		model.summary()

		return model

	def __build_vgg(self):
		vgg = vgg16.VGG16(include_top=False, input_shape=(self.config.img_shape[0], self.config.img_shape[1], 3))

		layer_ids = [2, 5, 8, 13, 18]
		layer_outputs = [Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]

		base_model = Model(inputs=vgg.inputs, outputs=Concatenate(axis=-1)(layer_outputs))

		img = Input(shape=self.config.img_shape)
		model = Model(inputs=img, outputs=base_model(NormalizeForVGG()(img)), name='vgg')

		print('vgg arch:')
		model.summary()

		return model


class AdaptiveInstanceNormalization(Layer):

	def __init__(self, adain_layer_idx, **kwargs):
		super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
		self.adain_layer_idx = adain_layer_idx

	def call(self, inputs, **kwargs):
		assert isinstance(inputs, list)

		x, adain_params = inputs
		adain_offset = adain_params[:, self.adain_layer_idx, :, 0]
		adain_scale = adain_params[:, self.adain_layer_idx, :, 1]

		adain_dim = x.shape[-1]
		adain_offset = K.reshape(adain_offset, (-1, 1, 1, adain_dim))
		adain_scale = K.reshape(adain_scale, (-1, 1, 1, adain_dim))

		mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
		x_standard = (x - mean) / (tf.sqrt(var) + 1e-7)

		return (x_standard * adain_scale) + adain_offset

	def get_config(self):
		config = {
			'adain_layer_idx': self.adain_layer_idx
		}

		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))


class NormalizeForVGG(Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs, **kwargs):
		x = inputs * 255

		x = tf.tile(x, (1, 1, 1, 3))

		return vgg16.preprocess_input(x)
