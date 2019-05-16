import os
import pickle
import imageio

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Dense, UpSampling2D, BatchNormalization, LeakyReLU, Activation, GlobalAveragePooling2D
from keras.layers import Layer, Input, Reshape, Flatten, Concatenate, Embedding
from keras.models import Model, load_model
from keras.applications import vgg16

from model.evaluation import TrainEvaluationCallback, TrainEncodersEvaluationCallback


class Converter:

	class Config:

		def __init__(self, img_shape, n_imgs, n_identities, pose_dim, identity_dim, n_adain_layers, adain_dim):
			self.img_shape = img_shape
			self.n_imgs = n_imgs
			self.n_identities = n_identities
			self.pose_dim = pose_dim
			self.identity_dim = identity_dim
			self.n_adain_layers = n_adain_layers
			self.adain_dim = adain_dim

	@classmethod
	def build(cls, img_shape, n_imgs, n_identities, pose_dim, identity_dim, n_adain_layers, adain_dim):
		config = Converter.Config(img_shape, n_imgs, n_identities, pose_dim, identity_dim, n_adain_layers, adain_dim)

		pose_embedding = cls.__build_pose_embedding(n_imgs, pose_dim)
		identity_embedding = cls.__build_identity_embedding(n_identities, identity_dim)
		identity_modulation = cls.__build_identity_modulation(identity_dim, n_adain_layers, adain_dim)
		generator = cls.__build_generator(pose_dim, n_adain_layers, adain_dim, img_shape)

		return Converter(config, pose_embedding, identity_embedding, identity_modulation, generator)

	@classmethod
	def load(cls, model_dir, include_encoders=False):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		pose_embedding = load_model(os.path.join(model_dir, 'pose_embedding.h5py'), custom_objects={
			'GaussianSampling': GaussianSampling
		})

		identity_embedding = load_model(os.path.join(model_dir, 'identity_embedding.h5py'))
		identity_modulation = load_model(os.path.join(model_dir, 'identity_modulation.h5py'))

		generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		if not include_encoders:
			return Converter(config, pose_embedding, identity_embedding, identity_modulation, generator)

		pose_encoder = load_model(os.path.join(model_dir, 'pose_encoder.h5py'), custom_objects={
			'GaussianSampling': GaussianSampling
		})

		identity_encoder = load_model(os.path.join(model_dir, 'identity_encoder.h5py'))

		return Converter(config, pose_embedding, identity_embedding, identity_modulation, generator, pose_encoder, identity_encoder)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.pose_embedding.save(os.path.join(model_dir, 'pose_embedding.h5py'))
		self.identity_embedding.save(os.path.join(model_dir, 'identity_embedding.h5py'))
		self.identity_modulation.save(os.path.join(model_dir, 'identity_modulation.h5py'))
		self.generator.save(os.path.join(model_dir, 'generator.h5py'))

		if self.pose_encoder:
			self.pose_encoder.save(os.path.join(model_dir, 'pose_encoder.h5py'))

		if self.identity_encoder:
			self.identity_encoder.save(os.path.join(model_dir, 'identity_encoder.h5py'))

	def __init__(self, config,
				 pose_embedding, identity_embedding,
				 identity_modulation, generator,
				 pose_encoder=None, identity_encoder=None):

		self.config = config

		self.pose_embedding = pose_embedding
		self.identity_embedding = identity_embedding
		self.identity_modulation = identity_modulation
		self.generator = generator
		self.pose_encoder = pose_encoder
		self.identity_encoder = identity_encoder

		self.vgg = self.__build_vgg()

	def train(self, imgs, identities,
			  batch_size, n_epochs,
			  n_epochs_per_decay, n_epochs_per_checkpoint,
			  model_dir, tensorboard_dir):

		img_id = K.placeholder(shape=(batch_size, 1))
		identity = K.placeholder(shape=(batch_size, 1))
		target_img = K.placeholder(shape=(batch_size, *self.config.img_shape))

		pose_code = self.pose_embedding(img_id)
		identity_code = self.identity_embedding(identity)
		identity_adain_params = self.identity_modulation(identity_code)
		generated_img = self.generator([pose_code, identity_adain_params])

		target_perceptual_codes = self.vgg(target_img)
		generated_perceptual_codes = self.vgg(generated_img)

		loss = K.mean(K.abs(generated_perceptual_codes - target_perceptual_codes))

		pose_optimizer = optimizers.Adam(lr=1e-3, beta_1=0.5, beta_2=0.999)
		identity_optimizer = optimizers.Adam(lr=1e-3, beta_1=0.5, beta_2=0.999)
		identity_modulation_optimizer = optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
		generator_optimizer = optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)

		train_function = K.function(
			inputs=[img_id, identity, target_img], outputs=[loss],
			updates=(
				pose_optimizer.get_updates(loss, self.pose_embedding.trainable_weights)
				+ identity_optimizer.get_updates(loss, self.identity_embedding.trainable_weights)
				+ identity_modulation_optimizer.get_updates(loss, self.identity_modulation.trainable_weights)
				+ generator_optimizer.get_updates(loss, self.generator.trainable_weights)
			)
		)

		evaluation_callback = TrainEvaluationCallback(imgs, identities,
			self.pose_embedding, self.identity_embedding,
			self.identity_modulation, self.generator,
			tensorboard_dir
		)

		n_samples = imgs.shape[0]
		for e in range(1, n_epochs + 1):
			epoch_idx = np.random.permutation(n_samples)
			for i in np.arange(start=0, stop=(n_samples - batch_size + 1), step=batch_size):
				batch_idx = epoch_idx[i:(i + batch_size)]
				loss_val = train_function([batch_idx, identities[batch_idx], imgs[batch_idx]])

			evaluation_callback.on_epoch_end(epoch=e, logs={
				'loss': loss_val[0],
				'pose_lr': K.get_value(pose_optimizer.lr),
				'identity_lr': K.get_value(identity_optimizer.lr),
				'identity_modulation_lr': K.get_value(identity_modulation_optimizer.lr),
				'generator_lr': K.get_value(generator_optimizer.lr)
			})

			if e % n_epochs_per_decay == 0:
				K.set_value(pose_optimizer.lr, K.get_value(pose_optimizer.lr) * 0.5)
				K.set_value(identity_optimizer.lr, K.get_value(identity_optimizer.lr) * 0.5)
				K.set_value(identity_modulation_optimizer.lr, K.get_value(identity_modulation_optimizer.lr) * 0.5)
				K.set_value(generator_optimizer.lr, K.get_value(generator_optimizer.lr) * 0.5)

			if e % n_epochs_per_checkpoint == 0:
				self.save(model_dir)

		evaluation_callback.on_train_end(None)

	def train_encoders(self, imgs, identities, embeddings_weight,
					   batch_size, n_epochs,
					   n_epochs_per_decay, n_epochs_per_checkpoint,
					   model_dir, tensorboard_dir):

		self.pose_encoder = self.__build_pose_encoder(self.config.img_shape, self.config.pose_dim)
		self.identity_encoder = self.__build_identity_encoder(self.config.img_shape, self.config.identity_dim)

		img_id = K.placeholder(shape=(batch_size, 1))
		identity = K.placeholder(shape=(batch_size, 1))
		target_img = K.placeholder(shape=(batch_size, *self.config.img_shape))

		pose_code = self.pose_encoder(target_img)
		identity_code = self.identity_encoder(target_img)
		identity_adain_params = self.identity_modulation(identity_code)
		generated_img = self.generator([pose_code, identity_adain_params])

		target_perceptual_codes = self.vgg(target_img)
		generated_perceptual_codes = self.vgg(generated_img)

		pose_loss = K.mean(K.abs(pose_code - self.pose_embedding(img_id)))
		identity_loss = K.mean(K.abs(identity_code - self.identity_embedding(identity)))
		reconstruction_loss = K.mean(K.abs(generated_perceptual_codes - target_perceptual_codes))

		loss = reconstruction_loss + embeddings_weight * pose_loss + embeddings_weight * identity_loss

		optimizer = optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999)
		trainable_weights = (
			self.pose_encoder.trainable_weights + self.identity_encoder.trainable_weights
			+ self.identity_modulation.trainable_weights + self.generator.trainable_weights
		)

		train_function = K.function(
			inputs=[img_id, identity, target_img], outputs=[loss, pose_loss, identity_loss, reconstruction_loss],
			updates=(optimizer.get_updates(loss, trainable_weights))
		)

		evaluation_callback = TrainEncodersEvaluationCallback(imgs,
			self.pose_encoder, self.identity_encoder,
			self.identity_modulation, self.generator,
			tensorboard_dir
		)

		n_samples = imgs.shape[0]
		for e in range(1, n_epochs + 1):
			epoch_idx = np.random.permutation(n_samples)
			for i in np.arange(start=0, stop=(n_samples - batch_size + 1), step=batch_size):
				batch_idx = epoch_idx[i:(i + batch_size)]
				loss_val = train_function([batch_idx, identities[batch_idx], imgs[batch_idx]])

			evaluation_callback.on_epoch_end(epoch=e, logs={
				'loss-with-encoders': loss_val[0],
				'pose-loss-with-encoders': loss_val[1],
				'identity-loss-with-encoders': loss_val[2],
				'reconstruction-loss-with-encoders': loss_val[3],
				'lr-with-encoders': K.get_value(optimizer.lr)
			})

			if e % n_epochs_per_decay == 0:
				K.set_value(optimizer.lr, K.get_value(optimizer.lr) * 0.5)

			if e % n_epochs_per_checkpoint == 0:
				self.save(model_dir)

		evaluation_callback.on_train_end(None)

	def test(self, imgs, prediction_dir, n_samples):
		img_idxs = np.random.choice(imgs.shape[0], size=n_samples, replace=False)

		pose_codes = self.pose_encoder.predict(imgs[img_idxs])
		identity_codes = self.identity_encoder.predict(imgs[img_idxs])
		identity_adain_params = self.identity_modulation.predict(identity_codes)

		blank = np.zeros_like(imgs[img_idxs][0])
		summary = [np.concatenate([blank] + list(imgs[img_idxs]), axis=1)]
		for i in range(n_samples):
			converted_imgs = [imgs[img_idxs][i]] + [
				self.generator.predict([pose_codes[[j]], identity_adain_params[[i]]])[0]
				for j in range(n_samples)
			]

			summary.append(np.concatenate(converted_imgs, axis=1))

		summary_img = np.concatenate(summary, axis=0)
		imageio.imwrite(os.path.join(prediction_dir, 'summary.png'), (summary_img * 255).astype(np.uint8))

	@classmethod
	def __build_pose_embedding(cls, n_imgs, pose_dim):
		img_id = Input(shape=(1, ))

		pose_embedding_mean = Embedding(input_dim=n_imgs, output_dim=pose_dim)(img_id)
		pose_embedding_mean = Reshape(target_shape=(pose_dim, ))(pose_embedding_mean)

		pose_embedding_log_var = Embedding(input_dim=n_imgs, output_dim=pose_dim)(img_id)
		pose_embedding_log_var = Reshape(target_shape=(pose_dim, ))(pose_embedding_log_var)

		pose_embedding = GaussianSampling()([pose_embedding_mean, pose_embedding_log_var])

		model = Model(inputs=img_id, outputs=pose_embedding, name='pose-embedding')

		print('pose embedding:')
		model.summary()

		return model

	@classmethod
	def __build_identity_embedding(cls, n_identities, identity_dim):
		identity = Input(shape=(1, ))

		identity_embedding = Embedding(input_dim=n_identities, output_dim=identity_dim)(identity)
		identity_embedding = Reshape(target_shape=(identity_dim, ))(identity_embedding)

		model = Model(inputs=identity, outputs=identity_embedding, name='identity-embedding')

		print('identity embedding:')
		model.summary()

		return model

	@classmethod
	def __build_identity_modulation(cls, identity_dim, n_adain_layers, adain_dim):
		identity_code = Input(shape=(identity_dim, ))

		adain_per_layer = [Dense(units=adain_dim * 2)(identity_code) for _ in range(n_adain_layers)]
		adain_all = Concatenate(axis=-1)(adain_per_layer)
		identity_adain_params = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(adain_all)

		model = Model(inputs=[identity_code], outputs=identity_adain_params, name='identity-modulation')

		print('identity-modulation arch:')
		model.summary()

		return model

	@classmethod
	def __build_generator(cls, pose_dim, n_adain_layers, adain_dim, img_shape):
		pose_code = Input(shape=(pose_dim, ))
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		initial_height = img_shape[0] // (2 ** n_adain_layers)
		initial_width = img_shape[1] // (2 ** n_adain_layers)

		x = Dense(units=initial_height * initial_width * (adain_dim // 8))(pose_code)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Dense(units=initial_height * initial_width * (adain_dim // 4))(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Dense(units=initial_height * initial_width * adain_dim)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Reshape(target_shape=(initial_height, initial_width, adain_dim))(x)

		for i in range(n_adain_layers):
			x = UpSampling2D(size=(2, 2))(x)
			x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
			x = LeakyReLU()(x)

			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])

		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=img_shape[-1], kernel_size=(7, 7), padding='same')(x)
		target_img = Activation('sigmoid')(x)

		model = Model(inputs=[pose_code, identity_adain_params], outputs=target_img, name='generator')

		print('generator arch:')
		model.summary()

		return model

	def __build_vgg(self):
		vgg = vgg16.VGG16(include_top=False, input_shape=(self.config.img_shape[0], self.config.img_shape[1], 3))

		layer_ids = [2, 5, 8, 13, 18]
		layer_outputs = [Flatten()(vgg.layers[layer_id].output) for layer_id in layer_ids]

		base_model = Model(inputs=vgg.inputs, outputs=Concatenate(axis=-1)(layer_outputs))

		img = Input(shape=self.config.img_shape)
		model = Model(inputs=img, outputs=base_model(VggNormalization()(img)), name='vgg')

		print('vgg arch:')
		model.summary()

		return model

	@classmethod
	def __build_pose_encoder(cls, img_shape, pose_dim):
		img = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(img)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = BatchNormalization()(x)
			x = LeakyReLU()(x)

		pose_code_mean = Dense(units=pose_dim)(x)
		pose_code_log_var = Dense(units=pose_dim)(x)

		pose_code = GaussianSampling()([pose_code_mean, pose_code_log_var])

		model = Model(inputs=img, outputs=pose_code, name='pose-encoder')

		print('pose-encoder arch:')
		model.summary()

		return model

	@classmethod
	def __build_identity_encoder(cls, img_shape, identity_dim):
		img = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(img)
		x = LeakyReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU()(x)

		x = GlobalAveragePooling2D()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = LeakyReLU()(x)

		identity_code = Dense(units=identity_dim)(x)

		model = Model(inputs=img, outputs=identity_code, name='identity-encoder')

		print('identity-encoder arch:')
		model.summary()

		return model


class AdaptiveInstanceNormalization(Layer):

	def __init__(self, adain_layer_idx, **kwargs):
		super().__init__(**kwargs)
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


class VggNormalization(Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs, **kwargs):
		x = inputs * 255

		if x.shape[-1] == 1:
			x = tf.tile(x, (1, 1, 1, 3))

		return vgg16.preprocess_input(x)


class GaussianSampling(Layer):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs, **kwargs):
		z_mean, z_log_var = inputs

		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]

		z_standard = K.random_normal(shape=(batch, dim))
		return z_mean + K.exp(0.5 * z_log_var) * z_standard
