import os
import pickle
import imageio

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers, losses
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation, GlobalAveragePooling2D
from keras.layers import Layer, Input, Reshape, Lambda, Flatten, Concatenate, Embedding, GaussianNoise
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from keras.applications import vgg16
from keras_lr_multiplier import LRMultiplier

from model.evaluation import TrainEvaluationCallback, TrainEncodersEvaluationCallback


class Converter:

	class Config:

		def __init__(self, img_shape, n_imgs, n_identities,
					 pose_dim, identity_dim, pose_std, n_adain_layers, adain_dim,
					 perceptual_loss_layers, perceptual_loss_weights):

			self.img_shape = img_shape

			self.n_imgs = n_imgs
			self.n_identities = n_identities

			self.pose_dim = pose_dim
			self.identity_dim = identity_dim

			self.pose_std = pose_std
			self.n_adain_layers = n_adain_layers
			self.adain_dim = adain_dim

			self.perceptual_loss_layers = perceptual_loss_layers
			self.perceptual_loss_weights = perceptual_loss_weights

	@classmethod
	def build(cls, img_shape, n_imgs, n_identities,
			  pose_dim, identity_dim, pose_std, n_adain_layers, adain_dim,
			  perceptual_loss_layers, perceptual_loss_weights):

		config = Converter.Config(
			img_shape, n_imgs, n_identities,
			pose_dim, identity_dim, pose_std, n_adain_layers, adain_dim,
			perceptual_loss_layers, perceptual_loss_weights
		)

		pose_embedding = cls.__build_pose_embedding(n_imgs, pose_dim, pose_std)
		identity_embedding = cls.__build_identity_embedding(n_identities, identity_dim)
		identity_modulation = cls.__build_identity_modulation(identity_dim, n_adain_layers, adain_dim)
		generator = cls.__build_generator(pose_dim, n_adain_layers, adain_dim, img_shape)

		return Converter(config, pose_embedding, identity_embedding, identity_modulation, generator)

	@classmethod
	def load(cls, model_dir, include_encoders=False):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		pose_embedding = load_model(os.path.join(model_dir, 'pose_embedding.h5py'))
		identity_embedding = load_model(os.path.join(model_dir, 'identity_embedding.h5py'))
		identity_modulation = load_model(os.path.join(model_dir, 'identity_modulation.h5py'))

		generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		if not include_encoders:
			return Converter(config, pose_embedding, identity_embedding, identity_modulation, generator)

		pose_encoder = load_model(os.path.join(model_dir, 'pose_encoder.h5py'))
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

	def train(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
		img_id = Input(shape=(1, ))
		identity = Input(shape=(1, ))

		pose_code = self.pose_embedding(img_id)
		identity_code = self.identity_embedding(identity)
		identity_adain_params = self.identity_modulation(identity_code)
		generated_img = self.generator([pose_code, identity_adain_params])

		model = Model(inputs=[img_id, identity], outputs=generated_img)

		model.compile(
			optimizer=LRMultiplier(
				optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999),
				multipliers={
					'pose-embedding': 10,
					'identity-embedding': 10
				}
			),

			loss=self.__perceptual_loss
		)

		reduce_lr = ReduceLROnPlateau(monitor='loss', mode='min', min_delta=1, factor=0.5, patience=5, verbose=1)
		early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=1, patience=10, verbose=1)

		tensorboard = TrainEvaluationCallback(
			imgs, identities,
			self.pose_embedding, self.identity_embedding,
			self.identity_modulation, self.generator,
			tensorboard_dir
		)

		checkpoint = CustomModelCheckpoint(self, model_dir)

		model.fit(
			x=[np.arange(imgs.shape[0]), identities], y=imgs,
			batch_size=batch_size, epochs=n_epochs,
			callbacks=[reduce_lr, early_stopping, checkpoint, tensorboard],
			verbose=1
		)

	def train_encoders(self, imgs, identities, batch_size, n_epochs, model_dir, tensorboard_dir):
		self.pose_encoder = self.__build_pose_encoder(self.config.img_shape, self.config.pose_dim)
		self.identity_encoder = self.__build_identity_encoder(self.config.img_shape, self.config.identity_dim)

		img = Input(shape=self.config.img_shape)

		pose_code = self.pose_encoder(img)
		identity_code = self.identity_encoder(img)

		model = Model(inputs=img, outputs=[pose_code, identity_code])
		model.compile(
			optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999),
			loss=[losses.mean_squared_error, losses.mean_squared_error]
		)

		tensorboard = TrainEncodersEvaluationCallback(imgs,
			self.pose_encoder, self.identity_encoder,
			self.identity_modulation, self.generator,
			tensorboard_dir
		)

		reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', min_delta=0.01, factor=0.5, patience=5, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=10, verbose=1)

		checkpoint = CustomModelCheckpoint(self, model_dir)

		model.fit(
			x=imgs, y=[self.pose_embedding.predict(np.arange(imgs.shape[0])), self.identity_embedding.predict(identities)],
			validation_split=0.1,
			batch_size=batch_size, epochs=n_epochs,
			callbacks=[reduce_lr, early_stopping, checkpoint, tensorboard],
			verbose=1
		)

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

	def __perceptual_loss(self, y_true, y_pred):
		perceptual_codes_pred = self.vgg(y_pred)
		perceptual_codes_true = self.vgg(y_true)

		normalized_weights = self.config.perceptual_loss_weights / np.sum(self.config.perceptual_loss_weights)
		loss = 0

		for i, (p, t) in enumerate(zip(perceptual_codes_pred, perceptual_codes_true)):
			loss += normalized_weights[i] * K.mean(K.abs(p - t))

		return loss

	@classmethod
	def __build_pose_embedding(cls, n_imgs, pose_dim, pose_std):
		img_id = Input(shape=(1, ))

		pose_embedding = Embedding(input_dim=n_imgs, output_dim=pose_dim, name='pose-embedding')(img_id)
		pose_embedding = Reshape(target_shape=(pose_dim, ))(pose_embedding)

		pose_embedding = GaussianNoise(stddev=pose_std)(pose_embedding)

		model = Model(inputs=img_id, outputs=pose_embedding)

		print('pose embedding:')
		model.summary()

		return model

	@classmethod
	def __build_identity_embedding(cls, n_identities, identity_dim):
		identity = Input(shape=(1, ))

		identity_embedding = Embedding(input_dim=n_identities, output_dim=identity_dim, name='identity-embedding')(identity)
		identity_embedding = Reshape(target_shape=(identity_dim, ))(identity_embedding)

		model = Model(inputs=identity, outputs=identity_embedding)

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
		x = LeakyReLU()(x)

		x = Dense(units=initial_height * initial_width * (adain_dim // 4))(x)
		x = LeakyReLU()(x)

		x = Dense(units=initial_height * initial_width * adain_dim)(x)
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

		layer_outputs = [vgg.layers[layer_id].output for layer_id in self.config.perceptual_loss_layers]
		feature_extractor = Model(inputs=vgg.inputs, outputs=layer_outputs)

		img = Input(shape=self.config.img_shape)

		if self.config.img_shape[-1] == 1:
			x = Lambda(lambda t: tf.tile(t, multiples=(1, 1, 1, 3)))(img)
		else:
			x = img

		x = VggNormalization()(x)
		features = feature_extractor(x)

		model = Model(inputs=img, outputs=features, name='vgg')

		print('vgg arch:')
		model.summary()

		return model

	@classmethod
	def __build_pose_encoder(cls, img_shape, pose_dim):
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

		x = Flatten()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = LeakyReLU()(x)

		pose_code = Dense(units=pose_dim)(x)

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

		x = Flatten()(x)

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
		return vgg16.preprocess_input(x)


class CustomModelCheckpoint(Callback):

	def __init__(self, model, path):
		super().__init__()
		self.__model = model
		self.__path = path

	def on_epoch_end(self, epoch, logs=None):
		self.__model.save(self.__path)
