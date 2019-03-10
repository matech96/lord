import os
import pickle

import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dense, UpSampling2D, GlobalAveragePooling2D, BatchNormalization, ReLU, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Flatten
from keras.models import Model, load_model

from model.generator import DataGenerator
from model.checkpoint import MultiModelCheckpoint
from model.evaluation import EvaluationCallback


class FaceConverter:

	class Config:

		def __init__(self, img_shape):
			self.img_shape = img_shape

	@classmethod
	def build(cls, img_shape, content_dim, identity_dim, n_adain_layers, adain_dim):
		config = FaceConverter.Config(img_shape)

		content_encoder = cls.__build_content_encoder(img_shape, content_dim)
		identity_encoder = cls.__build_identity_encoder(img_shape, identity_dim)
		mlp = cls.__build_mlp(identity_dim, n_adain_layers, adain_dim)

		decoder = cls.__build_decoder(content_dim, n_adain_layers, adain_dim)

		return FaceConverter(config, content_encoder, identity_encoder, mlp, decoder)

	@classmethod
	def load(cls, model_dir):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		content_encoder = load_model(os.path.join(model_dir, 'content_encoder.h5py'))
		identity_encoder = load_model(os.path.join(model_dir, 'identity_encoder.h5py'))
		mlp = load_model(os.path.join(model_dir, 'mlp.h5py'))

		decoder = load_model(os.path.join(model_dir, 'decoder.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		return FaceConverter(config, content_encoder, identity_encoder, mlp, decoder)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.content_encoder.save(os.path.join(model_dir, 'content_encoder.h5py'))
		self.identity_encoder.save(os.path.join(model_dir, 'identity_encoder.h5py'))
		self.mlp.save(os.path.join(model_dir, 'mlp.h5py'))

		self.decoder.save(os.path.join(model_dir, 'decoder.h5py'))

	def __init__(self, config, content_encoder, identity_encoder, mlp, decoder):
		self.config = config

		self.content_encoder = content_encoder
		self.identity_encoder = identity_encoder
		self.mlp = mlp
		self.decoder = decoder

		self.converter = self.__build_converter()

	def train(self, imgs, batch_size,
			  n_epochs, n_iterations_per_epoch, n_epochs_per_checkpoint,
			  model_dir, tensorboard_dir):

		evaluation_callback = EvaluationCallback(self, imgs, tensorboard_dir)
		checkpoint = MultiModelCheckpoint(saver=self, model_dir=model_dir, n_epochs=n_epochs_per_checkpoint)
		lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-6, verbose=1)

		data_generator = DataGenerator(imgs, batch_size, n_iterations_per_epoch)

		self.converter.fit_generator(
			data_generator, epochs=n_epochs,
			callbacks=[evaluation_callback, checkpoint, lr_decay], verbose=1
		)

	def __build_converter(self):
		source_image = Input(shape=self.config.img_shape)
		identity_image = Input(shape=self.config.img_shape)

		content_code = self.content_encoder(source_image)

		identity_code = self.identity_encoder(identity_image)
		identity_adain_params = self.mlp(identity_code)

		model = Model(
			inputs=[source_image, identity_image],
			outputs=self.decoder([content_code, identity_adain_params]),
			name='converter'
		)

		model.compile(
			optimizer=optimizers.Adam(lr=5e-4),
			loss='mean_absolute_error',
		)

		print('converter arch:')
		model.summary()

		return model

	@classmethod
	def __build_content_encoder(cls, img_shape, content_dim):
		image = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(image)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Flatten()(x)

		for i in range(2):
			x = Dense(units=256)(x)
			x = BatchNormalization()(x)
			x = ReLU()(x)

		content_code = Dense(units=content_dim)(x)

		model = Model(inputs=image, outputs=content_code, name='content-encoder')

		print('content-encoder arch:')
		model.summary()

		return model

	@classmethod
	def __build_identity_encoder(cls, img_shape, identity_dim):
		image = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(image)
		x = ReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = ReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = ReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = ReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = ReLU()(x)

		x = GlobalAveragePooling2D()(x)

		identity_code = Dense(units=identity_dim)(x)

		model = Model(inputs=image, outputs=identity_code, name='identity-encoder')

		print('identity-encoder arch:')
		model.summary()

		return model

	@classmethod
	def __build_mlp(cls, identity_dim, n_adain_layers, adain_dim):
		identity_code = Input(shape=(identity_dim,))

		x = Dense(units=adain_dim)(identity_code)
		x = ReLU()(x)

		for i in range(3):
			x = Dense(units=adain_dim)(x)
			x = ReLU()(x)

		adain_params = Dense(units=n_adain_layers * adain_dim * 2)(x)
		adain_params = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(adain_params)

		model = Model(inputs=identity_code, outputs=adain_params, name='mlp')

		print('mlp arch:')
		model.summary()

		return model

	@classmethod
	def __build_decoder(cls, content_dim, n_adain_layers, adain_dim):
		content_code = Input(shape=(content_dim,))
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		x = Dense(units=4*4*512)(content_code)
		x = BatchNormalization()(x)
		x = ReLU()(x)

		x = Reshape(target_shape=(4, 4, 512))(x)

		for i in range(n_adain_layers):
			x = UpSampling2D(size=(2, 2))(x)
			x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
			x = LeakyReLU()(x)

			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])

		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=3, kernel_size=(7, 7), padding='same')(x)
		target_image = Activation('sigmoid')(x)

		model = Model(inputs=[content_code, identity_adain_params], outputs=target_image, name='decoder')

		print('decoder arch:')
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
