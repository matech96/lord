import os
import pickle

import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras import losses
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, Dense, UpSampling2D, GlobalAveragePooling2D, BatchNormalization, ReLU, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape, Flatten
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.applications import vgg16
from keras_vggface import vggface

from model.generator import DataGenerator
from model.checkpoint import MultiModelCheckpoint
from model.evaluation import EvaluationCallback


class FaceConverter:

	class Config:

		def __init__(self, img_shape, use_vgg_face):
			self.img_shape = img_shape
			self.use_vgg_face = use_vgg_face

	@classmethod
	def build(cls, img_shape, content_dim, identity_dim, n_adain_layers, adain_dim, use_vgg_face):
		config = FaceConverter.Config(img_shape, use_vgg_face)

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
		self.vgg = self.__build_vgg()

	def train(self, imgs, batch_size, n_gpus,
			  n_epochs, n_iterations_per_epoch, n_epochs_per_checkpoint,
			  model_dir, tensorboard_dir):

		evaluation_callback = EvaluationCallback(self, imgs, tensorboard_dir)
		checkpoint = MultiModelCheckpoint(saver=self, model_dir=model_dir, n_epochs=n_epochs_per_checkpoint)
		lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-6, verbose=1)

		data_generator = DataGenerator(self.vgg, imgs, batch_size, n_iterations_per_epoch)

		model = self.__build_perceptual(n_gpus)
		model.fit_generator(
			data_generator, epochs=n_epochs,
			callbacks=[evaluation_callback, checkpoint, lr_decay], verbose=1
		)

	def __build_converter(self):
		source_img = Input(shape=self.config.img_shape)
		identity_img = Input(shape=self.config.img_shape)

		content_code = self.content_encoder(source_img)

		identity_code = self.identity_encoder(identity_img)
		identity_adain_params = self.mlp(identity_code)

		model = Model(
			inputs=[source_img, identity_img],
			outputs=self.decoder([content_code, identity_adain_params]),
			name='converter'
		)

		print('converter arch:')
		model.summary()

		return model

	def __build_vgg(self):
		if self.config.use_vgg_face:
			vgg = vggface.VGGFace(model='vgg16', include_top=False, input_shape=(64, 64, 3))
		else:
			vgg = vgg16.VGG16(include_top=False, input_shape=(64, 64, 3))

		layer_ids = [1, 2, 5, 8, 13, 18]
		layer_outputs = [vgg.layers[layer_id].output for layer_id in layer_ids]

		base_model = Model(inputs=vgg.inputs, outputs=layer_outputs)

		img = Input(shape=self.config.img_shape)
		model = Model(inputs=img, outputs=base_model(NormalizeForVGG(self.config.use_vgg_face)(img)), name='vgg')

		print('vgg arch:')
		model.summary()

		model._make_predict_function()
		return model

	def __build_perceptual(self, n_gpus):
		source_img = Input(shape=self.config.img_shape)
		identity_img = Input(shape=self.config.img_shape)

		converted_img = self.converter([source_img, identity_img])
		perceptual_codes = self.vgg(converted_img)

		self.vgg.trainable = False

		model = Model(inputs=[source_img, identity_img], outputs=perceptual_codes, name='perceptual')

		if n_gpus > 1:
			model = multi_gpu_model(model, n_gpus)

		model.compile(
			optimizer=optimizers.Adam(lr=1e-4),
			loss=[losses.mean_absolute_error] * 6,
			loss_weights=[1] * 6
		)

		print('perceptual arch:')
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


class NormalizeForVGG(Layer):

	def __init__(self, vgg_face, **kwargs):
		super().__init__(**kwargs)

		self.vgg_face = vgg_face

	def call(self, inputs, **kwargs):
		x = inputs * 255

		if self.vgg_face:
			r, g, b = tf.split(axis=3, num_or_size_splits=3, value=x)

			return tf.concat(axis=3, values=[
				b - 93.5940,
				g - 104.7624,
				r - 129.1863,
			])

		else:
			return vgg16.preprocess_input(x)

	def get_config(self):
		config = {
			'vgg_face': self.vgg_face
		}

		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))
