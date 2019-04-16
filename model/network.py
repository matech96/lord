import os
import pickle
import random

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Dense, UpSampling2D, LeakyReLU, Activation
from keras.layers import Layer, Input, Reshape
from keras.models import Model, load_model
from keras.applications import vgg16

from model.evaluation import EvaluationCallback


class FaceConverter:

	class Config:

		def __init__(self, img_shape, content_dim, n_adain_layers, adain_dim):
			self.img_shape = img_shape
			self.content_dim = content_dim
			self.n_adain_layers = n_adain_layers
			self.adain_dim = adain_dim

	@classmethod
	def build(cls, img_shape, content_dim, n_adain_layers, adain_dim):
		config = FaceConverter.Config(img_shape, content_dim, n_adain_layers, adain_dim)
		generator = cls.__build_generator(content_dim, n_adain_layers, adain_dim)

		return FaceConverter(config, generator)

	@classmethod
	def load(cls, model_dir):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		generator = load_model(os.path.join(model_dir, 'generator.h5py'), custom_objects={
			'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization
		})

		return FaceConverter(config, generator)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.generator.save(os.path.join(model_dir, 'generator.h5py'))

	def __init__(self, config, generator):
		self.config = config
		self.generator = generator

	def train(self, imgs, batch_size,
			  n_epochs, n_iterations_per_epoch, n_epochs_per_checkpoint,
			  model_dir, tensorboard_dir):

		pose_code = K.variable(
			value=np.zeros(shape=(batch_size, self.config.content_dim)),
			dtype=np.float32
		)

		identity_code = K.variable(
			value=np.zeros(shape=(batch_size, self.config.n_adain_layers, self.config.adain_dim, 2)),
			dtype=np.float32
		)

		target_img = K.placeholder(shape=(batch_size, *self.config.img_shape))
		loss = K.mean(K.abs(self.generator([pose_code, identity_code]) - target_img))  # TODO: regularize pose code

		generator_optimizer = optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.999, decay=1e-4)
		z_optimizer = optimizers.Adam(lr=1e-3, beta_1=0.5, beta_2=0.999, decay=1e-4)

		train_function = K.function(
			inputs=[target_img], outputs=[loss],
			updates=(
				generator_optimizer.get_updates(loss, self.generator.trainable_weights)
				+ z_optimizer.get_updates(loss, [pose_code, identity_code])
			)
		)

		evaluation_callback = EvaluationCallback(tensorboard_dir)
		evaluation_callback.set_model(self.generator)

		pose_codes_batch = np.empty(shape=(batch_size, self.config.content_dim), dtype=np.float32)
		identity_codes_batch = np.empty(shape=(batch_size, self.config.n_adain_layers, self.config.adain_dim, 2), dtype=np.float32)
		imgs_batch = np.empty(shape=(batch_size, *self.config.img_shape), dtype=np.float32)
		img_ids = np.empty(shape=(batch_size, ), dtype=np.uint32)

		pose_codes, identity_codes = self.__init_codes(imgs)

		for e in range(n_epochs):
			for i in range(n_iterations_per_epoch):
				object_ids = random.sample(imgs.keys(), k=batch_size)

				for b in range(batch_size):
					img_ids[b] = np.random.choice(imgs[object_ids[b]].shape[0], size=1)

					pose_codes_batch[b] = pose_codes[object_ids[b]][img_ids[b]]
					identity_codes_batch[b] = identity_codes[object_ids[b]]
					imgs_batch[b] = imgs[object_ids[b]][img_ids[b]][..., np.newaxis] / 255.0

				K.set_value(pose_code, pose_codes_batch)
				K.set_value(identity_code, identity_codes_batch)

				loss_val = train_function([imgs_batch])
				print('loss: %f' % loss_val[0])

				# TODO: gradient clipping?
				# TODO: code normalization?

				pose_codes_batch = K.get_value(pose_code)
				identity_codes_batch = K.get_value(identity_code)

				for b in range(batch_size):
					pose_codes[object_ids[b]][img_ids[b]] = pose_codes_batch[b]
					identity_codes[object_ids[b]] = identity_codes_batch[b]

			evaluation_callback.call(epoch=e, logs={'loss': loss_val[0]}, pose_codes=pose_codes, identity_codes=identity_codes)
			# TODO: save model and codes

		evaluation_callback.on_train_end(None)

	def __init_codes(self, imgs):
		pose_codes = dict()
		for object_id, object_imgs in imgs.items():
			pose_codes[object_id] = np.random.random(size=(object_imgs.shape[0], self.config.content_dim)).astype(np.float32)

		identity_codes = dict()
		for object_id in imgs.keys():
			identity_codes[object_id] = np.random.random(size=(self.config.n_adain_layers, self.config.adain_dim, 2)).astype(np.float32)

		return pose_codes, identity_codes

	@classmethod
	def __build_generator(cls, content_dim, n_adain_layers, adain_dim):
		content_code = Input(shape=(content_dim,))
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		x = Dense(units=6*6*256)(content_code)
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

		model = Model(inputs=[content_code, identity_adain_params], outputs=target_img, name='generator')

		print('decoder arch:')
		model.summary()

		return model

	def __build_vgg(self):
		vgg = vgg16.VGG16(include_top=False, input_shape=(self.config.img_shape[0], self.config.img_shape[1], 3))

		layer_ids = [2, 5, 8, 13, 18]
		layer_outputs = [vgg.layers[layer_id].output for layer_id in layer_ids]

		base_model = Model(inputs=vgg.inputs, outputs=layer_outputs)

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
