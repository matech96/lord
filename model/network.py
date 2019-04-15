import os
import pickle
import random

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras import losses
from keras.layers import Conv2D, Dense, UpSampling2D, ReLU, LeakyReLU, Activation, Lambda
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

		pose_codes = dict()
		for object_id, object_imgs in imgs.items():
			pose_codes[object_id] = np.random.random(size=(object_imgs.shape[0], self.config.content_dim)).astype(np.float32)

		identity_codes = dict()
		for object_id in imgs.keys():
			identity_codes[object_id] = np.random.random(size=(1, self.config.n_adain_layers, self.config.adain_dim, 2)).astype(np.float32)

		pose_code = K.variable(value=np.zeros(shape=(1, self.config.content_dim)), dtype=np.float32)
		identity_code = K.variable(value=np.zeros(shape=(1, self.config.n_adain_layers, self.config.adain_dim, 2)), dtype=np.float32)

		gamma = 1e-3
		target_img = K.placeholder(shape=(1, *self.config.img_shape))
		loss = K.mean(K.abs(self.generator([pose_code, identity_code]) - target_img)) # + gamma * K.sum(K.square(pose_code))

		z_optimizer = optimizers.Adam(lr=1e-4, beta_1=0.5)

		f = K.function(
			inputs=[target_img], outputs=[loss],
			updates=z_optimizer.get_updates(loss, [pose_code, identity_code])
		)

		evaluation_callback = EvaluationCallback(pose_codes, identity_codes, tensorboard_dir)
		evaluation_callback.set_model(self.generator)

		for e in range(n_epochs):
			for i in range(n_iterations_per_epoch):
				object_id = random.choice(list(imgs.keys()))
				idx = np.random.choice(imgs[object_id].shape[0], size=1)

				imgs_batch = imgs[object_id][idx]
				imgs_batch = imgs_batch.astype(np.float64) / 255

				pose_codes_batch = pose_codes[object_id][idx]
				identity_codes_batch = identity_codes[object_id]

				loss_val = self.generator.train_on_batch(x=[pose_codes_batch, identity_codes_batch], y=imgs_batch)
				print('loss: %f' % loss_val)

				K.set_value(pose_code, pose_codes_batch)
				K.set_value(identity_code, identity_codes_batch)

				z_loss_val = f([imgs_batch])[0]
				print('z-loss: %f' % z_loss_val)

				# TODO: gradient clipping?

				# norm = np.sqrt(np.sum(pose_codes_batch ** 2, axis=1))
				# pose_codes_batch = pose_codes_batch / norm[:, np.newaxis]
				#
				# norm = np.sqrt(np.sum(identity_codes_batch ** 2, axis=1))
				# identity_codes_batch = identity_codes_batch / norm[:, np.newaxis]

				pose_codes[object_id][idx] = K.get_value(pose_code)
				identity_codes[object_id] = K.get_value(identity_code)

			evaluation_callback.on_epoch_end(epoch=e, logs={'loss': loss_val})
			# TODO: save model and codes

		evaluation_callback.on_train_end(None)

	@classmethod
	def __build_generator(cls, content_dim, n_adain_layers, adain_dim):
		content_code = Input(shape=(content_dim,))
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		x = Dense(units=6*6*256)(content_code)
		x = ReLU()(x)

		x = Reshape(target_shape=(6, 6, 256))(x)

		for i in range(n_adain_layers):
			x = UpSampling2D(size=(2, 2))(x)
			x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
			x = LeakyReLU()(x)

			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])

		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
		x = LeakyReLU()(x)

		x = Conv2D(filters=1, kernel_size=(7, 7), padding='same')(x)
		x = Activation('sigmoid')(x)

		target_img = Lambda(lambda t: K.squeeze(t, axis=-1))(x)

		model = Model(inputs=[content_code, identity_adain_params], outputs=target_img, name='generator')
		model.compile(
			optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5),
			loss=losses.mean_absolute_error
		)

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
