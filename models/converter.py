import os
import pickle
import random
import io

from tqdm import tqdm
from PIL import Image

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Dense, UpSampling2D, GlobalAveragePooling2D, BatchNormalization, ReLU, LeakyReLU, Activation
from keras.layers import Layer, Input, Add, Reshape, Flatten
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, Callback

from models.instance_normalization import InstanceNormalization


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

		decoder = load_model(os.path.join(model_dir, 'decoder.h5py'))

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

	def train(self, images, batch_size,
			  n_total_iterations, n_checkpoint_iterations, n_log_iterations,
			  model_dir, tensorboard_dir):

		evaluation_callback = EvaluationCallback(self, images, tensorboard_dir)
		evaluation_callback.set_model(self.converter)

		for i in tqdm(range(n_total_iterations)):
			converter_state = self.__train_converter(images, batch_size)

			if i % n_log_iterations == 0:
				evaluation_callback.on_epoch_end(epoch=i, logs={**converter_state})

			if i % n_checkpoint_iterations == 0:
				self.save(model_dir)

		evaluation_callback.on_train_end(None)

	def __train_converter(self, images, batch_size):
		source_images_batch, target_images_batch = self.__sample_batch(images, batch_size)

		loss = self.converter.train_on_batch(
			x=[source_images_batch, target_images_batch],
			y=source_images_batch
		)

		return dict(
			converter_reconstruction_loss=loss,
			converter_lr=K.get_value(self.converter.optimizer.lr)
		)

	def __sample_batch(self, images, batch_size):
		identity_ids = random.choices(list(images.keys()), k=batch_size)

		source_images_batch = [
			images[identity_id][np.random.randint(0, images[identity_id].shape[0], size=1)]
			for identity_id in identity_ids
		]

		target_images_batch = [
			images[identity_id][np.random.randint(0, images[identity_id].shape[0], size=1)]
			for identity_id in identity_ids
		]

		return np.concatenate(source_images_batch, axis=0), np.concatenate(target_images_batch, axis=0)

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
			optimizer=optimizers.Adam(lr=1e-4),
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
		target_image = Activation('tanh')(x)

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


class EvaluationCallback(TensorBoard):

	def __init__(self, converter, images, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)

		self.converter = converter
		self.images = images

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		identity_id = random.choice(list(self.images.keys()))

		idx = np.random.randint(0, self.images[identity_id].shape[0], size=2)
		imgs = self.images[identity_id][idx]

		source_img = imgs[0]
		target_img = imgs[1]

		summaries = [
			self.dump_source(source_img),
			self.dump_target(target_img),

			self.dump_reconstruction(source_img, target_img)
		]

		for summary in summaries:
			self.writer.add_summary(summary, global_step=epoch)

		self.writer.flush()

	def dump_source(self, img):
		image = self.make_image(img)
		return tf.Summary(value=[tf.Summary.Value(tag='source', image=image)])

	def dump_target(self, img):
		image = self.make_image(img)
		return tf.Summary(value=[tf.Summary.Value(tag='target', image=image)])

	def dump_reconstruction(self, source_img, identity_img):
		reconstructed_img = self.converter.converter.predict([
			np.expand_dims(source_img, axis=0), np.expand_dims(identity_img, axis=0)
		])

		image = self.make_image(reconstructed_img[0])
		return tf.Summary(value=[tf.Summary.Value(tag='reconstructed', image=image)])

	@staticmethod
	def make_image(tensor):
		height, width, channel = tensor.shape
		image = Image.fromarray((((tensor + 1) / 2) * 255).astype(np.uint8))

		with io.BytesIO() as out:
			image.save(out, format='PNG')
			image_string = out.getvalue()

		return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
