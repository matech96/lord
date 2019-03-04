import os
import pickle
import random
from functools import partial
import io

from tqdm import tqdm
from PIL import Image

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, Dense, UpSampling2D, GlobalAveragePooling2D, LeakyReLU, ReLU, Activation
from keras.layers import Layer, Input, Add, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, Callback

from instance_normalization import InstanceNormalization


class FaceConverter:

	class Config:

		def __init__(self, img_shape, batch_size):
			self.img_shape = img_shape
			self.batch_size = batch_size

	@classmethod
	def build(cls, img_shape, identity_dim, n_adain_layers, adain_dim, batch_size):
		config = FaceConverter.Config(img_shape, batch_size)

		content_encoder = cls.__build_content_encoder(img_shape)
		identity_encoder = cls.__build_identity_encoder(img_shape, identity_dim)
		mlp = cls.__build_mlp(identity_dim, n_adain_layers, adain_dim)

		decoder = cls.__build_decoder(img_shape, n_adain_layers, adain_dim)
		discriminator = cls.__build_discriminator(img_shape)

		return FaceConverter(config, content_encoder, identity_encoder, mlp, decoder, discriminator)

	@classmethod
	def load(cls, model_dir):
		print('loading models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		content_encoder = load_model(os.path.join(model_dir, 'content_encoder.h5py'))
		identity_encoder = load_model(os.path.join(model_dir, 'identity_encoder.h5py'))
		mlp = load_model(os.path.join(model_dir, 'mlp.h5py'))

		decoder = load_model(os.path.join(model_dir, 'decoder.h5py'))
		discriminator = load_model(os.path.join(model_dir, 'discriminator.h5py'))

		return FaceConverter(config, content_encoder, identity_encoder, mlp, decoder, discriminator)

	def save(self, model_dir):
		print('saving models...')

		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		self.content_encoder.save(os.path.join(model_dir, 'content_encoder.h5py'))
		self.identity_encoder.save(os.path.join(model_dir, 'identity_encoder.h5py'))
		self.mlp.save(os.path.join(model_dir, 'mlp.h5py'))

		self.decoder.save(os.path.join(model_dir, 'decoder.h5py'))
		self.discriminator.save(os.path.join(model_dir, 'discriminator.h5py'))

	def __init__(self, config, content_encoder, identity_encoder, mlp, decoder, discriminator):
		self.config = config

		self.content_encoder = content_encoder
		self.identity_encoder = identity_encoder
		self.mlp = mlp
		self.decoder = decoder
		self.discriminator = discriminator

		self.extended_discriminator = self.__build_extended_discriminator()
		self.converter = self.__build_converter()

	def train(self, images, anchor_images,
			  n_total_iterations, n_checkpoint_iterations, n_log_iterations,
			  model_dir, tensorboard_dir):

		tensorboard = TensorBoard(log_dir=tensorboard_dir)
		tensorboard.set_model(self.converter)

		evaluation_callback = EvaluationCallback(self, images, tensorboard_dir)
		evaluation_callback.set_model(self.converter)

		for i in tqdm(range(n_total_iterations)):
			identity_id = random.choice(list(images.keys()))

			for j in range(5):
				discriminator_state = self.__train_discriminator(images, identity_id, anchor_images)

			for j in range(1):
				converter_state = self.__train_converter(images, identity_id)

			if i % n_log_iterations == 0:
				tensorboard.on_epoch_end(epoch=i, logs={**discriminator_state, **converter_state})
				evaluation_callback.on_epoch_end(epoch=i)

			if i % n_checkpoint_iterations == 0:
				self.save(model_dir)

		tensorboard.on_train_end(None)

	def __train_converter(self, images, identity_id):
		batch_idx = np.random.randint(0, images[identity_id].shape[0], size=self.config.batch_size)
		source_images_batch = images[identity_id][batch_idx]

		batch_idx = np.random.randint(0, images[identity_id].shape[0], size=self.config.batch_size)
		target_images_batch = images[identity_id][batch_idx]

		loss = self.converter.train_on_batch(
			x=[source_images_batch, target_images_batch],
			y=[source_images_batch, -1*np.ones([self.config.batch_size, 1])]
		)

		return dict(
			converter_total_loss=loss[0],
			converter_reconstruction_loss=loss[1],
			converter_discriminator_loss=loss[2],
			converter_lr=K.get_value(self.converter.optimizer.lr)
		)

	def __train_discriminator(self, images, identity_id, anchor_images):
		batch_idx = np.random.randint(0, images[identity_id].shape[0], size=self.config.batch_size)
		source_images_batch = images[identity_id][batch_idx]

		batch_idx = np.random.randint(0, anchor_images.shape[0], size=self.config.batch_size)
		anchor_images_batch = anchor_images[batch_idx]

		y_dummy = np.zeros([self.config.batch_size, 1])  # not used by gradient_penalty loss

		total_loss, _, _, _ = self.extended_discriminator.train_on_batch(
			x=[anchor_images_batch, source_images_batch],
			y=[-1*np.ones([self.config.batch_size, 1]), np.ones([self.config.batch_size, 1]), y_dummy]
		)

		return dict(
			discriminator_loss=total_loss,
			discriminator_lr=K.get_value(self.extended_discriminator.optimizer.lr)
		)

	def __build_converter(self):
		source_image = Input(shape=self.config.img_shape)
		identity_image = Input(shape=self.config.img_shape)

		content_image = self.content_encoder(source_image)
		identity_code = self.identity_encoder(identity_image)
		identity_adain_params = self.mlp(identity_code)

		model = Model(
			inputs=[source_image, identity_image],
			outputs=[
				self.decoder([content_image, identity_adain_params]),
				self.discriminator(content_image)
			],
			name='converter'
		)

		self.__set_trainable(frozens=[self.discriminator])

		w_reconstruction = 1
		w_adversarial = 1

		model.compile(
			optimizer=optimizers.Adam(lr=5e-4, beta_1=0.5, beta_2=0.9),
			loss=[
				'mean_absolute_error',
				wasserstein_loss
			],
			loss_weights=[
				w_reconstruction,
				w_adversarial
			]
		)

		print('converter arch:')
		model.summary()

		return model

	def __build_extended_discriminator(self):
		anchor_image = Input(shape=self.config.img_shape)
		source_image = Input(shape=self.config.img_shape)

		content_image = self.content_encoder(source_image)

		discriminator_output_from_real = self.discriminator(anchor_image)
		discriminator_output_from_fake = self.discriminator(content_image)

		averaged_image = RandomWeightedAverage(batch_size=self.config.batch_size)([anchor_image, content_image])
		discriminator_output_from_averaged = self.discriminator(averaged_image)

		partial_gp_loss = partial(
			gradient_penalty_loss,
			averaged_samples=averaged_image,
			gradient_penalty_weight=10
		)
		partial_gp_loss.__name__ = 'gradient_penalty'

		model = Model(
			inputs=[anchor_image, source_image],
			outputs=[
				discriminator_output_from_real,
				discriminator_output_from_fake,
				discriminator_output_from_averaged
			],
			name='extended-discriminator'
		)

		self.__set_trainable(frozens=[self.content_encoder, self.identity_encoder, self.mlp, self.decoder])

		model.compile(
			optimizer=optimizers.Adam(lr=5e-4, beta_1=0.5, beta_2=0.9),
			loss=[
				wasserstein_loss,
				wasserstein_loss,
				partial_gp_loss
			]
		)

		print('extended discriminator arch:')
		model.summary()

		return model

	@classmethod
	def __build_content_encoder(cls, img_shape):
		source_image = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(source_image)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = cls.__res_block(x, filters=256, kernel_size=(3, 3))
		x = cls.__res_block(x, filters=256, kernel_size=(3, 3))
		x = cls.__res_block(x, filters=256, kernel_size=(3, 3))
		x = cls.__res_block(x, filters=256, kernel_size=(3, 3))

		x = UpSampling2D(size=(2, 2))(x)
		x = Conv2D(filters=128, kernel_size=(4, 4), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = UpSampling2D(size=(2, 2))(x)
		x = Conv2D(filters=64, kernel_size=(4, 4), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same')(x)
		target_image = Activation('tanh')(x)

		model = Model(inputs=source_image, outputs=target_image, name='content-encoder')

		print('content-encoder arch:')
		model.summary()

		return model

	@staticmethod
	def __res_block(input_tensor, filters, kernel_size):
		x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters, kernel_size, padding='same')(x)
		x = InstanceNormalization()(x)
		x = Add()([input_tensor, x])

		return x

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

		x = Dense(units=adain_dim)(x)
		x = ReLU()(x)

		adain_params = Dense(units=n_adain_layers * adain_dim * 2)(x)
		adain_params = Reshape(target_shape=(n_adain_layers, adain_dim, 2))(adain_params)

		model = Model(inputs=identity_code, outputs=adain_params, name='mlp')

		print('mlp arch:')
		model.summary()

		return model

	@classmethod
	def __build_decoder(cls, img_shape, n_adain_layers, adain_dim):
		content_image = Input(shape=img_shape)
		identity_adain_params = Input(shape=(n_adain_layers, adain_dim, 2))

		x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same')(content_image)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		# maybe res blocks here? before adain

		for i in range(n_adain_layers):
			x = Conv2D(filters=adain_dim, kernel_size=(3, 3), padding='same')(x)
			x = AdaptiveInstanceNormalization(adain_layer_idx=i)([x, identity_adain_params])
			x = ReLU()(x) # TODO: change to resblock?

		x = UpSampling2D(size=(2, 2))(x)
		x = Conv2D(filters=128, kernel_size=(5, 5), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = UpSampling2D(size=(2, 2))(x)
		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
		x = InstanceNormalization()(x)
		x = ReLU()(x)

		x = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same')(x)
		target_image = Activation('tanh')(x)

		model = Model(inputs=[content_image, identity_adain_params], outputs=target_image, name='decoder')

		print('decoder arch:')
		model.summary()

		return model

	@staticmethod
	def __build_discriminator(img_shape):
		image = Input(shape=img_shape)

		x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same')(image)
		x = LeakyReLU(alpha=0.2)(x)

		x = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU(alpha=0.2)(x)

		x = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = LeakyReLU(alpha=0.2)(x)

		x = Flatten()(x)

		real_or_fake_score = Dense(1)(x)

		model = Model(inputs=image, outputs=real_or_fake_score, name='discriminator')

		print('discriminator arch:')
		model.summary()

		return model

	def __set_trainable(self, frozens):
		models = [
			self.content_encoder, self.identity_encoder, self.mlp,
			self.decoder, self.discriminator
		]

		for model in models:
			model.trainable = True
			for layer in model.layers:
				layer.trainable = True

		for model in frozens:
			model.trainable = False
			for layer in model.layers:
				layer.trainable = False


def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
	"""Calculates the gradient penalty loss for a batch of "averaged" samples.
	In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
	that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
	this function at all points in the input space. The compromise used in the paper is to choose random points
	on the lines between real and generated samples, and check the gradients at these points. Note that it is the
	gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
	In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
	Then we get the gradients of the discriminator w.r.t. the input averaged samples.
	The l2 norm and penalty can then be calculated for this gradient.
	Note that this loss function requires the original averaged samples as input, but Keras only supports passing
	y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
	averaged_samples argument, and use that for model training."""
	# first get the gradients:
	#   assuming: - that y_pred has dimensions (batch_size, 1)
	#             - averaged_samples has dimensions (batch_size, nbr_features)
	# gradients afterwards has dimension (batch_size, nbr_features), basically
	# a list of nbr_features-dimensional gradient vectors
	gradients = K.gradients(y_pred, averaged_samples)[0]
	# compute the euclidean norm by squaring ...
	gradients_sqr = K.square(gradients)
	#   ... summing over the rows ...
	gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
	#   ... and sqrt
	gradient_l2_norm = K.sqrt(gradients_sqr_sum)
	# compute lambda * (1 - ||grad||)^2 still for each single sample
	gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
	# return the mean as loss over all the batch samples
	return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):

	def __init__(self, batch_size, **kwargs):
		super().__init__(**kwargs)
		self.__batch_size = batch_size

	def _merge_function(self, inputs):
		shape = inputs[0].get_shape()[1:].as_list()
		weights = K.random_uniform([self.__batch_size] + shape)

		return (weights * inputs[0]) + ((1 - weights) * inputs[1])

	def get_config(self):
		config = {
			'batch_size': self.__batch_size
		}

		base_config = super().get_config()
		return dict(list(base_config.items()) + list(config.items()))


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
		return tf.nn.batch_normalization(x, mean, var, adain_offset, adain_scale, variance_epsilon=1e-7)


class EvaluationCallback(Callback):

	def __init__(self, converter, images, tensorboard_dir):
		super().__init__()

		self.converter = converter
		self.images = images

		self.writer = tf.summary.FileWriter(tensorboard_dir)

	def on_epoch_end(self, epoch, logs={}):
		source_id = random.choice(list(self.images.keys()))
		target_id = random.choice(list(self.images.keys()))

		image_idx = np.random.randint(0, self.images[source_id].shape[0])
		source_img = self.images[source_id][image_idx]

		image_idx = np.random.randint(0, self.images[target_id].shape[0])
		target_img = self.images[target_id][image_idx]

		summaries = [
			self.dump_source(source_img),
			self.dump_target(target_img),

			self.dump_intermediate(source_img),
			self.dump_reconstruction(source_img),
			self.dump_conversion(source_img, target_img)
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

	def dump_intermediate(self, img):
		intermediate_img = self.converter.content_encoder.predict(np.expand_dims(img, axis=0))[0]

		image = self.make_image(intermediate_img)
		return tf.Summary(value=[tf.Summary.Value(tag='intermediate', image=image)])

	def dump_reconstruction(self, img):
		reconstructed_img, _ = self.converter.converter.predict([np.expand_dims(img, axis=0), np.expand_dims(img, axis=0)])

		image = self.make_image(reconstructed_img[0])
		return tf.Summary(value=[tf.Summary.Value(tag='reconstructed', image=image)])

	def dump_conversion(self, source_img, identity_img):
		converted_img, _ = self.converter.converter.predict([
			np.expand_dims(source_img, axis=0), np.expand_dims(identity_img, axis=0)
		])

		image = self.make_image(converted_img[0])
		return tf.Summary(value=[tf.Summary.Value(tag='converted', image=image)])

	@staticmethod
	def make_image(tensor):
		height, width, channel = tensor.shape
		image = Image.fromarray((((tensor + 1) / 2) * 255).astype(np.uint8))

		with io.BytesIO() as out:
			image.save(out, format='PNG')
			image_string = out.getvalue()

		return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
