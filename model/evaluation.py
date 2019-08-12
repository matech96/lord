import io

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.callbacks import TensorBoard


class TrainEvaluationCallback(TensorBoard):

	def __init__(self, imgs, identities, pose_embedding, identity_embedding, identity_modulation, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs
		self.__identities = identities

		self.__pose_embedding = pose_embedding
		self.__identity_embedding = identity_embedding
		self.__identity_modulation = identity_modulation
		self.__generator = generator

		self.__n_samples_per_evaluation = 5

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]
		identities = self.__identities[img_ids]

		pose_codes = self.__pose_embedding.predict(img_ids)
		identity_codes = self.__identity_embedding.predict(identities)
		identity_adain_params = self.__identity_modulation.predict(identity_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		for i in range(self.__n_samples_per_evaluation):
			converted_imgs = [imgs[i]] + [
				self.__generator.predict([pose_codes[[j]], identity_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]

			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)

		summary = tf.Summary(value=[tf.Summary.Value(tag='sample', image=make_image(merged_img))])
		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()


class TrainEncodersEvaluationCallback(TensorBoard):

	def __init__(self, imgs, pose_encoder, identity_encoder, identity_modulation, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs

		self.__pose_encoder = pose_encoder
		self.__identity_encoder = identity_encoder
		self.__identity_modulation = identity_modulation
		self.__generator = generator

		self.__n_samples_per_evaluation = 10

	def on_epoch_end(self, epoch, logs={}):
		for k in logs.keys():
			logs['encoders_%s' % k] = logs.pop(k)

		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]

		pose_codes = self.__pose_encoder.predict(imgs)
		identity_codes = self.__identity_encoder.predict(imgs)
		identity_adain_params = self.__identity_modulation.predict(identity_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		for i in range(self.__n_samples_per_evaluation):
			converted_imgs = [imgs[i]] + [
				self.__generator.predict([pose_codes[[j]], identity_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]

			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)

		summary = tf.Summary(value=[tf.Summary.Value(tag='sample-with-encoders', image=make_image(merged_img))])
		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()


def make_image(tensor):
	height, width, channels = tensor.shape
	image = Image.fromarray((np.squeeze(tensor) * 255).astype(np.uint8))

	with io.BytesIO() as out:
		image.save(out, format='PNG')
		image_string = out.getvalue()

	return tf.Summary.Image(height=height, width=width, colorspace=channels, encoded_image_string=image_string)
