import io

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.callbacks import TensorBoard


class EvaluationCallback(TensorBoard):

	def __init__(self, imgs, classes, content_embedding, class_embedding, class_modulation, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs
		self.__classes = classes

		self.__content_embedding = content_embedding
		self.__class_embedding = class_embedding
		self.__class_modulation = class_modulation
		self.__generator = generator

		self.__n_samples_per_evaluation = 5

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]
		classes = self.__classes[img_ids]

		content_codes = self.__content_embedding.predict(img_ids)
		class_codes = self.__class_embedding.predict(classes)
		class_adain_params = self.__class_modulation.predict(class_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		for i in range(self.__n_samples_per_evaluation):
			converted_imgs = [imgs[i]] + [
				self.__generator.predict([content_codes[[j]], class_adain_params[[i]]])[0]
				for j in range(self.__n_samples_per_evaluation)
			]

			output.append(np.concatenate(converted_imgs, axis=1))

		merged_img = np.concatenate(output, axis=0)

		summary = tf.Summary(value=[tf.Summary.Value(tag='sample', image=make_image(merged_img))])
		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()


class TrainEncodersEvaluationCallback(TensorBoard):

	def __init__(self, imgs, content_encoder, class_encoder, class_modulation, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs

		self.__content_encoder = content_encoder
		self.__class_encoder = class_encoder
		self.__class_modulation = class_modulation
		self.__generator = generator

		self.__n_samples_per_evaluation = 10

	def on_epoch_end(self, epoch, logs={}):
		if 'loss' in logs:
			logs['loss-encoders'] = logs.pop('loss')

		if 'lr' in logs:
			logs['lr-encoders'] = logs.pop('lr')

		super().on_epoch_end(epoch, logs)

		img_ids = np.random.choice(self.__imgs.shape[0], size=self.__n_samples_per_evaluation, replace=False)
		imgs = self.__imgs[img_ids]

		content_codes = self.__content_encoder.predict(imgs)
		class_codes = self.__class_encoder.predict(imgs)
		class_adain_params = self.__class_modulation.predict(class_codes)

		blank = np.zeros_like(imgs[0])
		output = [np.concatenate([blank] + list(imgs), axis=1)]
		for i in range(self.__n_samples_per_evaluation):
			converted_imgs = [imgs[i]] + [
				self.__generator.predict([content_codes[[j]], class_adain_params[[i]]])[0]
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
