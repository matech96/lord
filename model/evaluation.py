import random
import io

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.callbacks import TensorBoard


class EvaluationCallback(TensorBoard):

	def __init__(self, converter, imgs, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)

		self.__converter = converter
		self.__imgs = imgs

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		idx = np.random.choice(self.__imgs.shape[0], size=2, replace=False)
		imgs = self.__imgs[idx]
		imgs = imgs.astype(np.float64) / 255

		reconstructed_img = self.__converter.converter.predict([imgs[[0]], imgs[[0]]])[0]
		reconstructed_merged_img = np.concatenate((imgs[0], imgs[0], reconstructed_img), axis=1)

		converted_img = self.__converter.converter.predict([imgs[[0]], imgs[[1]]])[0]
		converted_merged_img = np.concatenate((imgs[0], imgs[1], converted_img), axis=1)

		reconstructed_summary = tf.Summary(value=[tf.Summary.Value(tag='reconstructed', image=self.make_image(reconstructed_merged_img))])
		converted_summary = tf.Summary(value=[tf.Summary.Value(tag='converted', image=self.make_image(converted_merged_img))])

		self.writer.add_summary(reconstructed_summary, global_step=epoch)
		self.writer.add_summary(converted_summary, global_step=epoch)
		self.writer.flush()

	@staticmethod
	def make_image(tensor):
		height, width, channel = tensor.shape
		image = Image.fromarray((tensor * 255).astype(np.uint8))

		with io.BytesIO() as out:
			image.save(out, format='PNG')
			image_string = out.getvalue()

		return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
