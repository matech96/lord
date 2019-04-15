import random
import io

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.callbacks import TensorBoard


class EvaluationCallback(TensorBoard):

	def __init__(self, pose_codes, identity_codes, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)

		self.__pose_codes = pose_codes
		self.__identity_codes = identity_codes

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		object_id = random.choice(list(self.__identity_codes.keys()))
		identity_code = self.__identity_codes[object_id]

		idx = np.random.choice(self.__pose_codes[object_id].shape[0], size=1)
		pose_code_a = self.__pose_codes[object_id][idx]

		idx = np.random.choice(self.__pose_codes[object_id].shape[0], size=1)
		pose_code_b = self.__pose_codes[object_id][idx]

		img_a = self.model.predict([pose_code_a, identity_code])[0]
		img_b = self.model.predict([pose_code_b, identity_code])[0]

		merged_img = np.concatenate((img_a, img_b), axis=1)

		summary = tf.Summary(value=[tf.Summary.Value(tag='sample', image=self.make_image(merged_img))])

		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()

	@staticmethod
	def make_image(tensor):
		height, width = tensor.shape
		image = Image.fromarray((np.squeeze(tensor) * 255).astype(np.uint8))

		with io.BytesIO() as out:
			image.save(out, format='PNG')
			image_string = out.getvalue()

		return tf.Summary.Image(height=height, width=width, colorspace=1, encoded_image_string=image_string)
