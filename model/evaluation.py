import io

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.callbacks import TensorBoard


class EvaluationCallback(TensorBoard):

	def __init__(self, imgs, identities, pose_embedding, identity_embedding, generator, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__imgs = imgs
		self.__identities = identities

		self.__pose_embedding = pose_embedding
		self.__identity_embedding = identity_embedding
		self.__generator = generator

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		imgs_id_a = np.random.choice(self.__imgs.shape[0], size=1)
		imgs_id_b = np.random.choice(self.__imgs.shape[0], size=1)

		pose_code_a = self.__pose_embedding.predict(imgs_id_a)
		pose_code_b = self.__pose_embedding.predict(imgs_id_b)

		identity_a = self.__identities[imgs_id_a]
		identity_b = self.__identities[imgs_id_b]

		identity_code_a = self.__identity_embedding.predict(identity_a)
		identity_code_b = self.__identity_embedding.predict(identity_b)

		img_a_a = self.model.predict([pose_code_a, identity_code_a])[0]
		img_b_a = self.model.predict([pose_code_b, identity_code_a])[0]
		img_a_b = self.model.predict([pose_code_a, identity_code_b])[0]
		img_b_b = self.model.predict([pose_code_b, identity_code_b])[0]

		merged_img = np.concatenate((
			np.concatenate((img_a_a, img_b_a), axis=1),
			np.concatenate((img_a_b, img_b_b), axis=1)
		), axis=0)

		summary = tf.Summary(value=[tf.Summary.Value(tag='sample', image=self.make_image(merged_img))])

		self.writer.add_summary(summary, global_step=epoch)
		self.writer.flush()

	@staticmethod
	def make_image(tensor):
		height, width, channels = tensor.shape
		image = Image.fromarray((np.squeeze(tensor) * 255).astype(np.uint8))

		with io.BytesIO() as out:
			image.save(out, format='PNG')
			image_string = out.getvalue()

		return tf.Summary.Image(height=height, width=width, colorspace=channels, encoded_image_string=image_string)
