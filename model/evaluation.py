import random
import io

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.callbacks import TensorBoard


class EvaluationCallback(TensorBoard):

	def __init__(self, generator, identity_embedding, tensorboard_dir):
		super().__init__(log_dir=tensorboard_dir)
		super().set_model(generator)

		self.__generator = generator
		self.__identity_embedding = identity_embedding

	def call(self, epoch, logs, pose_codes):
		super().on_epoch_end(epoch, logs)

		identities = list(pose_codes.keys())

		object_id_a = random.choice(list(identities))
		object_id_b = random.choice(list(identities))

		identity_a = np.array([identities.index(object_id_a)])[np.newaxis, ...]
		identity_b = np.array([identities.index(object_id_b)])[np.newaxis, ...]

		identity_code_a = self.__identity_embedding.predict(identity_a)
		identity_code_b = self.__identity_embedding.predict(identity_b)

		idx_a = np.random.choice(pose_codes[object_id_a].shape[0], size=1)
		pose_code_a = pose_codes[object_id_a][idx_a]

		idx_b = np.random.choice(pose_codes[object_id_b].shape[0], size=1)
		pose_code_b = pose_codes[object_id_b][idx_b]

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
