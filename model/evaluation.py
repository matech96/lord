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

		self.__identity_ids = list(self.__imgs.keys())

	def on_epoch_end(self, epoch, logs={}):
		super().on_epoch_end(epoch, logs)

		source_identity_id = random.choice(self.__identity_ids)
		idx = np.random.randint(0, self.__imgs[source_identity_id]['imgs'].shape[0], size=2)
		source_identity_imgs = self.__imgs[source_identity_id]['imgs'][idx]
		source_identity_masks = self.__imgs[source_identity_id]['masks'][idx]

		masked_source_identity_imgs = source_identity_imgs * source_identity_masks[..., np.newaxis]

		masked_reconstructed_img = self.__converter.converter.predict([
			masked_source_identity_imgs[[0]], masked_source_identity_imgs[[1]]
		])[0]

		reconstructed_img = (
			masked_reconstructed_img * source_identity_masks[0][..., np.newaxis]
			+ source_identity_imgs[0] * (1 - source_identity_masks[0][..., np.newaxis])
		)

		reconstructed_merged_img = np.concatenate((source_identity_imgs[0], source_identity_imgs[1], reconstructed_img), axis=1)

		target_identity_id = random.choice(self.__identity_ids)
		idx = np.random.randint(0, self.__imgs[target_identity_id]['imgs'].shape[0], size=1)
		target_identity_img = self.__imgs[target_identity_id]['imgs'][idx]
		target_identity_mask = self.__imgs[target_identity_id]['masks'][idx]

		masked_target_identity_img = target_identity_img * target_identity_mask[..., np.newaxis]

		masked_converted_img = self.__converter.converter.predict([
			masked_source_identity_imgs[[0]], masked_target_identity_img
		])[0]

		converted_img = (
			masked_converted_img * source_identity_masks[0][..., np.newaxis]
			+ source_identity_imgs[0] * (1 - source_identity_masks[0][..., np.newaxis])
		)

		converted_merged_img = np.concatenate((source_identity_imgs[0], target_identity_img[0], converted_img), axis=1)

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
