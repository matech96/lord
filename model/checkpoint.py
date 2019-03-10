from keras.callbacks import Callback


class PeriodicCallback(Callback):

	def __init__(self, n_epochs_in_period):
		super().__init__()

		self.__n_epochs_in_period = n_epochs_in_period
		self.__epochs_since_last_period = 0

	def on_epoch_end(self, epoch, logs=None):
		self.__epochs_since_last_period += 1
		if self.__epochs_since_last_period >= self.__n_epochs_in_period:
			self.__epochs_since_last_period = 0

			self.on_period_end(epoch)

	def on_period_end(self, epoch):
		pass


class MultiModelCheckpoint(PeriodicCallback):

	def __init__(self, saver, model_dir, n_epochs):
		super().__init__(n_epochs_in_period=n_epochs)

		self.__saver = saver
		self.__model_dir = model_dir

	def on_period_end(self, epoch):
		print('checkpointing at epoch %d...' % epoch)
		self.__saver.save(self.__model_dir)
