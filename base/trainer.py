"""
Authors: TamNV
"""
import numpy as np
import tensorflow as tf

from utils import *

class Trainer:
	"""
	Perfrom training phase
	"""
	def __init__(self, exe_config):
		"""
		Initialize method

		Params:
			exe_config: Dictionary
				configuration of training process
		Returns:
			None
		"""
		self.exe_config = exe_config

	def train(self, model, data, sess):
		"""
		Wrapper of do training

		Params:
			model: Model instance
			data: DataManager instance
			sess: tf.Session()
		Returns:
			None
		"""
		self.model = model
		self.data = data
		self.sess = sess

		self.sess.run(tf.global_variables_initializer())

		self.num_train_iters = int(np.floor(self.data.train_x.shape[0]
											/self.exe_config[KEY_BATCH_SIZE]
								))
		self.num_valid_iters = int(np.floor(self.data.valid_x.shape[0]
											/self.exe_config[KEY_BATCH_SIZE]
								))
		self._train()

	def _train(self):
		"""
		perform training
		"""	
		pass
