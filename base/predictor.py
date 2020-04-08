
"""
Authors: TamNV
"""
import numpy as np
import tensorflow as tf
from utils import *

class Predictor:
	"""
	perform predicting phase
	"""
	def __init__(self, exe_config):
		"""
		Initialize method

		Params:
			exe_config: dictionary
				configuration of prediction process
		Returns:
			None
		"""
		self.exe_config = exe_config

	def predict(self, model, data, sess):
		"""
		Wrapper of do predicting

		Params:
			model: model instance
			data: data manager instance
			sess: tf.Session()
		Returns:
			None
		"""
		self.model = model
		self.data = data
		self.sess = sess
		#Load trained model
		self.model.load(self.exe_config[KEY_CHECKPOINT], self.sess)
		
		self.num_test_iters = int(np.floor(self.data.test_x.shape[0]
											/self.exe_config[KEY_BATCH_SIZE]
							))
		self._predict()

	def _predict(self):
		"""
		perform predicting
		"""
		pass

		


