"""
@Authors: TamNV
Implement traffic jam predictor
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, "../../base")

from predictor import Predictor
from utils import *

class TrafficJamPredictor(Predictor):
	"""
	modify predict solution
	"""
	def __init__(self, filters, exe_config):
		"""
		Initilize method

		Params:
			exe_config: dictionary
				configuration of predicting process
		Returns:
			none
		"""
		self.filters = filters
		self.exe_config = exe_config

	def _predict(self):
		"""
		overwritting predicting method
		Params:
			None
		Returns:
			None
		"""
		test_log = []
		for i in range(self.num_test_iters):
			batch_x, batch_y = self.data.get_next_batch_test(
										self.exe_config[KEY_BATCH_SIZE]
								)

			feed_dict = {
							self.model.inputs: batch_x,
							self.model.labels: batch_y,
							self.model.dropout: 0.0,
						}
			
			for i in range(len(self.filters)):
				feed_dict.update({self.model.filters[i]:self.filters[i]})
			
			rmse, mae, mape = self.sess.run([self.model.rmse, self.model.mae, self.model.mape], 
											feed_dict=feed_dict)

			test_log.append([rmse, mae, mape])
		test_log = np.array(test_log)
		print("Test set: RMSE: {:.3f}, MAE {:.3f}, MAPE {:.3f}"
			.format(np.mean(test_log[:,0]), np.mean(test_log[:, 1]), np.mean(test_log[:, 2])))

	def get_impact_between_nodes(self):
		trained_filters = self.sess.run(self.model.trained_filters)
		return trained_filters
