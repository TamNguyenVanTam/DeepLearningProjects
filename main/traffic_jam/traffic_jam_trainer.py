"""
Authors: TamNV
"""
import sys
import numpy as np

sys.path.insert(0, "../../base")

from trainer import Trainer
from utils import *

class TrafficJamTrainer(Trainer):
	"""
	perform traffic jam training process
	"""
	def __init__(self, filters, exe_config, **kwargs):
		self.filters = filters
		self.exe_config = exe_config
	
	def _train(self):
		"""
		Minimize Model's loss function

		Params:
			None
		Returns:
			None
		"""
		best_rmse = 10000000.0
		
		for epoch in range(self.exe_config[KEY_NUM_EPOCHS]):
			print("epoch {}".format(epoch))
			# perfoming training
			train_log, valid_log = [], []
			for i in range(self.num_train_iters):
				batch_x, batch_y =  self.data.get_next_batch_train(
											self.exe_config[KEY_BATCH_SIZE]
									)
				feed_dict = {
								self.model.inputs: batch_x,
								self.model.labels: batch_y,
								self.model.learning_rate: self.exe_config[KEY_LEARNING_RATE],
								self.model.dropout: self.exe_config[KEY_DROPOUT],
								self.model.decay_factor: self.exe_config[KEY_DECAY_FACTOR]
							}
				if self.filters is not None:
					for i in range(len(self.filters)):
						feed_dict.update({self.model.filters[i]: self.filters[i]})

				_, loss, rmse, mae, mape = self.sess.run(
											[self.model.opt_op, self.model.loss, 
												self.model.rmse, self.model.mae, self.model.mape],
											feed_dict=feed_dict
									)
				train_log.append([loss, rmse, mae, mape])
			
			for i in range(self.num_valid_iters):
				batch_x, batch_y = self.data.get_next_batch_valid(
										self.exe_config[KEY_BATCH_SIZE]
								)
				feed_dict = {
								self.model.inputs: batch_x,
								self.model.labels: batch_y,
								self.model.dropout: 0.0,
								self.model.decay_factor: self.exe_config[KEY_DECAY_FACTOR]
							}
				if self.filters is not None:
					for i in range(len(self.filters)):
						feed_dict.update({self.model.filters[i]: self.filters[i]})

				loss, rmse, mae, mape = self.sess.run(
											[self.model.loss, self.model.rmse, 
												self.model.mae, self.model.mape],
											feed_dict=feed_dict
										)
				valid_log.append([loss, rmse, mae, mape])
			
			train_log, valid_log = np.array(train_log), np.array(valid_log)
			print("Training Set")
			print("\tTotal loss {:.3f}, RMSE {:.3f}, MAE {:.3f}, MAPE {:.3f}".format(
					np.mean(train_log[:, 0]), np.mean(train_log[:, 1]),
					np.mean(train_log[:, 2]), np.mean(train_log[:, 3])
			))
			print("Validation Set")
			print("\tTotal loss {:.3f}, RMSE {:.3f}, MAE {:.3f}, MAPE {:.3f}".format(
					np.mean(valid_log[:, 0]), np.mean(valid_log[:, 1]),
					np.mean(valid_log[:, 2]), np.mean(valid_log[:, 3])
			))
			if best_rmse > np.mean(valid_log[:, 1]):
				best_rmse = np.mean(valid_log[:, 1])
				best_mae = np.mean(valid_log[:, 2])
				best_mape = np.mean(valid_log[:, 3])
				self.model.save(self.exe_config[KEY_CHECKPOINT], self.sess)

		print("Optimizal point. RMSE {:.3f}, MAE {:.3f}, MAPE {:.3f}"
					.format(best_rmse, best_mae, best_mape))
