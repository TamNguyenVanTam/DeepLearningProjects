#Authors: TamNV
import os
import numpy as np

class DataManager:

	def __init__(self, train_x, train_y,
				valid_x, valid_y, test_x,
				test_y):
		"""
		Initialize method

		Params:
			train_x, train_y, valid_x, valid_y, test_x, test_y: Number Array
				Which correspond inputs and outputs of 
					training set, validation set, testing set
		Returns:
			None
		"""
		self.train_x = train_x
		self.train_y = train_y
		self.valid_x = valid_x
		self.valid_y = valid_y
		self.test_x = test_x
		self.test_y = test_y

		# Intialize properties
		self.cur_train_idx = 0
		self.cur_valid_idx = 0
		self.cur_test_idx = 0

		self.train_idxs = np.arange(self.train_x.shape[0])
		self.valid_idxs = np.arange(self.valid_x.shape[0])
		self.test_idxs = np.arange(self.test_x.shape[0])

	def get_next_batch_train(self, batch_size):
		"""
		Get one training batch

		Params:
			batch_size: Integer
				number samples of this batch
		Returns:
			batch_x, batch_y: Numpy Array 
		"""
		next_idx = min(self.train_x.shape[0],
						self.cur_train_idx + batch_size
					)
		batch_idxs = self.train_idxs[self.cur_train_idx:next_idx]

		batch_x = self.train_x[batch_idxs]
		batch_y = self.train_y[batch_idxs]

		if next_idx == self.train_x.shape[0]:
			self.cur_train_idx = 0
			np.random.shuffle(self.train_idxs)
		else:
			self.cur_train_idx = next_idx

		return batch_x, batch_y

	def get_next_batch_valid(self, batch_size):
		"""
		Get one validation batch

		Params:
			batch_size: Integer
				number samples of this batch
		Returns:
			batch_x, batch_y: Numpy Array 
		"""
		next_idx = min(self.valid_x.shape[0],
						self.cur_valid_idx + batch_size
					)
		batch_idxs = self.valid_idxs[self.cur_valid_idx:next_idx]

		batch_x = self.valid_x[batch_idxs]
		batch_y = self.valid_y[batch_idxs]

		if next_idx == self.valid_x.shape[0]:
			self.cur_valid_idx = 0
		else:
			self.cur_valid_idx = next_idx

		return batch_x, batch_y

	def get_next_batch_test(self, batch_size):
		"""
		Get one testing batch

		Params:
			batch_size: Integer
				number samples of this batch
		Returns:
			batch_x, batch_y: Numpy Array 
		"""
		next_idx = min(self.test_x.shape[0],
						self.cur_test_idx + batch_size
					)
		batch_idxs = self.test_idxs[self.cur_test_idx:next_idx]

		batch_x = self.test_x[batch_idxs]
		batch_y = self.test_y[batch_idxs]

		if next_idx == self.test_x.shape[0]:
			self.cur_test_idx = 0
		else:
			self.cur_test_idx = next_idx

		return batch_x, batch_y








