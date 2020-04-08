"""
Authors: TamNV
Implement graph convolution version1
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, "../../base")
from layers import LSTM, Dense, GraphConv
from abstract_model import Model

from utils import *
from metrics import *

class GraphCNNLSTM1(Model):
	"""
	this class uses 2 graph convolution layers
		and one LSTM layer
	"""
	def __init__(self, placeholders, model_configs, **kwargs):
		"""
		Initialize method

		Params:
			placeholders: List of placeholder
				which are used for building GraphCNNLSTM1 model 
			model_configs: Dictionary
		Returns:
			None
		"""
		super(GraphCNNLSTM1, self).__init__(**kwargs)
		# define paramaters
		self.inputs = placeholders["features"]
		self.labels = placeholders["labels"]
		self.learning_rate = placeholders["learning_rate"]
		self.dropout = placeholders["dropout"]
		self.decay_factor = placeholders["weight_decay"]
		self.filters = placeholders["filters"]
		# define optimizer
		self.optimizer = tf.train.AdamOptimizer(
								learning_rate=self.learning_rate
						)
		self.model_configs = model_configs
		# build model
		self.build(self.model_configs)

		self.trained_filters = []
		for _var_names in self.layers[0].vars.keys():
			self.trained_filters.append(self.layers[0].vars[_var_names])
		
	def _build(self, model_configs):
		"""
		This method defines layers
			which we will use in this model
		Params:
			None
		Returns:
			None
		"""
		num_units = 64
		
		self.layers.append(GraphConv(num_secs=self.model_configs[KEY_NUM_SECS],
									num_steps=self.model_configs[KEY_NUM_IN_TIME_STEPS],
									filters=self.filters))

		self.layers.append(LSTM(num_units=num_units,
								input_shape=model_configs[KEY_NUM_SECS],
								num_steps=model_configs[KEY_NUM_IN_TIME_STEPS],
								dropout=0.0,
								return_sequences=False))

		self.layers.append(Dense(input_dim=num_units,
								output_dim=model_configs[KEY_NUM_OUT_TIME_STEPS]
											* model_configs[KEY_NUM_SECS],
								dropout=0.0,
								act=lambda x:x,
								bias=True))
		
	def _loss(self):
		"""
		Define the loss function
		
		Params:
			None
		Returns:
			None
		"""
		# Caculate regulazation loss
		self.reg_loss = 0
		for var in tf.trainable_variables():
			self.reg_loss += self.decay_factor * tf.nn.l2_loss(var)
		# Caculate rmse loss
		self.rmse = get_rmse_loss(self.labels, self.outputs)
		# Total loss
		self.loss = self.reg_loss + self.rmse

		self.mae = get_mae(self.labels, self.outputs)
		self.mape = get_mape(self.labels, self.outputs)

	def _accuracy(self):
		"""
		Not use this metric
		"""
		pass
	
	def predict(self):
		return self.outputs

