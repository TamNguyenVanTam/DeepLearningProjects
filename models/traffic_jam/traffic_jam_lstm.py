"""
Authors: TamNV
This file implements LSTM model
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, "../../base")

from layers import LSTM, Dense
from abstract_model import  Model
from utils import *
from metrics import *

class LSTM1(Model):
	"""
	This class uses only LSTM layer
	This is considered as the benmask model
	"""
	def __init__(self, placeholders, model_configs, **kwargs):
		"""
		Initialize method

		Params:
			placeholders: List of placeholders
				Which are used for building LSTM model
		Returns:
			None
		"""
		super(LSTM1, self).__init__(**kwargs)

		self.inputs = placeholders["features"]
		self.labels = placeholders["labels"]
		self.learning_rate = placeholders["learning_rate"]
		self.dropout = placeholders["dropout"]
		self.decay_factor = placeholders["weight_decay"]

		self.num_classes = self.labels.get_shape().as_list()[1]
		self.optimizer = tf.train.AdamOptimizer(
								learning_rate=self.learning_rate)
		self.model_configs = model_configs
		self.build(self.model_configs)

	def _build(self, model_configs):
		"""
		This method defines layers
			which we will be use in this model
		Params:
			None
		Returns:
			None
		"""
		num_units = 64
		self.layers.append(LSTM(num_units=num_units,
								input_shape=model_configs[KEY_NUM_SECS],
								num_steps=model_configs[KEY_NUM_IN_TIME_STEPS],
								dropout=0.0))
		
		self.layers.append(Dense(input_dim=num_units,
								output_dim=model_configs[KEY_NUM_OUT_TIME_STEPS]
											* model_configs[KEY_NUM_SECS],
								dropout=0.0,
								act=lambda x:x,
								bias=True))

	def build(self, model_configs):
		"""
		Wrapper for _build
		"""
		with tf.variable_scope(self.name):
			self._build(model_configs)

		#Build sequential layer model
		self.activations.append([self.inputs, False])
		for layer in self.layers:
			print(layer)
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		print("Modeling sucessful!")
		self.outputs = self.activations[-1]
		self.outputs = tf.reshape(self.outputs,
									(-1, model_configs[KEY_NUM_OUT_TIME_STEPS], model_configs[KEY_NUM_SECS])
								)
		#Store model variables for easy access
		self.vars = {var.name:var for var in tf.trainable_variables()}

		self._loss()
		self._accuracy()
		self.opt_op = self.optimizer.minimize(self.loss)

	def _loss(self):
		"""
		Define the loss function use for this project
		
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
		self.rmse_loss = get_rmse_loss(self.labels, self.outputs)
		# Total loss
		self.loss = self.reg_loss + self.rmse_loss

		self.mae = get_mae(self.labels, self.outputs)
		self.mape = get_mape(self.labels, self.outputs)

	def _accuracy(self):
		"""
		Not use this metric 
		"""
		pass

	def predict(self):
		"""
		Perform predicting
		"""
		return self.outputs
