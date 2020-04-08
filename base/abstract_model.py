"""
Authors: TamNV
This file implements abstract model
"""
import os
import tensorflow as tf

class Model(object):
	def __init__(self, **kwargs):
		allowed_kwargs = {"name", "logging"}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, "Invalid Keyword Argument: {}".format(kwarg)
		name = kwargs.get("name")

		if not name:
			name = self.__class__.__name__.lower()
		self.name = name

		logging = kwargs.get("logging", False)
		self.logging = logging

		self.vars = {}
		self.placeholders = {}

		self.layers = []
		self.activations = []

		self.loss = 0
		self.accuracy = 0
		
		self.inputs = None
		self.outputs = None

		self.optimizer = None
		self.opt_op = None

	def _build(self, model_configs):
		raise NotImplementedError

	def build(self, model_configs):
		"""
		Wrapper for _build
		"""
		with tf.variable_scope(self.name):
			self._build(model_configs)
		#Build sequential layer model
		self.activations.append(self.inputs)
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		
		print("Modeling sucessfully")
		self.outputs = self.activations[-1]

		#Store model variables for easy access
		self._loss()
		self._accuracy()
		self.opt_op = self.optimizer.minimize(self.loss)

	def predict(self):
		pass

	def _loss(self):
		raise NotImplementedError

	def _accuracy(self):
		raise NotImplementedError

	def save(self, save_path, sess=None):
		if not sess:
			raise AttributeError("Tensorflow session not provided.")
		saver = tf.train.Saver()
		save_path = os.path.join(save_path, "{}.ckpt".format(self.name))
		save_path = saver.save(sess, save_path)
		print("Model Saved in file {}".format(save_path))

	def load(self, save_path, sess=None):
		if not sess:
			raise AttributeError("Tensorflow session not provided.")
		saver = tf.train.Saver()
		save_path = os.path.join(save_path, "{}.ckpt".format(self.name))
		saver.restore(sess, save_path)
		print("Model restored from file: {}".format(save_path))
