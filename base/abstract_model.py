"""
Authors: TamNV
This file implements abstract model
"""

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

	def _build(self):
		raise NotImplementedError

	def build(self):
		"""
		Wrapper for _build
		"""
		with tf.variable_scope(self.name):
			self._build()
		#Build sequential layer model
		self.activations.append(self.inputs)
		for layer in self.layers:
			print(layer)
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		print("Modeling sucessful")
		self.outputs = self.activations[-1]

		#Store model variables for easy access
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name : var for var in variables}
		self._loss()
		self._accuracy()
		self.opt_op = self.optimizer.minimize(self.loss)

	def predict(self):
		pass

	def _loss(self):
		raise NotImplementedError

	def _accuracy(self):
		raise NotImplementedError

	def save(self, sess=None):
		if not sess:
			raise AttributeError("Tensorflow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = saver.save(sess, "tmp/{}.ckpt".format(self.name))
		print("Model Saved in file {}".format(save_path))

	def load(self, sess=None):
		if not sess:
			raise AttributeError("Tensorflow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = "tmp/{}.ckpt".format(self.name)
		sess.restore(sess, save_path)
		print("Model restored from file: {}".format(save_path))
