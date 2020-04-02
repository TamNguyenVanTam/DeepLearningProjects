"""
#Authors: TamNV

This file implements wrapper layers in tensorlow
"""
import tensorflow as tf

_NAME2ID = {}

def get_layer_uid(name=""):
	"""
	Assign layer to a unique IDs
	"""
	if name not in _NAME2ID:
		_NAME2ID[name] = 1
		return 1
	else:
		_NAME2ID[name] += 1
		return _NAME2ID[name]

def dropout(x, keep_prob, noise_shape, is_sparse=False):
	"""
	Perform dropout
	"""
	if keep_prob == 0:
		return x
	random = tf.random_uniform([noise_shape])
	
	random_tensor = tf.add(random, keep_prob)
	
	dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
	if is_sparse:
		pre_out = tf.sparse_retain(x, dropout_mask)
	else:
		pre_out = tf.nn.dropout(x, keep_prob)

	return pre_out *1.0 / keep_prob

def dot(x, y, sparse=False):
	"""
	Perform mutiply two matrics
	"""
	if sparse:
		res = tf.sparse_tensor_dense_matmul(x, y)
	else:
		res = tf.matmul(x, y)
	return res

class Layer(object):
	"""
	Base layer class. Defines basic API for all layer object
	#Properties:
		name: String, defines the variable scope of the layer
		logging: Boolean, switches Tensorflow histogram logging on/off

	#Methods:
		_call(inputs): Defines computation graph of layer
			takes input, returns output
		__call__(inputs): Wrapper for _call
		_log_vars(): log all variables
	"""
	def __init__(self, **kwargs):
		allowed_kwargs = {"name", "logging"}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, "Invalid Keyword Argument: {}".format(kwarg)
		name = kwargs.get("name")
		if not name:
			layer = self.__class__.__name__.lower()
			name = "{}_{}".format(layer, get_layer_uid(layer))
		self.name = name
		self.vars = {}
		logging = kwargs.get("logging", False)

		self.logging = logging
		self.sparse_inputs = False

	def _call(self, inputs):
		return inputs

	def __call__(self, inputs):
		with tf.name_scope(self.name):
			if self.logging and not self.sparse_inputs:
				tf.summary.histogram(self.name + "/inputs", inputs)
			outputs = self._call(inputs)

			if self.logging:
				tf.summary.histogram(self.name + "/outputs", outputs)
			return outputs

	def _log_vars(self):
		for var in self.vars:
			tf.summary.histogram(self.name + "/vars/" + var, self.vars[var])
