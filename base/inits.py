"""
Authors: TamNV
This file implements some initial methods
"""

import tensorflow as tf
import numpy as np

def uniform(shape, scale=0.05, name=None, trainable=True):
	"""
	Create a random variable follow by uniform distribution
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		scale: Float Number
			The variance in uniform distribution
		name: String
			Variable name
	Returns:
		Tensor
	"""
	kernel = tf.random_uniform(shape, minval=-scale,
							maxval=scale,dtype=tf.float32,
							seed=1)

	return tf.Variable(kernel, name=name, trainable=trainable)

def glorot(shape, name=None, trainable=True):
	"""
	Create a random variable follow by Glorot & Bengio (AISTATS 2010)
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		name: String
			Variable name
	Returns:
		Tensor
	"""
	scale = np.sqrt(6.0/(shape[0]+shape[1]))
	kernel = tf.random_uniform(shape, minval=-scale,
								maxval=scale, dtype=tf.float32,
								seed=1)
	return tf.Variable(kernel, trainable=trainable)

def zeros(shape, name=None, trainable=True):
	"""
	Create a random variable what have all elements equal 0
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		name: String
			Variable name
	Returns:
		Tensor
	"""
	kernel = tf.zeros(shape, dtype=tf.float32)
	return tf.Variable(kernel, name=name, trainable=trainable)

def ones(shape, name=None, trainable=True):
	"""
	Create a random variable what have all elements equal 1
	Params:
		shape: A 1-D intege Python array
			The shape of the output tensor
		name: String
			Variable name
	Returns:
		Tensor
	"""
	kernel = tf.ones(shape, dtype=tf.float32)
	return tf.Variable(kernel, name=name, trainable=trainable)

