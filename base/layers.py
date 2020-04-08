"""
#Authors: TamNV
This file implements 4 basic layers in tensorlow 
	+Fully connected, LSTM, Graph Convolutional Neural Network
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from inits import uniform, glorot, zeros, ones
from abstract_layer import *

def leak_relu(x):
	return tf.maximum(x*0.2, x)

class Dense(Layer):
    """
    Dense Layer.
    """
    def __init__(self, input_dim, output_dim, dropout=0.0, sparse_inputs=False,
                    act=tf.nn.relu, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.input_dim = input_dim

        #helper variable for sparse dropout
        self.num_features_nonzero = input_dim

        with tf.variable_scope("{}_vars".format(self.name)):
            self.vars['weights'] = glorot([input_dim, output_dim],\
                                            name='weights')
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")
        
        if self.logging:
            self._log_vars()
        
    def _call(self, inputs):
        x = inputs
        # Dropout
        # x = dropout(x, 1-self.dropout, self.num_features_nonzero, self.sparse_inputs)
        #Transform
        x = tf.reshape(x, (-1, self.input_dim))

        outputs = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        #Bias
        if self.bias:
            outputs += self.vars["bias"]
        
        return self.act(outputs)

class Conv2D(Layer):
	"""
	Convolution 2D
	"""
	def __init__(self, num_in_channels, num_out_channels, filter_size, 
					strides, padding, dropout, bias, act, **kwargs):
		super(Conv2D, self).__init__(**kwargs)

		self.dropout = dropout
		self.num_in_channels = num_in_channels
		self.num_out_channels = num_out_channels
		self.filter_size = filter_size
		self.padding = padding
		self.strides = strides

		self.num_features_nonzero = num_in_channels
		self.bias = bias
		self.act = act
		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["weights"] = glorot([filter_size, filter_size, num_in_channels,
												num_out_channels], name='weights')
			if self.bias:
				self.vars["bias"] = zeros([num_out_channels], name="bias")

	def _call(self, inputs):
		"""
		Perform convolution operation
		
		Params:
			inputs: Tensor object [batch, in_height, in_width, in_channels]

		Returns:
			outputs: Tensor object [batch, out_height, out_width, out_channels]
		"""
		# Perform dropout
		_shape = tf.shape(inputs)
		x = inputs
		# x = dropout(x, 1-self.dropout, self.num_features_nonzero, False)
		x = tf.reshape(x, _shape)

		# Perform convolution operation
		x = tf.nn.conv2d(x, self.vars["weights"],
						[1, self.strides, self.strides, 1], 
						self.padding)
		if self.bias:
			x += self.vars["bias"]

		outputs = self.act(x)
		
		return outputs

class Conv1D(Layer):
	"""
	Convolution 1D
	"""
	def __init__(self, num_in_channels, num_out_channels, filter_size, 
					strides, padding, dropout, bias, act, **kwargs):

		super(Conv1D, self).__init__(**kwargs)

		self.num_in_channels = num_in_channels
		self.num_out_channels = num_out_channels
		self.filter_size = filter_size
		self.strides = strides
		self.padding = padding
		self.dropout = dropout

		self.num_features_nonzero = num_in_channels
		self.bias = bias
		self.act = act

		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["weights"] = glorot([filter_size, self.num_in_channels,
											self.num_out_channels], name="weights")
			if self.bias:
				self.vars["bias"] = zeros([self.num_out_channels], name="bias")

	def _call(self, inputs):
		"""
		Perform convolution 1D

		Params:
			inputs: Tensor object
				[batch, in_width, in_channels]
		Returns:
			outputs: Tensor object
		"""
		_shape = tf.shape(inputs)

		x = inputs

		# x = dropout(x, 1-self.dropout,
					# self.num_features_nonzero, False)

		x = tf.reshape(x, _shape)

		x = tf.nn.conv1d(x, self.vars["weights"],
						[1, self.strides, 1], self.padding)
		if self.bias:
			x += self.vars["bias"]

		outputs = self.act(x)
		return outputs

class MaxPooling2D(Layer):
	"""
	Maxpooling2D
	"""
	def __init__(self, ksize, strides, padding, **kwargs):
		super(MaxPooling2D, self).__init__(**kwargs)

		self.ksize = ksize
		self.strides = strides
		self.padding = padding

	def _call(self, inputs):
		"""
		Perform maxpoling2D operation 
		"""
		outputs = tf.nn.max_pool2d(inputs, [1, self.ksize, self.ksize, 1],
									[1, self.strides, self.strides, 1], 
									self.padding)
		return outputs

class MaxPooling1D(Layer):
	"""
	Maxpooling 1D
	"""
	def __init__(self, ksize, strides, padding, **kwargs):
		super(MaxPooling1D, self).__init__(**kwargs)

		self.ksize = ksize
		self.strides = strides
		self.padding = padding
	
	def _call(self, inputs):
		"""
		Perform maxpooling 1D operation
		"""
		x = tf.expand_dims(inputs, axis=-1)

		outputs = tf.nn.max_pool2d(x, [1, self.ksize, 1, 1],
									[1, self.strides, 1, 1],
									self.padding)
		outputs = tf.squeeze(outputs, axis=-1)

		return outputs

class BiLSTM(Layer):
	"""
	BiLSTM layer
	"""
	def __init__(self, num_units, input_shape,
				num_steps, dropout, **kwargs):
		super(BiLSTM, self).__init__(**kwargs)

		self.num_units = num_units
		self.dropout = dropout
		self.input_shape = input_shape
		self.num_steps = num_steps

		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["forward_lstm"] = rnn.BasicLSTMCell(self.num_units,
														forget_bias=1.0,
														name="forward_lstm_{}".format(self.name))

			self.vars["backward_lstm"] = rnn.BasicLSTMCell(self.num_units,
														forget_bias=1.0,
														name="backward_lstm_{}".format(self.name))
		if self.logging:
			self._log_vars()

	def _call(self, inputs):
		"""
		Inputs are followed by time series format. N_TIME_STEP x N_FEATURES
		"""
		x = inputs
		# Dropout
		# x = dropout(x, 1-self.dropout, self.input_shape, False)
		# BiLSTM layers
		# x = tf.reshape(x, (-1, self.num_steps, self.input_shape))

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.vars["forward_lstm"],
											self.vars["backward_lstm"],
											x, dtype="float32")

		outputs = tf.concat(outputs, axis=2)
		
		return outputs

class LSTM(Layer):
	def __init__(self, num_units, input_shape,
				num_steps, dropout, return_sequences,
				**kwargs):
		"""
		Initialize method

		Params:
			num_units: Integer
				The number of hidden units
			num_times: Integer
				The number of time steps
			dropout: Float32
				The dropout factor
		Returns:
			None
		"""
		super(LSTM, self).__init__(**kwargs)

		self.input_shape = input_shape
		self.num_units = num_units
		self.num_steps = num_steps
		self.dropout = dropout
		self.return_sequences = return_sequences

		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["lstm_cell"] = rnn.BasicLSTMCell(self.num_units, 1.0)

	def _call(self, inputs):
		"""
		Inference methods

		Params:
			inputs: [Features, return_sequences]
				return_sequences controls the hidden states which are selected
				return_sequences = True: Get entire hidden states of all timesteps
				return_sequences = False: Get only hidden states of the last time step

		Returns:
			outputs: Tensor object
		"""

		x = inputs
		
		if type(x) is not list:
			x = tf.unstack(x, self.num_steps, 1)
		
		outputs, _ = rnn.static_rnn(self.vars["lstm_cell"], x, dtype="float32")
		
		if not self.return_sequences:
			outputs = outputs[-1]

		return outputs

class BiLSTMExtract(Layer):
	def __init__(self, num_dims, **kwargs):
		super(BiLSTMExtract, self).__init__(**kwargs)
		self.num_dims = num_dims

	def _call(self, inputs):
		"""
		Get forward features and backward feature and concatenate them
		"""
		forward = inputs[:, -1, 0:self.num_dims]
		backward = inputs[:, 0, self.num_dims:]
		outputs = tf.concat([forward, backward], axis=-1)

		return outputs

class Flatten(Layer):
	def __init__(self, num_dims, **kwargs):
		super(Flatten, self).__init__(**kwargs)
		self.num_dims = num_dims

	def _call(self, inputs):
		outputs = tf.reshape(inputs, (-1, self.num_dims))
		return outputs

class CenterLoss(Layer):
	"""
	Perform center loss
	"""
	def __init__(self, num_classes, num_feas, learning_rate, **kwargs):
		super(CenterLoss, self).__init__(**kwargs)		
		
		self.num_classes = num_classes
		self.num_feas = num_feas
		self.learning_rate = learning_rate

		# Declare variables
		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["center"] = zeros(shape=[self.num_classes, self.num_feas], 
										trainable=False)	

	def _call(self, inputs):
		"""
		Perform center loss layer

		Params:
			inputs: Tensor object
				Embedding features:	N_Classes x N_Embedding
			labels: Tensor object
				Labels of this batchs: N_Samples x N x Classes 
		Returns:
			center loss optimizer
		"""
		embeded_preds = inputs[0]
		labels = inputs[1]


		_labels = tf.cast(labels, tf.float32)

		embeded_labels = tf.matmul(_labels, self.vars["center"])

		diff = embeded_labels - embeded_preds

		_labels = tf.transpose(_labels)
		grad = tf.matmul(_labels, diff)

		center_counts = tf.reduce_sum(_labels, axis=1, keepdims=True) + 1.0
		
		grad /= center_counts

		updated_center = self.vars["center"] - self.learning_rate * grad
		
		center_loss_opt = tf.assign(self.vars["center"], updated_center)

		return center_loss_opt

class GraphConv(Layer):
	"""Perform graph convolution layer"""
	def __init__(self, num_secs, num_steps, filters, **kwargs):
		"""
		Initialize method

		Params:
			num_secs: integer
				the number of input sections
			filters:  list
				reception fields
		Returns:
			None
		"""
		super(GraphConv, self).__init__(**kwargs)

		self.num_secs = num_secs
		self.num_steps = num_steps
		self.filters = filters

		with tf.variable_scope("{}_vars".format(self.name)):
			for i in range(len(self.filters)):
				self.vars["weights_{}".format(i)] = glorot([self.num_secs, num_secs],
														name="weights_{}".format(i))
		if self.logging:
			self._log_vars()

	def _call(self, inputs):
		"""
		Implements call method

		Params:
			inputs: tensor object
				Batchsize x num_steps x num_secs
		Returns:
			embedded features
				Batchsize x num_steps x num_secs 
		"""
		x = tf.unstack(inputs, self.num_steps, 1)

		spatial_feas = []
		for feas in x:
			gcn_feas = feas
			gcn_feas = 0
			for i in range(len(self.filters)):
				masked_filer = tf.matmul(
									self.vars["weights_{}".format(i)], 
									self.filters[i]
								)
				neiborhood_feas = tf.matmul(feas, masked_filer)
				neiborhood_feas = tf.nn.tanh(neiborhood_feas)

				gcn_feas += neiborhood_feas

			spatial_feas.append(gcn_feas)

		return spatial_feas
