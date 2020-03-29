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
        x = dropout(x, 1-self.dropout, self.num_features_nonzero, self.sparse_inputs)
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
		x = dropout(x, 1-self.dropout, self.num_features_nonzero, False)
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

		x = dropout(x, 1-self.dropout,
					self.num_features_nonzero, False)

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
		x = dropout(x, 1-self.dropout, self.input_shape, False)
		# BiLSTM layers
		x = tf.reshape(x, (-1, self.num_steps, self.input_shape))

		outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.vars["forward_lstm"],
											self.vars["backward_lstm"],
											x, dtype="float32")

		outputs = tf.concat(outputs, axis=2)
		
		return outputs

class LSTM(Layer):
	def __init__(self, num_units, input_shape,
				num_steps, dropout, **kwargs):
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

		with tf.variable_scope("{}_vars".format(self.name)):
			self.vars["lstm_cell"] = rnn.BasicLSTMCell(self.num_units, 1.0)

	def _call(self, inputs):
		"""
		Inference methods

		Params:
			inputs: Tensor object
		Returns:
			outputs: Tensor object
		"""

		x = inputs

		# Perform dropout
		x = dropout(x, 1-self.dropout, self.input_shape, False)
		x = tf.reshape(x, (-1, self.num_steps, self.input_shape))

		x = tf.unstack(x, self.num_steps, 1)

		outputs, _ = rnn.static_rnn(self.vars["lstm_cell"], x, dtype="float32")
		
		return outputs[-1]

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
			self.vars["center"] = uniform(
										shape=[self.num_classes, self.num_feas], 
										scale=0.05, name=None, trainable=False
									)	

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

		updated_center = self.vars["center"] - self.learning_rate * grad
		
		center_loss_opt = tf.assign(self.vars["center"], updated_center)

		return center_loss_opt

class GraphConv(Layer):
	"""Perform graph convolution layer"""
	def __init__(self, input_dim, output_dim, adj_matrics, num_nodes,
				dropout=0.0, act=tf.nn.relu, bias=False, **kwargs):
		"""
		Initialize method:

		Params:
			input_dim: Integer
				The number of feature dimensions in inputs
			output_dim: Integer
				The number of feature dimensions in outputs
			adj_matrics: A list
				maps consists of k-hop maps
			dropout: Float
				Dropout factor
			act: Activation function
		"""
		super(GraphConv, self).__init__(**kwargs)

		self.dropout = dropout
		self.act = act
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.bias = bias
		self.adj_mas =adj_matrics
		self.num_nodes = num_nodes

		with tf.variable_scope("{}_vars".format(self.name)):
			for i in range(len(self.adj_mas)):
				self.vars["weights_{}".format(i)] = glorot([input_dim, output_dim],
															name = "weights_{}".format(i))

			if self.bias:
				self.vars["bias"] = zeros([output_dim], name='bias')
		
		if self.logging:
			self._log_vars()

	def _call(self, inputs, is_time_series=True, num_times=2):
		"""
		Implements call method
	
		Params:
			inputs: Tensor Object
				Inputs either Time series format or Instance class
			is_time_series: Boolean 
				Check kind of data
		"""
		if is_time_series:
			x = tf.unstack(inputs, num_times, 1)
		else:
			x = [inputs]
		#Perform dropout
		x = [dropout(step_features, 1-self.dropout, self.input_dim) 
				for step_features in x]

		#Perform gaph convolution
		gcn_step_feas = []

		for step_features in x:
			abs_feas = []
			for i in range(len(self.adj_mas)):
				step_features = tf.transpose(step_features, perm=[0, 2, 1])
				step_features = tf.reshape(step_features, (-1, self.num_nodes))

				gcn_feas = dot(self.adj_mas[i], tf.transpose(step_features), True)
				gcn_feas = tf.transpose(gcn_feas)
				gcn_feas = tf.reshape(gcn_feas, (-1, self.input_dim, self.num_nodes))
				gcn_feas = tf.transpose(gcn_feas, perm=[0, 2, 1])

				gcn_feas = tf.reshape(gcn_feas, (-1, self.input_dim))
				gcn_feas = dot(gcn_feas, self.vars["weights_{}".format(i)])
				gcn_feas = tf.reshape(gcn_feas, (-1, self.num_nodes, self.input_dim))

				gcn_feas = self.act(gcn_feas)

				abs_feas.append(gcn_feas)
			abs_feas = tf.add_n(abs_feas)
			abs_feas = tf.reshape(abs_feas, (-1, 1, self.num_nodes, self.output_dim))
			gcn_step_feas.append(abs_feas)
		
		if len(x) > 1:
			outputs = tf.concat(gcn_step_feas, axis=1)
		else:
			outputs = tf.reshape(gcn_step_feas,
								(-1, self.num_nodes, self.output_dim))
		if self.bias:
			outputs += self.vars["bias"]

		return outputs, self.vars["weights_0"]



