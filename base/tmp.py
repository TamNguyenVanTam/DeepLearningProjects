import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
import tensorflow as tf

from layers import *


num_classes = 10
num_feas = 2
learning_rate = 1e-4

inputs = tf.placeholder(tf.float32, shape=(None, 2))
labels = tf.placeholder(tf.int32, shape=(None, num_classes))

centerLayer = CenterLoss(num_classes=num_classes,
						num_feas=num_feas,
						learning_rate=learning_rate)
# print(inputs)
# print(labels) 
centerLayer([inputs, labels])


# outputs = max1D(inputs)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# b = sess.run(outputs, feed_dict={inputs:A})
# print(A)
# print(b)


# conv2D = Conv2D(num_in_channels=1,
# 				num_out_channels=10,
# 				filter_size=3,
# 				strides=[1, 1, 1, 1],
# 				padding="SAME",
# 				dropout=0.5,
# 				bias=True,
# 				act=tf.nn.relu)

# outputs = conv2D(inputs)
# print(outputs)

# inputs = tf.placeholder(tf.float32, shape=(None, 20, 1))

# conv1D = Conv1D(num_in_channels=1,
# 				num_out_channels=10,
# 				filter_size=3,
# 				strides=1,
# 				padding="VALID",
# 				dropout=0.5,
# 				bias=True,
# 				act=tf.nn.relu)

# outputs = conv1D(inputs)
# print(outputs)


















# def sparse_to_tuple(sparse_mx):
# 	### Convert sparse matrix to tuple representation ###
# 	def to_tuple(mx):
# 		if not sp.isspmatrix_coo(mx):
# 			mx = mx.tocoo()
# 		coords = np.vstack((mx.row, mx.col)).transpose()
# 		values = mx.data
# 		shape = mx.shape

# 		return coords, values, shape

# 	if isinstance(sparse_mx, list):
# 		for i in range(len(sparse_mx)):
# 			sparse_mx[i] = to_tuple(sparse_mx[i])
# 	else:
# 		sparse_mx = to_tuple(sparse_mx)

# 	return sparse_mx

# def create_adj_matric():
# 	adj = np.zeros((5, 5))
# 	edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
# 	for v1, v2 in edges:
# 		adj[v1, v2] = 1
# 		adj[v2, v1] = 1
# 	adj += np.eye(5)
# 	sp_ma = coo_matrix(adj)
# 	sp_ma = sparse_to_tuple(sp_ma)
# 	return sp_ma

# sp_ma = [create_adj_matric()]

# input_features = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]).T
# input_features = np.reshape(input_features, (1, 2, 5, 1))
# input_features = np.concatenate([input_features, input_features], axis=0)

# # # #test Graph convolution
# input_dim, N_CLASSES, batch_size, n_epochs = 1, 10, 128, 30

# placeholders ={
# 	"adj_mas": [tf.sparse_placeholder(tf.float32)],
# 	"features": tf.placeholder(tf.float32, shape=(None, 2, 5, 1)),
# 	"labels": tf.placeholder(tf.int32, shape=(None, N_CLASSES)),
# 	"dropout": tf.placeholder(tf.float32),
# 	"learning_rate": tf.placeholder(tf.float32),
# 	"weight_decay": tf.placeholder(tf.float32),
# }

# gcn1 = GraphConv(input_dim=input_dim,
# 				output_dim=1,
# 				adj_matrics=placeholders["adj_mas"],
# 				num_nodes=5,
# 				dropout=1.0,
# 				act=leak_relu,
# 				bias=True)

# outputs, weights = gcn1(placeholders["features"])

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# feed_dict = {}
# feed_dict.update({placeholders['adj_mas'][i] : sp_ma[i] for i in range(len(sp_ma))})
# feed_dict.update({placeholders["features"]: input_features})

# _outputs, weights = sess.run([outputs, weights], feed_dict=feed_dict)
# print(_outputs.shape)
# print(_outputs[0, 0, :, :])
# print(_outputs[0, 1, :, :])
# print(weights)