"""
Authors: TamNV
"""
import os
import csv
import numpy as np
import pandas as pd
import scipy.sparse as sp

def read_seattle_data(data_dir):
	"""
	Read Seattle Dataset from a pickle file
	
	Params:
		data_dir: String
			Path to Settle dataset
	Returns:
		speed_matrix: Pandas Frame
			Volocicy in 2015
		adj_matrix: Numpy Array
			Adjancy Matrix
	"""
	speed_matrix = pd.read_pickle(
						os.path.join(data_dir, "speed_matrix_2015")
					)
	adj_matrix = np.load(
						os.path.join(data_dir, "Loop_Seattle_2015_A.npy")
					)
	return speed_matrix, adj_matrix

def prepare_settle_dataset(speed_matrix, seq_len=10,
						pred_len=1, train_proportion=0.7,
						valid_proportion=0.2):
	"""
	Split Settle dataset into samples and devide 
		there samples into training set & validation set 
											& testing set
	Params:
		speed_matrix: Pandas Frame
		seq_len: Integer
			Input Time Horizon
		pred_len: Integer
			Output Time Horizon
		train_proportion: float
			The proportion of the training set
		valid_proportion: float
			The proportion of the validation set
	Returns:
		train_x, train_y, valid_x, valid_y, test_x, test_y
	"""
	time_length = speed_matrix.shape[0]
	max_speed = speed_matrix.max().max()
	min_speed = speed_matrix.min().min()
	print("Min volocity: {:.3f}, Max volocity: {:.3f}"
			.format(min_speed, max_speed))
	# Normalize data
	speed_matrix = speed_matrix / max_speed

	speed_sequences, speed_labels = [], []

	for idx in range(500):
	# for idx in range(time_length - seq_len - pred_len):
		feats = speed_matrix.iloc[idx:idx+seq_len].values
		labels = speed_matrix.iloc[idx+seq_len:idx+seq_len+pred_len].values
		
		speed_sequences.append(feats)
		speed_labels.append(labels)

	speed_sequences = np.asarray(speed_sequences)
	speed_labels = np.asarray(speed_labels)

	#Shuffle and split the dataset to training and testing dataset
	num_samples = speed_sequences.shape[0]
	idxs = np.arange(num_samples, dtype=int)
	np.random.shuffle(idxs)

	train_idx = int(np.floor(num_samples * train_proportion))
	valid_idx = int(np.floor(num_samples
					* (train_proportion + valid_proportion)))
	train_x = speed_sequences[:train_idx]
	train_y = speed_labels[:train_idx] * max_speed

	valid_x = speed_sequences[train_idx:valid_idx]
	valid_y = speed_labels[train_idx:valid_idx] * max_speed

	test_x = speed_sequences[valid_idx:]
	test_y = speed_labels[valid_idx:] * max_speed

	return train_x, train_y, valid_x, valid_y, test_x, test_y

def get_seattle_dataset(data_dir):
	"""
	Get settle dataset

	Params:
		data_dir: String
			The path to Seattle dataset
	Returns;
		train_x, train_y, valid_x, valid_y, test_x, test_y
	"""
	speed_matrix, adj_matrix = read_seattle_data(data_dir)
	#process adjancy matrix
	# adj_matrix = get_graph_laplacian_spectrum(adj_matrix)
	# adj_matrix = sparse_to_tuple(adj_matrix)

	train_x, train_y, valid_x, valid_y, test_x, test_y = prepare_settle_dataset(speed_matrix,
																				seq_len=10,
																				pred_len=1)
	print("Dataset Information:")
	print("\tTraining Set {} samples".format(train_x.shape[0]))
	print("\tValidation Set {} samples".format(valid_x.shape[0]))
	print("\tTesting Set {} samples".format(test_x.shape[0]))
	
	return adj_matrix, train_x, train_y, valid_x, valid_y, test_x, test_y

def get_next_hop(prev_hop, adj_matrix):
	"""
	caculate next hop

	Params:
		prev_hop: numpy array
		adj_matrix: numpy array
	Returns:
		None
	"""
	next_hop = np.dot(prev_hop, adj_matrix)
	next_hop[next_hop > 1] = 1
	return next_hop

def get_k_hop_matrics(adj_matrix, K):
	"""
	get k hop matrics

	Params:
		adj_matrix: numpy array
		K: number hop we want to consider
	Returns:
		hops: List
	"""
	hops = [adj_matrix]
	for i in range(0, K-1):
		next_hop = get_next_hop(hops[-1], adj_matrix)
		hops.append(next_hop)
	return hops

# def sparse_to_tuple(adj_ma):
# 	"""
# 	Convert Sparse Matrix A to tuple presentation(indexs, values, shape)

# 	params:
# 		adj_ma: Sparse Matric
# 	Returns:
# 		coords: list of index
# 		values: float
# 		shape: list
# 	"""
# 	def to_tuple(mx):
# 		if not sp.isspmatrix_coo(mx):
# 			mx = mx.tocoo()
# 		coords = np.vstack((mx.row, mx.col)).transpose()
# 		values = mx.data
# 		shape = mx.shape

# 		return coords, values, shape
# 	if isinstance(adj_ma, list):
# 		for i in range(len(adj_ma)):
# 			adj_ma[i] = to_tuple(adj_ma[i])
# 	else:
# 		adj_ma = to_tuple(adj_ma)
# 	return adj_ma


# def get_graph_laplacian_spectrum(adj_ma):
# 	"""
# 	Caculate K follow the formulation

# 	Params:
# 		adj_ma: Numpy Array
# 			Adjcancy matric
# 	Returns:
# 		K: Numpy Array
# 	"""
# 	num_vers = adj_ma.shape[0]
# 	_A = adj_ma + np.eye(num_vers)
# 	diag_D = np.sum(_A, axis=0)
# 	diag_D = np.power(diag_D, -0.5)

# 	_D = np.eye(num_vers)*diag_D

# 	K = np.dot(np.dot(_D, _A),_D)
# 	K = sp.csr_matrix(K)

# 	return K


# def sparse2dense(edges):
# 	"""
# 	convert sparse to dense matrix

# 	params:
# 		list: list of edges
# 	Returns:
# 		Adjancency matrix
# 	"""
# 	num_nodes = np.unique(np.array(edges).reshape(-1)).shape[0]
# 	adj_matrix = np.zeros((num_nodes, num_nodes))
# 	for (v1, v2) in edges:
# 		adj_matrix[v1-1, v2-1] = 1
# 		adj_matrix[v2-1, v1-1] = 1
# 		adj_matrix[v1-1, v1-1] = 1
# 		adj_matrix[v2-1, v2-1] = 1
# 	return adj_matrix

# if __name__ == "__main__":
# 	K = 3
# 	edges = [
# 				[1, 2],
# 				[2, 3],
# 				[2, 4],
# 				[2, 5],
# 				[6, 7],
# 				[7, 8]
# 			]
# 	adj_matrix = sparse2dense(edges)
# 	hops = get_k_hop_matrics(adj_matrix, K)
# 	for i in range(K):
# 		print("HOP {}".format(i+1))
# 		print(hops[i])