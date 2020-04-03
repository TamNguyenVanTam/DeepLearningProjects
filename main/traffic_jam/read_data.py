"""
Authors: TamNV
"""
import os
import csv
import numpy as np
import pandas as pd

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

	for idx in range(time_length - seq_len - pred_len):
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
	train_y = speed_labels[:train_idx]

	valid_x = speed_sequences[train_idx:valid_idx]
	valid_y = speed_labels[train_idx:valid_idx]

	test_x = speed_sequences[valid_idx:]
	test_y = speed_labels[valid_idx:]			

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
	train_x, train_y,
	valid_x, valid_y,
	test_x, test_y = prepare_settle_dataset(speed_matrix,
											seq_len=10,
											pred_len=1)
	return adj_matrix, train_x, train_y,
			valid_x, valid_y, test_x, test_y


