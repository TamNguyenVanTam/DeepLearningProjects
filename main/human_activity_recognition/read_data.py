#Authors: TamNV
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def normalize_data(x_train, x_valid, x_test):
	num_trains, num_valids, num_tests = x_train.shape[0], x_valid.shape[0], x_test.shape[0] 
	num_time_steps, num_feas = x_train.shape[1], x_train.shape[2]
	scaler = MinMaxScaler(feature_range=(0, 1))

	x_train = np.reshape(x_train, (num_trains * num_time_steps, num_feas))
	x_valid = np.reshape(x_valid, (num_valids * num_time_steps, num_feas))
	x_test = np.reshape(x_test, (num_tests * num_time_steps, num_feas))

	x_train = scaler.fit_transform(x_train)
	x_valid = scaler.transform(x_valid)
	x_test = scaler.transform(x_test)

	x_train = np.reshape(x_train, (num_trains, num_time_steps, num_feas))
	x_valid = np.reshape(x_valid, (num_valids, num_time_steps, num_feas))
	x_test = np.reshape(x_test, (num_tests, num_time_steps, num_feas))

	return x_train, x_valid, x_test

def convert2onehot(y, num_classes):
	"""
	Convert y to onehot form

	params:
		y: numpy array 
			labels
		num_classes: Integer
			number of classes
	returns:
		One-hoted label 
	"""
	y = np.reshape(y, -1)
	num_sams = y.shape[0]
	_y = np.zeros((num_sams, num_classes), np.int32)
	
	for i in range(num_sams):
		_y[i, y[i]] = 1

	return _y


def get_smartphone_dataset():
	"""
	Read pickle file

	Params:
		None
	Returns:
		x_train, y_train, x_test, y_test
	"""
	file = open("../../data/human_activity_recognition/SmartPhoneDataset", "rb")
	(x_train, y_train), (x_test, y_test) = pickle.load(file)
	file.close()
	
	x_train, x_valid, x_test = normalize_data(x_train, x_test, x_test)
	num_classes = np.unique(y_train).shape[0]
	
	for _class in range(num_classes):
		print("Class {} proportion {:.3f}".format(_class, np.mean(y_train==_class)))

	y_train = convert2onehot(y_train, num_classes)
	y_valid = convert2onehot(y_test, num_classes)
	y_test = convert2onehot(y_test, num_classes)

	return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == "__main__":
	get_smartphone_dataset()
