"""
Authors: TamNV
This file implements traffic jam main
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, "../../models/traffic_jam")
sys.path.insert(0, "../../base")

from traffic_jam_lstm import LSTM1
from traffic_jam_gcn_lstmv1 import GraphCNNLSTM1
from read_data import get_seattle_dataset, get_k_hop_matrics
from abstract_data import DataManager
from traffic_jam_trainer import TrafficJamTrainer
from traffic_jam_predictor import TrafficJamPredictor
from utils import *

def read_json_file(filename):
	with open(filename) as file:
		contents = json.load(file)
	if contents == None:
		print("Meeting wrong went read {}".format(filename))
		exit(0)
	return contents

def do_training(model, data, sess, filters, exe_config):
	"""
	Perform training

	Params:
		model: Model instance
		data: DataManager instance
		sess: tf.Session()
		exe_config: Dictionary
	Returns:
		None
	"""
	trainer = TrafficJamTrainer(filters, exe_config)
	trainer.train(model, data, sess)

def do_predicting(model, data, sess, filters, exe_config):
	"""
	Perform predicting

	Params:
		model: model instance
		data: dat manager instance
		sess: tf.Session()
		exe_config: dictionary
	Returns:
		none
	"""
	predictor = TrafficJamPredictor(filters, exe_config)
	predictor.predict(model, data, sess)
	trained_filters = predictor.get_impact_between_nodes()

	for trained_filter in trained_filters:
		plt.imshow(trained_filter, "gray")
		plt.show()
import argparse
parser = argparse.ArgumentParser(description="arguments of traffic jam project")
parser.add_argument("--model_config", dest="model_config", default="configuration/lstm_model_config.json")
parser.add_argument("--exe_config", dest="exe_config", default="configuration/lstm_exe_config.json")
parser.add_argument("--phase", dest="phase", default="predict")
args = parser.parse_args()

if __name__ == "__main__":
	model_config = read_json_file(args.model_config)
	exe_config = read_json_file(args.exe_config)

	adj_matrix, train_x, train_y, valid_x, valid_y, test_x, test_y = get_seattle_dataset(exe_config[KEY_DATADIR])
	adj_matrix = adj_matrix.astype(np.float32)
	filters = get_k_hop_matrics(adj_matrix, model_config[KEY_NUM_HOPS])

	placeholders = {
		"features": tf.placeholder(tf.float32,
						shape=(
								None,
								model_config[KEY_NUM_IN_TIME_STEPS],
								model_config[KEY_NUM_SECS]
						)),
		"labels": tf.placeholder(tf.float32,
						shape=(
								None,
								model_config[KEY_NUM_OUT_TIME_STEPS],
								model_config[KEY_NUM_SECS]
					)),
		"filters": [tf.placeholder(tf.float32,
						shape=(model_config[KEY_NUM_SECS], model_config[KEY_NUM_SECS])
					) for _ in range(len(filters))],
		"dropout": tf.placeholder(tf.float32),
		"learning_rate": tf.placeholder(tf.float32),
		"weight_decay": tf.placeholder(tf.float32)
	}

	model = GraphCNNLSTM1(placeholders, model_config)

	
	sess = tf.Session()

	data = DataManager(train_x, train_y,
						valid_x, valid_y,
						test_x, test_y)

	if args.phase == "train":
		#perform training
		do_training(model, data, sess, filters, exe_config)
	elif args.phase == "predict":
		#perform testing
		do_predicting(model, data, sess, filters, exe_config)
