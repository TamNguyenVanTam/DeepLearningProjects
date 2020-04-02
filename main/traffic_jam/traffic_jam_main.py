"""
Authors: TamNV
This file implements traffic jam main
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, "../../models/traffic_jam")
sys.path.insert(0, "../../base")

from traffic_jam_lstm import LSTM1
from utils import *



if __name__ == "__main__":
	file_path = "model_config.csv"
	# read file and get model config file
	model_config = {}
	model_config[KEY_NUM_SECS] = 20
	model_config[KEY_NUM_IN_TIME_STEPS] = 10
	model_config[KEY_NUM_OUT_TIME_STEPS] = 1
	
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
		"dropout": tf.placeholder(tf.float32),
		"learning_rate": tf.placeholder(tf.float32),
		"weight_decay": tf.placeholder(tf.float32)
	}

	model = LSTM1(placeholders, model_config)
	

