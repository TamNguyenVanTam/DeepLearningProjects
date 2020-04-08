"""
Authors: TamNV
"""
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, "../../models/human_activity_recognition")
sys.path.insert(0, "../../base")

from bknet1 import BKNet1
from bknet2 import BKNet2
from bknet3 import BKNet3
from bknet3_center_loss import BKNet3CenterLoss

from utils import *
from read_data import get_smartphone_dataset

from har_trainer_center_loss import HARCenterLossTrainer
from abstract_data import DataManager

def do_training(model, data, sess, exe_config):
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
	sess.run(tf.global_variables_initializer())
	trainer = HARCenterLossTrainer(exe_config)
	trainer.train(model, data, sess)

def do_predict(model, data, sess):
	"""
		Perform the predicting process

		Params:
			model: Substance of Model
			data
	"""
	pass

if __name__ == "__main__":
	# file_path = "model_config.csv"
	# #get model configuration
	model_config, exe_config = {}, {}
	model_config[KEY_INPUT_SIZE] = 128
	model_config[KEY_NUM_INPUT_CHANNELS] = 9
	model_config[KEY_NUM_CLASSES] = 6
	model_config[KEY_MODEL_NAME] = "bknet3_centerloss"


	exe_config["batch_size"] = 16
	exe_config["num_epochs"] = 300
	exe_config["learning_rate"] = 1e-3
	exe_config["dropout"] = 0.0
	exe_config["decay_factor"] = 5*1e-4
	exe_config["checkpoint"] = "../../checkpoint/har"
	exe_config["data_dir"] = "../../data/traffic_jam"


	x_train, y_train, x_valid, y_valid, x_test, y_test = get_smartphone_dataset()
	
	placeholders = {
		"features": tf.placeholder(tf.float32,
						shape=(
								None,
								model_config[KEY_INPUT_SIZE],
								model_config[KEY_NUM_INPUT_CHANNELS]
						)),
		"labels": tf.placeholder(tf.float32,
						shape=(
								None,
								model_config[KEY_NUM_CLASSES]
						)),
		"dropout": tf.placeholder(tf.float32),
		"learning_rate": tf.placeholder(tf.float32),
		"weight_decay": tf.placeholder(tf.float32)
	}


	model_name = model_config[KEY_MODEL_NAME]
	model = None
	
	if model_name == "bknet1":
		model = BKNet1(placeholders)
	
	elif model_name == "bknet2":
		model = BKNet2(placeholders)
	
	elif model_name == "bknet3":
		model = BKNet3(placeholders)
	
	elif model_name == "bknet3_centerloss":
		model = BKNet3CenterLoss(placeholders)
	else:
		print("Known model type")
		exit(0)
	
	sess = tf.Session()
	data = DataManager(x_train, y_train, x_valid, y_valid, x_test, y_test)

	do_training(model, data, sess, exe_config)
	



