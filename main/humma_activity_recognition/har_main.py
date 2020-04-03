"""
Authors: TamNV
"""
import os
import sys
import tensorflow as tf

sys.path.insert(0, "../../models/human_activity_recognition")
sys.path.insert(0, "../../base")

from bknet1 import BKNet1
from bknet2 import BKNet2
from bknet3 import BKNet3

from utils import *

def do_training(model, data, sess):
	"""
		Perform the training process

		Params:
			model: Substance of Model
			data: Substance of DataManager
			sess: tf.Session()
		Returns:
			None
	"""

def do_predict(model, data, sess):
	"""
		Perform the predicting process

		Params:
			model: Substance of Model
			data
	"""








if __name__ == "__main__":
	file_path = "model_config.csv"
	#get model configuration
	model_config = {}
	model_config[KEY_INPUT_SIZE] = 30
	model_config[KEY_NUM_INPUT_CHANNELS] = 120
	model_config[KEY_NUM_CLASSES] = 10
	model_config[KEY_MODEL_NAME] = "bknet3"

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
	
	else:
		print("Known model type")
		exit(0)
	
	if args.phase == "train":
		do_training(model, data, sess)
	elif args.phase == "test":
		do_predict(model, data, sess)
	else:
		print("KNOWN PHASE!")
	



