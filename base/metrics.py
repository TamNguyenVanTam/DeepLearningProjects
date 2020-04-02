import tensorflow as tf

def get_softmax_cross_entropy(preds, labels):
	"""
	Caculate softmax cross entropy loss

	params:
		preds: Tensor Object
			Predictions
		labels: Tensor Object
			Labels
	Returns: Tensor object (Cross entropy loss) 
	"""
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(
						logits=preds, labels=labels)
	return tf.reduce_mean(loss)

def get_accuracy(preds, labels):
	"""
	Caculate accuracy

	params:
		preds: Tensor Object
			Predictions
		labels: Tensor Object
			Labels
	Returns: Tensor object (Accuracy) 
	"""
	correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
	accuracy_all = tf.cast(correct_prediction, tf.float32)

	return tf.reduce_mean(accuracy_all)

def get_center_loss(embedded_features, labels, centers):
	"""
	Caculate center loss
	Params:
		embedded_features: Tensor object, BatchSize x N_Features
			Embedding of each sample
		labels: Tensor object, BatchSize x N_Classes
		
		Center: Tensor object, N_classes x N_Features
	Returns:
		Center loss
	"""
	_labels = tf.cast(labels, tf.float32)
	_embedded_features = tf.matmul(_labels, centers)
	loss  = tf.nn.l2_loss(embedded_features - _embedded_features,
							name="center_loss")
	return loss

def get_rmse_loss(labels, preds):
	"""
	Caculate RMSE loss
	
	Params:
		labels: Tensor Object
			Ground truth
		preds: Tensor Object
			Traffic Velocity Prediction 
	Returns:
		Root Mean Square Loss
	"""
	diff = labels - preds
	diff = tf.reduce_mean(diff ** 2)
	return tf.math.sqrt(diff)

def get_mape(labels, preds):
	"""
	Caculate MAPE

	Params:
		labels: Tensor Object
			Ground truth
		preds: Tensor Object
			Labels
	Returns
		Mean Absolute Percentage Error(MAPE)
	"""
	diff = labels - preds
	diff = tf.math.abs(diff / (labels + 1e-8))
	return 100 * tf.reduce_mean(diff)

def get_mae(labels, preds):
	"""
	Caculate MAE

	Params:
		labels: Tensor Object
			Ground truth
		preds: Tensor Object
			Labels
	Returns:
		Mean absolute error
	"""
	diff = labels - preds
	diff = tf.math.abs(diff)
	return tf.reduce_mean(diff)
