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
	loss = tf.nn.softmax_cross_entropy_with_logits(
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

def center_loss(embedded_features, labels, centers):
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
	loss  = tf.nn.l2_loss(embedded_features - labels, name="center_loss")
	return loss
