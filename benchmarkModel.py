
import constants as c

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


class FeedForward:
	def __init__(self, num_hidden):
		self.num_hidden = num_hidden

		self.sess = tf.Session()
		self.defineGraph()
		self.sess.run(tf.global_variables_initializer())

	def defineGraph(self):
		self.initializer = tf.random_normal_initializer(stddev=0.1)
		self.img_batch_in = tf.placeholder(tf.float32, shape=[c.BATCH_SZ, c.IMG_VEC_SZ]) 
		self.labels = tf.placeholder(tf.int64, shape=[c.BATCH_SZ])


		weights_in = tf.get_variable("W_IN", dtype=tf.float32, shape=[c.IMG_VEC_SZ, self.num_hidden], initializer=self.initializer)
		biases_in = tf.get_variable("B_IN", dtype=tf.float32, shape=[self.num_hidden], initializer=self.initializer)

		hidden = tf.nn.relu(tf.matmul(self.img_batch_in, weights_in) + biases_in)

		weights_out = tf.get_variable("W_OUT", dtype=tf.float32, shape=[self.num_hidden, 10], initializer=self.initializer)
		biases_out = tf.get_variable("B_OUT", dtype=tf.float32, shape=[10], initializer=self.initializer)
		logits = tf.matmul(hidden, weights_out) +  biases_out


		self.probabilities = tf.nn.softmax(logits)

		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))

		self.trainOp = tf.train.AdamOptimizer(learning_rate=c.LEARN_RATE).minimize(self.loss)

		self.preds = tf.argmax(self.probabilities, axis=1)

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.labels), tf.float32))


	def train(self, trainImagesLabelsTup):
		for epoch in range(c.NUM_EPOCHS):

			print("\nTrain BENCHMARK MODEL Batch (Epoch):", epoch)

			feedDict = {self.img_batch_in: trainImagesLabelsTup[0], self.labels: trainImagesLabelsTup[1]}
			sessArgs = [self.accuracy, self.loss, self.trainOp]

			# RUN #
			acc, lossReturned, _ = self.sess.run(sessArgs, feed_dict=feedDict)

			print("loss -", lossReturned)
			print("accuracy -", acc)



	def test(self, testImagesLabelsTup):
		feedDict = {self.img_batch_in: testImagesLabelsTup[0], self.labels: testImagesLabelsTup[1]}
		sessArgs = [self.accuracy, self.loss]

		# RUN #
		acc, lossReturned = self.sess.run(sessArgs, feed_dict=feedDict)

		print("TEST BENCHMARK loss -", lossReturned)
		print("TEST BENCHMARK accuracy -", acc)
		return acc




