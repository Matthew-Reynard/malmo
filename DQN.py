'''
Need to make my code more readable and user friendly. Need to be able to change the architecture quickly

'''

import tensorflow as tf
import numpy as np
import sys, os

class Network():

	def __init__(self, name="test_model", load=False):

		self.LOCAL_GRID_SIZE = 9

		self.n_input_channels = 3

		self.n_out_channels_conv1 = 16
		self.n_out_channels_conv2 = 32
		self.n_out_fc = 256

		self.filter1_size = 3
		self.filter2_size = 3

		self.n_actions = 5

		# input
		self.input = tf.placeholder(tf.float32, [self.n_input_channels, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE], name="Input")

		# output
		self.actions = tf.placeholder(tf.float32, [1, self.n_actions], name="Output")

		self.q_values = None
		self.optimiser = None
		self.error = None
		self.y_prime_max = None
		self.action_t = None

		self.weights = None
		self.biases = None

		self.path = "./Models/Tensorflow/"
		self.name = name
		self.load = load


	# 2D convolution
	def conv2d(self, x, W, name = None):

		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding ="VALID", name = name)


	# Max pooling
	# strides changed from 2 to 1
	def maxpool2d(self, x, name = None):

		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID', name = name)


	def create(self, data):

		if self.load:

			path = self.path + self.name + ".npz"

			model = np.load(path)

			weights = {'W_conv1':tf.Variable(model["W_conv1"], name = 'W_conv1'),
			   	       'W_conv2':tf.Variable(model["W_conv2"], name = 'W_conv2'),
			   	       'W_fc':tf.Variable(model["W_fc"], name = 'W_fc'),
			   	   	   'W_out':tf.Variable(model["W_out"], name = 'W_out')}

			biases = {'b_conv1':tf.Variable(model["b_conv1"], name = 'b_conv1'),
				   	  'b_conv2':tf.Variable(model["b_conv2"], name = 'b_conv2'),
				   	  'b_fc':tf.Variable(model["b_fc"], name = 'b_fc'),
				   	  'b_out':tf.Variable(model["b_out"], name = 'b_out')}

		else:

			weights = {'W_conv1':tf.Variable(tf.truncated_normal([3, 3, self.n_input_channels, self.n_out_channels_conv1], mean=0, stddev=1.0, seed=0), name = 'W_conv1'),
				   	   'W_conv2':tf.Variable(tf.truncated_normal([3, 3, self.n_out_channels_conv1, self.n_out_channels_conv2], mean=0, stddev=1.0, seed=1), name = 'W_conv2'),
				   	   'W_fc':tf.Variable(tf.truncated_normal([4*4*self.n_out_channels_conv2, self.n_out_fc], mean=0, stddev=1.0, seed=2), name = 'W_fc'),
				   	   'W_out':tf.Variable(tf.truncated_normal([self.n_out_fc, self.n_actions], mean=0, stddev=1.0, seed=3), name = 'W_out')}

			biases = {'b_conv1':tf.Variable(tf.constant(0.1, shape=[self.n_out_channels_conv1]), name = 'b_conv1'),
				   	  'b_conv2':tf.Variable(tf.constant(0.1, shape=[self.n_out_channels_conv2]), name = 'b_conv2'),
				   	  'b_fc':tf.Variable(tf.constant(0.1, shape=[self.n_out_fc]), name = 'b_fc'),
				   	  'b_out':tf.Variable(tf.constant(0.1, shape=[self.n_actions]), name = 'b_out')}


		x = tf.reshape(data, shape=[-1, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE, self.n_input_channels])

		conv1 = self.conv2d(x, weights['W_conv1'], name = 'conv1')
		# conv1 = maxpool2d(conv1, name = 'max_pool1')

		conv2 = self.conv2d(conv1, weights['W_conv2'], name = 'conv2')
		conv2 = self.maxpool2d(conv2, name = 'max_pool2')

		fc = tf.reshape(conv2,[-1, 4*4*self.n_out_channels_conv2])
		fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

		actions = tf.matmul(fc, weights['W_out']) + biases['b_out']
		actions = tf.nn.l2_normalize(actions)

		return actions, weights, biases


	def setup(self, brain):

		with tf.name_scope('Model'):
			self.q_values, self.weights, self.biases = self.create(self.input)

		# Error / Loss function 
		# reduce_max -> it reduces the [1,4] tensor to a scalar of the max value
		with tf.name_scope('Error'):
			self.error = tf.losses.mean_squared_error(labels=self.q_values, predictions=self.actions)

		# Gradient descent optimizer - minimizes error/loss function
		with tf.name_scope('Optimizer'):
			self.optimizer = tf.train.GradientDescentOptimizer(brain.ALPHA).minimize(self.error)
			# optimizer = tf.train.AdamOptimizer(alpha).minimize(error)

		# The next states action-value [1,4] tensor, reduced to a scalar of the max value
		with tf.name_scope('Max_y_prime'):
			self.y_prime_max = tf.reduce_max(self.actions, axis=1)

		# Action at time t, the index of the max value in the action-value tensor (Made a global variable)
		with tf.name_scope('Max_action'):
			self.action_t = tf.argmax(self.actions, axis=1)


	def save(self, sess, verbose = False):

		path = self.path + self.name + ".npz"
		
		try:
			os.makedirs('./Models/Tensorflow/', exist_ok=True)
		except Exception as e:
			print(e,'Could not create directory ./Models/Tensorflow')

		w1 = np.array(sess.run(self.weights['W_conv1']))
		w2 = np.array(sess.run(self.weights['W_conv2']))
		w3 = np.array(sess.run(self.weights['W_fc']))
		w4 = np.array(sess.run(self.weights['W_out']))

		b1 = np.array(sess.run(self.biases['b_conv1']))
		b2 = np.array(sess.run(self.biases['b_conv2']))
		b3 = np.array(sess.run(self.biases['b_fc']))
		b4 = np.array(sess.run(self.biases['b_out']))

		np.savez(path, W_conv1=w1, W_conv2 = w2, W_fc=w3, W_out = w4,
					   b_conv1=b1, b_conv2 = b2, b_fc=b3, b_out = b4)
		
		if verbose: print("\nModel saved in:", path)






