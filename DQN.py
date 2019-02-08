'''
Need to make my code more readable and user friendly. Need to be able to change the architecture quickly

'''

import tensorflow as tf
import numpy as np
import sys, os

class Network():
	def __init__(self, ):

		self.LOCAL_GRID_SIZE = 7

		self.n_input_channels = 3

		self.n_out_channels_conv1 = 16
		self.n_out_channels_conv2 = 32
		self.n_out_fc = 256

		self.filter1_size = 3
		self.filter2_size = 3

		self.n_actions = 5


		self.input = tf.placeholder(tf.float32, [self.n_input_channels, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE], name="Input")

		self.conv1 = 1
		self.maxp1 = 2
		self.fc1 = 3

		self.actions = tf.placeholder(tf.float32, [1, self.n_actions], name="Output")

		self.optimiser = None
		self.error = None

		self.weights = None
		self.biases = None






	def create(self, data):


		weights = {'W_conv1':tf.Variable(tf.truncated_normal([3, 3, self.n_input_channels, self.n_out_channels_conv1], mean=0, stddev=1.0, seed=0), name = 'W_conv1'),
			   	   'W_conv2':tf.Variable(tf.truncated_normal([3, 3, self.n_out_channels_conv1, self.n_out_channels_conv2], mean=0, stddev=1.0, seed=1), name = 'W_conv2'),
			   	   'W_fc':tf.Variable(tf.truncated_normal([2*2*self.n_out_channels_conv2, self.n_out_fc], mean=0, stddev=1.0, seed=2), name = 'W_fc'),
			   	   'W_out':tf.Variable(tf.truncated_normal([self.n_out_fc, self.n_actions], mean=0, stddev=1.0, seed=3), name = 'W_out')}

		biases = {'b_conv1':tf.Variable(tf.constant(0.1, shape=[self.n_out_channels_conv1]), name = 'b_conv1'),
			   	  'b_conv2':tf.Variable(tf.constant(0.1, shape=[self.n_out_channels_conv2]), name = 'b_conv2'),
			   	  'b_fc':tf.Variable(tf.constant(0.1, shape=[self.n_out_fc]), name = 'b_fc'),
			   	  'b_out':tf.Variable(tf.constant(0.1, shape=[self.n_actions]), name = 'b_out')}


		x = tf.reshape(data, shape=[-1, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE, self.n_input_channels])

		conv1 = conv2d(x, weights['W_conv1'], name = 'conv1')
		# conv1 = maxpool2d(conv1, name = 'max_pool1')

		conv2 = conv2d(conv1, weights['W_conv2'], name = 'conv2')
		conv2 = maxpool2d(conv2, name = 'max_pool2')

		fc = tf.reshape(conv2,[-1, 2*2*self.n_out_channels_conv2])
		fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

		output = tf.matmul(fc, weights['W_out']) + biases['b_out']
		output = tf.nn.l2_normalize(output)

		return actions, weights, biases



