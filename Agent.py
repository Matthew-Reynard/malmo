#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import os


class Brain():

	def __init__(self, gamma=0.99, epsilon=0.1, alpha=0.005, maxMemorySize=100, action_space=5):
		self.ALPHA = alpha
		self.GAMMA = gamma
		self.EPSILON = epsilon
		self.memory = []
		self.memSize = maxMemorySize
		self.memCntr = 0
		self.steps = 0
		self.action_space = np.arange(action_space)


	def store_transition(self, state, action, reward, terminal, state_):
		if self.memCntr < self.memSize:
			self.memory.append([state, action, reward, terminal, state_])
		else:
			self.memory[self.memCntr%self.memSize] = [state, action, reward, terminal, state_]
		self.memCntr += 1

		# print(self.memory[self.memCntr%self.memSize])


	def store_transition_dojo(self, state, action, reward, terminal, state_, dojo):
		if self.memCntr < self.memSize:
			self.memory.append([state, action, reward, terminal, state_, dojo])
		else:
			self.memory[self.memCntr%self.memSize] = [state, action, reward, terminal, state_, dojo]
		self.memCntr += 1

		# print(self.memory[self.memCntr%self.memSize])


	def choose_action(self, state, sess, model):
				
		# Deciding one which action to take
		if np.random.rand() <= self.EPSILON:
			action = np.random.choice(self.action_space)
		else:
			Q_vector = sess.run(model.q_values, feed_dict={model.input: state})
			# print(Q_vector)
			action = sess.run(model.action_t, feed_dict={model.actions: Q_vector})
			action = action.item()
		self.steps += 1

		return action


	def choose_dojo(self, state, sess, model, action_space, epsilon):
		Q_vector = sess.run(model.q_values, feed_dict={model.input: state})
		
		# Deciding one which action to take
		if np.random.rand() <= epsilon:
			action = np.random.choice(action_space)
		else:
			action = sess.run(model.action_t, feed_dict={model.actions: Q_vector})
		
		# self.steps += 1

		return action


	def linear_epsilon_decay(self, total, episode, start=0.5, end=0.05, percentage=0.5):
		
		self.EPSILON = (-(start-end)/ (percentage*total)) * episode + (start)
		
		if self.EPSILON < end: 
			self.EPSILON = end


	def linear_aplha_decay(self, total, episode, start=0.03, end=0.01, percentage=0.3):
		
		self.ALPHA = (-(start-end)/ (percentage*total)) * episode + (start)
		
		if self.ALPHA < end: 
			self.ALPHA = end


	def train_batch(self, batch_size, model, sess):

		mini_batch_indicies = np.random.choice(self.memSize, batch_size, replace=False)
		# print(mini_batch)

		# creating a mini batch
		mini_batch = []
		for i in mini_batch_indicies:
			mini_batch.append(self.memory[i])

		for memory in mini_batch:
			output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

			if memory[3]:
				output_vector[:,memory[1]] = memory[2]
				# print("Reward:", reward)
			else:
				# Gathering the now current state's action-value vector
				y_prime = sess.run(model.q_values, feed_dict={model.input: memory[4]})

				# Equation for training
				maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

				# RL Equation
				output_vector[:,memory[1]] = memory[2] + (self.GAMMA * maxq)

			_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		return e, output_vector


	# Train with fixed target
	def train_batch_fixed_target(self, batch_size, model, target, sess):

		mini_batch_indicies = np.random.choice(self.memSize, batch_size, replace=False)
		# print(mini_batch)

		# creating a mini batch
		mini_batch = []
		for i in mini_batch_indicies:
			mini_batch.append(self.memory[i])

		for memory in mini_batch:
			output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

			if memory[3]:
				output_vector[:,memory[1]] = memory[2]
				# print("Reward:", reward)
			else:
				# Gathering the now current state's action-value vector
				y_prime = sess.run(target.q_values, feed_dict={target.input: memory[4]})

				# Equation for training
				maxq = sess.run(target.y_prime_max, feed_dict={target.actions: y_prime})

				# RL Equation
				output_vector[:,memory[1]] = memory[2] + (self.GAMMA * maxq)

			_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		return e, output_vector


	# Train with double DQN
	def train_batch_double_DQN(self, batch_size, model, target, sess):

		mini_batch_indicies = np.random.choice(self.memSize, batch_size, replace=False)
		# print(mini_batch)

		# creating a mini batch
		mini_batch = []
		for i in mini_batch_indicies:
			mini_batch.append(self.memory[i])

		for memory in mini_batch:
			output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

			if memory[3]:
				output_vector[:,memory[1]] = memory[2]
				# print("Reward:", reward)
			else:
				# Gathering the now current state's action-value vector
				y_prime = sess.run(model.q_values, feed_dict={model.input: memory[4]})

				# Equation for training
				argmaxq = sess.run(model.y_prime_argmax, feed_dict={model.actions: y_prime})

				# Gathering the now current state's action-value vector
				target_y_prime = sess.run(target.q_values, feed_dict={target.input: memory[4]})

				argmax_action_q = target_y_prime[:,argmaxq]

				# RL Equation
				output_vector[:,memory[1]] = memory[2] + (self.GAMMA * argmax_action_q)

			_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		return e, output_vector


	def train(self, model, sess):

		memory = self.memory[self.memCntr%self.memSize-1]

		output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

		# print(output_vector)

		if memory[3]:
			output_vector[:,memory[1]] = memory[2]
		else:
			# Gathering the now current state's action-value vector
			y_prime = sess.run(model.q_values, feed_dict={model.input: memory[4]})

			# Equation for training
			maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

			# RL (Bellman) Equation
			output_vector[:,memory[1]] = memory[2] + (self.GAMMA * maxq)

		_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		# print(output_vector)

		return e, output_vector


	def train_2_dojos(self, model, sess, dojo):

		memory = self.memory[self.memCntr%self.memSize-1]

		if dojo == 0:
			memory[0] = np.delete(memory[0], 2, 0)# Take out the zombie layer
			memory[4] = np.delete(memory[4], 2, 0)# Take out the zombie layer
		if dojo == 1:
			memory[0] = np.delete(memory[0], 1, 0)# Take out the diamond layer
			memory[4] = np.delete(memory[4], 1, 0)# Take out the diamond layer

		output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

		if memory[3]:
			output_vector[:,memory[1]] = memory[2]
			# print("Reward:", reward)
		else:
			# Gathering the now current state's action-value vector
			y_prime = sess.run(model.q_values, feed_dict={model.input: memory[4]})

			# Equation for training
			maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

			# RL Equation
			output_vector[:,memory[1]] = memory[2] + (self.GAMMA * maxq)

		_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		return e, output_vector


	def train_3_dojos(self, model, sess, dojo):

		memory = self.memory[self.memCntr%self.memSize-1]

		# if dojo == 0:
		# 	memory[0] = np.delete(memory[0], 2, 0)# Take out the zombie layer
		# 	memory[0] = np.delete(memory[0], 2, 0)# Take out the history layer

		# 	memory[4] = np.delete(memory[4], 2, 0)# Take out the zombie layer
		# 	memory[4] = np.delete(memory[4], 2, 0)# Take out the history layer

		# if dojo == 1:
		# 	memory[0] = np.delete(memory[0], 1, 0)# Take out the diamond layer
		# 	memory[0] = np.delete(memory[0], 2, 0)# Take out the history layer

		# 	memory[4] = np.delete(memory[4], 1, 0)# Take out the diamond layer
		# 	memory[4] = np.delete(memory[4], 2, 0)# Take out the history layer
			
		# if dojo == 2:
		# 	memory[0] = np.delete(memory[0], 1, 0)# Take out the diamond layer
		# 	memory[0] = np.delete(memory[0], 1, 0)# Take out the zombie layer

		# 	memory[4] = np.delete(memory[4], 1, 0)# Take out the diamond layer
		# 	memory[4] = np.delete(memory[4], 1, 0)# Take out the zombie layer

		output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

		if memory[3]:
			output_vector[:,memory[1]] = memory[2]
			# print("Reward:", reward)
		else:
			# Gathering the now current state's action-value vector
			y_prime = sess.run(model.q_values, feed_dict={model.input: memory[4]})

			# Equation for training
			maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

			# RL Equation
			output_vector[:,memory[1]] = memory[2] + (self.GAMMA * maxq)

		_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		return e, output_vector


	def train_3(self, sess, diamond, zombie, explore):

		memory = self.memory[self.memCntr%self.memSize-1]

		if memory[5] == 0:
			model = diamond
			# memory[0][2] = 0
			# memory[0][3] = 0
			# memory[4][2] = 0
			# memory[4][3] = 0

		if memory[5] == 1:
			model = zombie
			# memory[0][1] = 0
			# memory[0][3] = 0
			# memory[4][1] = 0
			# memory[4][3] = 0
			
		if memory[5] == 2:
			model = explore
			# memory[0][1] = 0
			# memory[0][2] = 0
			# memory[4][1] = 0
			# memory[4][2] = 0

		output_vector = sess.run(model.q_values, feed_dict={model.input: memory[0]})

		if memory[3]:
			output_vector[:,memory[1]] = memory[2]
			# print("Reward:", reward)
		else:
			# Gathering the now current state's action-value vector
			y_prime = sess.run(model.q_values, feed_dict={model.input: memory[4]})

			# Equation for training
			maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

			# RL Equation
			output_vector[:,memory[1]] = memory[2] + (self.GAMMA * maxq)

		_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: memory[0], model.actions: output_vector})

		return e, output_vector


	# Used for DDQN
	def update_target_graph(self):
	
		# Get the parameters of our DQNNetwork
		from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Model")

		# Get the parameters of our Target_network
		to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Target")

		op_holder = []

		# Update our target_network parameters with DQNNetwork parameters
		for from_var,to_var in zip(from_vars,to_vars):
			op_holder.append(to_var.assign(from_var))

		return op_holder
