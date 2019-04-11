#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import os


class Brain():

	def __init__(self, gamma=0.99, epsilon=0.1, alpha=0.01, maxMemorySize=100, action_space=5):
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


	def choose_action(self, state, sess, model):
		Q_vector = sess.run(model.q_values, feed_dict={model.input: state})
		
		# Deciding one which action to take
		if np.random.rand() <= self.EPSILON:
			action = np.random.choice(self.action_space)
		else:
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


	def train(self, model, sess):

		memory = self.memory[self.memCntr%self.memSize-1]

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

		if dojo == 0:
			memory[0] = np.delete(memory[0], 2, 0)# Take out the zombie layer
			memory[0] = np.delete(memory[0], 2, 0)# Take out the history layer

			memory[4] = np.delete(memory[4], 2, 0)# Take out the zombie layer
			memory[4] = np.delete(memory[4], 2, 0)# Take out the history layer

		if dojo == 1:
			memory[0] = np.delete(memory[0], 1, 0)# Take out the diamond layer
			memory[0] = np.delete(memory[0], 2, 0)# Take out the history layer

			memory[4] = np.delete(memory[4], 1, 0)# Take out the diamond layer
			memory[4] = np.delete(memory[4], 2, 0)# Take out the history layer
			
		if dojo == 2:
			memory[0] = np.delete(memory[0], 1, 0)# Take out the diamond layer
			memory[0] = np.delete(memory[0], 1, 0)# Take out the zombie layer

			memory[4] = np.delete(memory[4], 1, 0)# Take out the diamond layer
			memory[4] = np.delete(memory[4], 1, 0)# Take out the zombie layer

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
