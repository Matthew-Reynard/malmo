#!/usr/bin/python3

# Imports
import numpy as np
import tensorflow as tf
import time
import math

# Custom imports
from DQN import Network
from Agent import Brain
from Malmo_Environment import Environment
from utils import print_readable_time

MODEL_PATH_SAVE = "./Models/Tensorflow/model_0.ckpt"

LOGDIR = "./Logs/log0"


def train():

	print("\n ---- Training the Deep Neural Network ----- \n")

	RENDER_TO_SCREEN = False

	env = Environment(wrap = False, 
					  grid_size = 7, 
					  rate = 80, 
					  max_time = 30,
					  food_count = 1,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0, 
					  action_space = 5,
					  map_path = None)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = Network(name="model_8x8", load=True)

	brain = Brain(action_space = env.number_of_actions())

	# Hyper-parameters
	# alpha = 0.001  # Learning rate, i.e. which fraction of the Q values should be updated
	# gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
	# epsilon = 0.01  # Probability to choose random action instead of best action

	model.setup(brain)

	tf.summary.scalar('error', tf.squeeze(model.error))

	avg_time = 0
	avg_score = 0
	avg_error = 0


	print_episode = 100
	total_episodes = 10000

	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	# Adds a summary graph of the error over time
	merged_summary = tf.summary.merge_all()

	# Tensorboard capabilties
	writer = tf.summary.FileWriter(LOGDIR)


	with tf.Session() as sess:

		sess.run(init)

		writer.add_graph(sess.graph)

		start_time = time.time()

		print("")

		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			brain.linear_epsilon_decay(total_episodes, episode, start=0.4, end=0.05, percentage= 0.5)

			# brain.linear_alpha_decay(total_episodes, episode)

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				# Retrieve the Q values from the NN in vector form
				Q_vector = sess.run(model.q_values, feed_dict={model.input: state})

				action = brain.choose_action(state, sess, model)

				# print(action)

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				brain.store_transition(state, action, reward, done, new_state)
				
				#'''
				## Standard training with learning after every step

				# Q_vector = sess.run(Q_values, feed_dict={x: state})
				# if final state of the episode
				# print("Q_vector:", Q_vector)
				if done:
					Q_vector[:,action] = reward
					# print("Reward:", reward)
				else:
					# Gathering the now current state's action-value vector
					y_prime = sess.run(model.q_values, feed_dict={model.input: new_state})

					# Equation for training
					maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

					# RL Equation
					Q_vector[:,action] = reward + (brain.GAMMA * maxq)

				_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: state, model.actions: Q_vector})

				state = new_state

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					avg_error += e

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				current_time = math.floor(time.time()-start_time)
				print("Ep:", episode, 
					"\tavg t: {0:.3f}".format(avg_time/print_episode), 
					"\tavg score: {0:.3f}".format(avg_score/print_episode), 
					"\tErr {0:.3f}".format(avg_error/print_episode), 
					"\tepsilon {0:.3f}".format(brain.EPSILON), 
					#"\ttime {0:.0f}:{1:.0f}".format(current_time/60, current_time%60),
					end="")
				print_readable_time(current_time)
				avg_time = 0
				avg_score = 0
				avg_error = 0

				# Save the model's weights and biases to .npz file
				model.save(sess)

				# s = sess.run(merged_summary, feed_dict={model.input: state, model.actions: Q_vector})
				# writer.add_summary(s, episode)

		model.save(sess, verbose=True)

		# save_path = saver.save(sess, MODEL_PATH_SAVE)
		# print("Model saved in path: %s" % save_path)


# Play the game
def play():
	print("\n ----- Playing the game -----\n")

	GRID_SIZE = 5

	# MAP_PATH = "./Maps/Grid{}/map4.txt".format(GRID_SIZE)
	MAP_PATH = None

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  rate = 100,
					  food_count = 1,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  action_space = 5,
					  map_path = MAP_PATH)

	env.play()


if __name__ == '__main__':

	train()

	# play()