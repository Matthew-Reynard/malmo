#!/usr/bin/python3

# Imports
import tensorflow as tf
import numpy as np
import time
import math

# Custom imports
from DQN import Network, MetaNetwork
from Agent import Brain
from Malmo_Environment import Environment
from utils import print_readable_time

# MODEL_NAME = "diamond_dojo_local9"
# MODEL_NAME = "diamond_dojo_local15"
# diamond_dojo_local15_maps

# MODEL_NAME = "zombie_dojo_local9"
# MODEL_NAME = "zombie_dojo_local15"

# MODEL_NAME = "complex_local9"
# MODEL_NAME = "complex_local15"

# MODEL_NAME = "meta_network_local9"
# MODEL_NAME = "meta_network_local15"

# MODEL_NAME = "explore_dojo_local15"

# meta_local15_input4
# diamond_local15_input3
# zombie_local15_input3
# explore_local15_input3

# complex_local15_10r

# diamond_local15_input4
# zombie_local15_input4
# explore_local15_input4
# diamond_local15_input4_5f

# complex_local15_input6

# Train
def train():

	MODEL_NAME = "explore_local15_input4_tfgraph"

	FOLDER = "Dojos"

	MODEL_PATH_SAVE = "./Models/Tensorflow/"+FOLDER+"/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	LOGDIR = "./Logs/"+MODEL_NAME

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 10
	LOCAL_GRID_SIZE = 15
	MAP_NUMBER = 0
	RANDOMIZE_MAPS = True

	# MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
	MAP_PATH = None


	print("\n ---- Training the Deep Neural Network ----- \n")

	RENDER_TO_SCREEN = False
	# RENDER_TO_SCREEN = True

	env = Environment(wrap = False,
					  grid_size = GRID_SIZE,
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80,
					  max_time = 200,
					  food_count = 0,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  history = 40,
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = Network(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, load=False, path="./Models/Tensorflow/"+FOLDER+"/")

	brain = Brain(epsilon=0.05, action_space = env.number_of_actions())

	model.setup(brain)

	tf.summary.scalar('error', tf.squeeze(model.error))

	avg_time = 0
	avg_score = 0
	avg_error = 0
 
	# Number of episodes
	print_episode = 1000
	total_episodes = 100000 

	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	# Adds a summary graph of the error over time
	merged_summary = tf.summary.merge_all()

	# Tensorboard capabilties
	writer = tf.summary.FileWriter(LOGDIR)

	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

	# Begin session
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_SAVE)
			print("Model restored.")

		sess.run(init)

		writer.add_graph(sess.graph)

		start_time = time.time()

		print("")

		for episode in range(total_episodes):

			if RANDOMIZE_MAPS:
				# Make a random map 0: lava, 1: obstacle
				MAP_PATH = "./Maps/Grid10/map{}.txt".format(np.random.randint(10))
				env.set_map(MAP_PATH)

			state, info = env.reset()
			done = False

			brain.linear_epsilon_decay(total_episodes, episode, start=0.5, end=0.05, percentage=0.5)

			# brain.linear_alpha_decay(total_episodes, episode)

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				# Retrieve the Q values from the NN in vector form
				# Q_vector = sess.run(model.q_values, feed_dict={model.input: state})

				action = brain.choose_action(state, sess, model)

				# print(action)

				# Update environment by performing action
				new_state, reward, done, info = env.step(action)

				# print(new_state)

				brain.store_transition(state, action, reward, done, new_state)
				
				e, Q_vector = brain.train(model, sess)

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
					end="")
				print_readable_time(current_time)

				avg_time = 0
				avg_score = 0
				avg_error = 0

				# Save the model's weights and biases to .npz file
				model.save(sess)
				save_path = saver.save(sess, MODEL_PATH_SAVE)

				s = sess.run(merged_summary, feed_dict={model.input: state, model.actions: Q_vector})
				writer.add_summary(s, episode)

		model.save(sess, verbose=True)

		save_path = saver.save(sess, MODEL_PATH_SAVE)
		print("Model saved in path: %s" % save_path)

		writer.close()


# Meta Network training with fixed Dojo networks
def train_MetaNetwork():

	print("\n ---- Training the Meta Network ----- \n")

	MODEL_NAME = "meta_local15_input6"
	DIAMOND_MODEL_NAME = "diamond_local15_input4"
	ZOMBIE_MODEL_NAME = "zombie_local15_input4"
	EXPLORE_MODEL_NAME = "explore_local15_input4"

	MODEL_PATH_SAVE = "./Models/Tensorflow/Meta/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	LOGDIR = "./Logs/"+MODEL_NAME

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 10 
	LOCAL_GRID_SIZE = 15
	MAP_PATH = None

	RANDOMIZE_MAPS = True

	RENDER_TO_SCREEN = False
	# RENDER_TO_SCREEN = True

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE,
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80, 
					  max_time = 200,
					  food_count = 3,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 2, 
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = MetaNetwork(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, load=False)

	diamond_net = Network(local_size=LOCAL_GRID_SIZE, name=DIAMOND_MODEL_NAME, path="./Models/Tensorflow/Dojos/", load=True, trainable = False)

	zombie_net = Network(local_size=LOCAL_GRID_SIZE, name=ZOMBIE_MODEL_NAME, path="./Models/Tensorflow/Dojos/", load=True, trainable = False)

	explore_net = Network(local_size=LOCAL_GRID_SIZE, name=EXPLORE_MODEL_NAME, path="./Models/Tensorflow/Dojos/", load=True, trainable = False)

	brain = Brain(epsilon=0.01, action_space = 2)

	model.setup(brain)
	diamond_net.setup(brain)
	zombie_net.setup(brain)
	explore_net.setup(brain)

	tf.summary.scalar('error', tf.squeeze(model.error))

	avg_time = 0
	avg_score = 0
	avg_error = 0

	# Number of episodes
	print_episode = 1000
	total_episodes = 200000

	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	# Adds a summary graph of the error over time
	merged_summary = tf.summary.merge_all()

	# Tensorboard capabilties
	writer = tf.summary.FileWriter(LOGDIR)

 	# GPU capabilities
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

	# Begin session
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_SAVE)
			print("Model restored.")
		else:
			sess.run(init)

		writer.add_graph(sess.graph)

		start_time = time.time()

		print("")

		for episode in range(total_episodes):

			if RANDOMIZE_MAPS:
				# Make a random map 0: lava, 1: obstacle
				MAP_PATH = "./Maps/Grid10/map{}.txt".format(np.random.randint(10))
				env.set_map(MAP_PATH)

			state, info = env.reset()
			done = False

			brain.linear_epsilon_decay(total_episodes, episode, start=0.5, end=0.05, percentage=0.5)

			# brain.linear_alpha_decay(total_episodes, episode)

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				# Retrieve the Q values from the NN in vector form
				Dojo_vector = sess.run(model.q_values, feed_dict={model.input: state})

				dojo = brain.choose_action(state, sess, model)

				# print(dojo)

				if dojo == 0:
					# state[2] = 0 # Zero out the zombies layer
					dojo_state = state
					dojo_state = np.delete(dojo_state, 2, 0)# Take out the zombie layer
					dojo_state = np.delete(dojo_state, 3, 0)# Take out the history layer
					action = brain.choose_dojo(dojo_state, sess, diamond_net, env.number_of_actions(), 0.01)
				elif dojo == 1:
					# state[1] = 0 # Zero out the diamond layer
					dojo_state = state
					dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					dojo_state = np.delete(dojo_state, 3, 0)# Take out the history layer
					action = brain.choose_dojo(dojo_state, sess, zombie_net, env.number_of_actions(), 0.01)
				elif dojo == 2:
					dojo_state = state
					dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					dojo_state = np.delete(dojo_state, 2, 0)# Take out the zombie layer
					action = brain.choose_dojo(dojo_state, sess, explore_net, env.number_of_actions(), 0.01)

				# print(action)

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				# print(new_state)

				brain.store_transition(state, dojo, reward, done, new_state)
				
				## Standard training with learning after every step

				# print(tf.trainable_variables(scope=None))

				if done:
					Dojo_vector[:,dojo] = reward
					# print("Reward:", reward)
				else:
					# Gathering the now current state's action-value vector
					y_prime = sess.run(model.q_values, feed_dict={model.input: new_state})

					# Equation for training
					maxq = sess.run(model.y_prime_max, feed_dict={model.actions: y_prime})

					# RL Equation
					Dojo_vector[:,dojo] = reward + (brain.GAMMA * maxq)

				_, e = sess.run([model.optimizer, model.error], feed_dict={model.input: state, model.actions: Dojo_vector})

				## Training using replay memory

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
					end="")
				print_readable_time(current_time)

				avg_time = 0
				avg_score = 0
				avg_error = 0

				# Save the model's weights and biases to .npz file
				model.save(sess)
				save_path = saver.save(sess, MODEL_PATH_SAVE)

				s = sess.run(merged_summary, feed_dict={model.input: state, model.actions: Dojo_vector})
				writer.add_summary(s, episode)

		model.save(sess, verbose=True)

		save_path = saver.save(sess, MODEL_PATH_SAVE)
		print("Model saved in path: %s" % save_path)

		writer.close()


# Run the given model
def run():

	MODEL_NAME = "complex_local15_10r"

	MODEL_PATH_SAVE = "./Models/Tensorflow/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	LOGDIR = "./Logs/"+MODEL_NAME

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 10
	LOCAL_GRID_SIZE = 15
	MAP_NUMBER = 2

	# MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
	MAP_PATH = None

	print("\n ---- Running the Deep Q Network ----- \n")

	RENDER_TO_SCREEN = False

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80, 
					  max_time = 200,
					  food_count = 10,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 1, 
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = Network(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, load=True, path="./Models/Tensorflow/Complex/", trainable = False, )

	brain = Brain(epsilon=0.01, action_space = env.number_of_actions())

	model.setup(brain)

	avg_time = 0
	avg_score = 0

	# Number of episodes
	print_episode = 100
	total_episodes = 1000

	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	# Begin session
	with tf.Session() as sess:

		# if USE_SAVED_MODEL_FILE:
		# 	saver.restore(sess, MODEL_PATH_SAVE)
		# 	print("Model restored.")

		sess.run(init)

		print("")

		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				action = brain.choose_action(state, sess, model)

				# print(action)

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				# print(new_state)

				state = new_state

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				
				print("Ep:", episode,
					"\tavg t: {0:.3f}".format(avg_time/print_episode),
					"\tavg score: {0:.3f}".format(avg_score/print_episode),
					"\tepsilon {0:.3f}".format(brain.EPSILON),
					end="\n")

				avg_time = 0
				avg_score = 0


# Run Meta Network with fixed Dojo networks
def run_MetaNetwork():

	print("\n ---- Running the Meta Network ----- \n")

	MODEL_NAME = "meta_local15_input4"
	DIAMOND_MODEL_NAME = "diamond_local15_input3"
	ZOMBIE_MODEL_NAME = "zombie_local15_input3"

	MODEL_PATH_LOAD = "./Models/Tensorflow/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 8
	LOCAL_GRID_SIZE = 15
	MAP_PATH = None

	RENDER_TO_SCREEN = False

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE,
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80, 
					  max_time = 200,
					  food_count = 10,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 1, 
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = MetaNetwork(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, load=True, trainable = False)

	diamond_net = Network(local_size=LOCAL_GRID_SIZE, name=DIAMOND_MODEL_NAME, load=True, path="./Models/Tensorflow/Dojos/", trainable = False, )

	zombie_net = Network(local_size=LOCAL_GRID_SIZE, name=ZOMBIE_MODEL_NAME, load=True, path="./Models/Tensorflow/Dojos/", trainable = False)

	brain = Brain(epsilon=0.00, action_space = 2)

	model.setup(brain)
	diamond_net.setup(brain)
	zombie_net.setup(brain)

	avg_time = 0
	avg_score = 0

	# Number of episodes
	print_episode = 100
	total_episodes = 1000

	# Checkpoint saver
	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

	# Begin session
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_LOAD)
			print("Model restored.")
		else:
			sess.run(init)

		start_time = time.time()

		print("")

		for episode in range(total_episodes):
			state, info = env.reset()
			done = False

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				dojo = brain.choose_action(state, sess, model)

				if dojo == 0:
					# state[2] = 0 # Zero out the zombies layer
					dojo_state = state
					dojo_state = np.delete(dojo_state, 2, 0)# Take out the zombie layer
					action = brain.choose_dojo(dojo_state, sess, diamond_net, env.number_of_actions(), 0.0)
				elif dojo == 1:
					# state[1] = 0 # Zero out the diamond layer
					dojo_state = state
					dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					action = brain.choose_dojo(dojo_state, sess, zombie_net, env.number_of_actions(), 0.0)

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				# print(new_state)

				brain.store_transition(state, dojo, reward, done, new_state)

				state = new_state

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				
				current_time = math.floor(time.time()-start_time)
				print("Ep:", episode,
					"\tavg t: {0:.3f}".format(avg_time/print_episode),
					"\tavg score: {0:.3f}".format(avg_score/print_episode),
					"\tepsilon {0:.3f}".format(brain.EPSILON),
					end="")
				print_readable_time(current_time)

				avg_time = 0
				avg_score = 0


# Play the game
def play():
	print("\n ----- Playing the game -----\n")

	GRID_SIZE = 10
	LOCAL_GRID_SIZE = 15 # for printing out the state

	# MAP_NUMBER = np.random.randint(10)
	# MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
	MAP_PATH = None

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  local_size = LOCAL_GRID_SIZE,
					  rate = 200,
					  food_count = 3,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  history = 0,
					  action_space = 5,
					  map_path = MAP_PATH)

	env.play()


# Main function
if __name__ == '__main__':

	train()

	# train_MetaNetwork()

	# run()

	# run_MetaNetwork()

	# play()
 