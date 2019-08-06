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
from utils import print_readable_time, Histogram


# Train
def train():

	MODEL_NAME = "explore_grid16"
	MODEL_NAME_save = "explore_grid16"

	FOLDER = "Impossible"

	MODEL_PATH_SAVE = "./Models/Tensorflow/"+FOLDER+"/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	LOGDIR = "./Logs/"+FOLDER+"/"+MODEL_NAME_save+""

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 16
	LOCAL_GRID_SIZE = 15
	MAP_NUMBER = 0
	RANDOMIZE_MAPS = True

	# MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
	MAP_PATH = None
	# MAP_PATH = "./Maps/Grid{}/impossible_map0.txt".format(GRID_SIZE, MAP_NUMBER)

	print("\n ---- Training the Deep Neural Network ----- \n")

	RENDER_TO_SCREEN = False
	# RENDER_TO_SCREEN = True 

	env = Environment(wrap = False,
					  grid_size = GRID_SIZE,
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80,
					  max_time = 100,
					  food_count = 0,
					  stick_count = 0,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  history = 100,
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = Network(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, load=False, path="./Models/Tensorflow/"+FOLDER+"/")

	brain = Brain(epsilon=0.1, action_space = env.number_of_actions())

	model.setup(brain)

	score = tf.placeholder(tf.float32, [])
	avg_t = tf.placeholder(tf.float32, [])
	epsilon = tf.placeholder(tf.float32, [])
	avg_r = tf.placeholder(tf.float32, [])

	tf.summary.scalar('error', tf.squeeze(model.error))
	tf.summary.scalar('score', score)
	tf.summary.scalar('average time', avg_t)
	tf.summary.scalar('epsilon', epsilon)
	tf.summary.scalar('avg reward', avg_r)

	avg_time = 0
	avg_score = 0
	avg_error = 0
	avg_reward = 0
	cumulative_reward = 0

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
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

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
				MAP_PATH = "./Maps/Grid{}/impossible_map_empty{}.txt".format(GRID_SIZE, np.random.randint(5))
				env.set_map(MAP_PATH)

			state, info = env.reset()
			# state, info = env.quick_reset()
			done = False

			# brain.linear_epsilon_decay(total_episodes, episode, start=1.0, end=0.05, percentage=0.5)

			# brain.linear_alpha_decay(total_episodes, episode)

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				action = brain.choose_action(state, sess, model)
				# print(action)

				# Update environment by performing action
				new_state, reward, done, info = env.step(action)
				# print(new_state[3], reward)

				brain.store_transition(state, action, reward, done, new_state)
				
				e, Q_vector = brain.train(model, sess)

				state = new_state

				cumulative_reward += reward

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					avg_error += e
					avg_reward += cumulative_reward 
					cumulative_reward = 0

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				
				current_time = math.floor(time.time()-start_time)
				print("Ep:", episode,
					"\tavg t: {0:.3f}".format(avg_time/print_episode),
					"\tavg score: {0:.3f}".format(avg_score/print_episode),
					"\terr {0:.3f}".format(avg_error/print_episode),
					"\tavg_reward {0:.3f}".format(avg_reward/print_episode), # avg cumulative reward
					"\tepsilon {0:.3f}".format(brain.EPSILON),
					end="")
				print_readable_time(current_time)

				# Save the model's weights and biases to .npz file
				model.save(sess, name=MODEL_NAME_save)
				# save_path = saver.save(sess, MODEL_PATH_SAVE)

				s = sess.run(merged_summary, feed_dict={model.input: state, model.actions: Q_vector, score:avg_score/print_episode, avg_t:avg_time/print_episode, epsilon:brain.EPSILON, avg_r:avg_reward/print_episode})
				writer.add_summary(s, episode)

				avg_time = 0
				avg_score = 0
				avg_error = 0
				avg_reward = 0

		model.save(sess, verbose=True, name=MODEL_NAME_save)

		# save_path = saver.save(sess, MODEL_PATH_SAVE)
		# print("Model saved in path: %s" % save_path)

		writer.close()


# Meta Network training with fixed Dojo networks
def train_MetaNetwork():

	print("\n ---- Training the Meta Network ----- \n")

	MODEL_NAME = "meta_grid16_all"
	MODEL_NAME_save = "meta_grid16_all"

	DIAMOND_MODEL_NAME = "diamond_grid16"
	ZOMBIE_MODEL_NAME = "zombie_grid16"
	EXPLORE_MODEL_NAME = "explore_grid16"
	# EXTRA_MODEL_NAME = "extra15_input6_2"

	# MODEL_NAME = "meta15_input6_1M_unfrozen_dojos"
	# DIAMOND_MODEL_NAME = "diamond15_input4_best_unfrozen_at_1M"
	# ZOMBIE_MODEL_NAME = "zombie15_input4_best_unfrozen_at_1M"
	# EXPLORE_MODEL_NAME = "explore15_input4_best_unfrozen_at_1M"

	# MODEL_NAME = "meta15_input6_1M_random_unfrozen_cointoss"
	# DIAMOND_MODEL_NAME = "diamond15_input4_1M_random_unfrozen_cointoss"
	# ZOMBIE_MODEL_NAME = "zombie15_input4_1M_random_unfrozen_cointoss"
	# EXPLORE_MODEL_NAME = "explore15_input4_1M_random_unfrozen_cointoss"k

	FOLDER = "Impossible"
	DOJO_FOLDER = "Impossible"

	MODEL_PATH_SAVE = "./Models/Tensorflow/"+FOLDER+"/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	LOGDIR = "./Logs/"+FOLDER+"/"+MODEL_NAME_save+""

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 16
	LOCAL_GRID_SIZE = 15
	MAP_PATH = None

	RANDOMIZE_MAPS = True

	RENDER_TO_SCREEN = False
	# RENDER_TO_SCREEN = True

	env = Environment(wrap = False,
					  grid_size = GRID_SIZE,
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80,
					  max_time = 120,
					  food_count = 0,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  history = 50,
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = MetaNetwork(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, path="./Models/Tensorflow/"+FOLDER+"/", load=False,  trainable=True)
 
	diamond_net = Network(local_size=LOCAL_GRID_SIZE, name=DIAMOND_MODEL_NAME, path="./Models/Tensorflow/"+DOJO_FOLDER+"/", load=True, trainable=False)

	zombie_net = Network(local_size=LOCAL_GRID_SIZE, name=ZOMBIE_MODEL_NAME, path="./Models/Tensorflow/"+DOJO_FOLDER+"/", load=True, trainable=False)

	explore_net = Network(local_size=LOCAL_GRID_SIZE, name=EXPLORE_MODEL_NAME, path="./Models/Tensorflow/"+DOJO_FOLDER+"/", load=True, trainable=False)

	# extra_net = Network(local_size=LOCAL_GRID_SIZE, name=EXTRA_MODEL_NAME, path="./Models/Tensorflow/"+DOJO_FOLDER+"/", load=False, trainable=True)

	brain = Brain(epsilon=0.05, action_space=3)

	model.setup(brain)
	diamond_net.setup(brain)
	zombie_net.setup(brain)
	explore_net.setup(brain)
	# extra_net.setup(brain)

	score = tf.placeholder(tf.float32, [])
	avg_t = tf.placeholder(tf.float32, [])
	epsilon = tf.placeholder(tf.float32, [])
	avg_r = tf.placeholder(tf.float32, [])

	tf.summary.scalar('error', tf.squeeze(model.error))
	tf.summary.scalar('score', score)
	tf.summary.scalar('average time', avg_t)
	tf.summary.scalar('epsilon', epsilon)
	tf.summary.scalar('avg reward', avg_r)

	avg_time = 0
	avg_score = 0
	avg_error = 0
	avg_reward = 0
	cumulative_reward = 0

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

	# Histogram
	histogram = Histogram(3, 10, total_episodes)

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
				MAP_PATH = "./Maps/Grid{}/impossible_map{}.txt".format(GRID_SIZE, np.random.randint(5))
				env.set_map(MAP_PATH)

			# state, info = env.reset()
			state, info = env.quick_reset()
			done = False

			# brain.linear_epsilon_decay(total_episodes, episode, start=1.0, end=0.05, percentage=0.5)

			# brain.linear_alpha_decay(total_episodes, episode)

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				# Retrieve the Q values from the NN in vector form
				Dojo_vector = sess.run(model.q_values, feed_dict={model.input: state})

				dojo = brain.choose_action(state, sess, model)
				
				histogram.check_section(episode)
				histogram.add(dojo)

				# dojo = np.random.randint(3)
				# dojo = 0

				# print(dojo)

				if dojo == 0:
					dojo_state = state
					# dojo_state[2]=0
					# dojo_state[3]=0
					# dojo_state = np.delete(dojo_state, 2, 0)# Take out the zombie layer
					# dojo_state = np.delete(dojo_state, 2, 0)# Take out the history layer
					action = brain.choose_dojo(dojo_state, sess, diamond_net, env.number_of_actions(), 0.05)

				elif dojo == 1:
					dojo_state = state
					# dojo_state[1]=0
					# dojo_state[3]=0
					# dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					# dojo_state = np.delete(dojo_state, 2, 0)# Take out the history layer
					action = brain.choose_dojo(dojo_state, sess, zombie_net, env.number_of_actions(), 0.05)

				elif dojo == 2:
					dojo_state = state
					# dojo_state[1]=0
					# dojo_state[2]=0
					# dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					# dojo_state = np.delete(dojo_state, 1, 0)# Take out the zombie layer
					action = brain.choose_dojo(dojo_state, sess, explore_net, env.number_of_actions(), 0.05)

				# elif dojo == 3:
				# 	dojo_state = state
				# 	action = brain.choose_dojo(dojo_state, sess, extra_net, env.number_of_actions(), 0.05)

				# print(action)

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)

				# print(new_state)

				brain.store_transition_dojo(state, action, reward, done, new_state, dojo)

				# print(tf.trainable_variables(scope=None))

				# if dojo == 0:
				# 	e, Q_vector = brain.train_3_dojos(diamond_net, sess, dojo)

				# elif dojo == 1:
				# 	e, Q_vector = brain.train_3_dojos(zombie_net, sess, dojo)

				# elif dojo == 2:
				# 	e, Q_vector = brain.train_3_dojos(explore_net, sess, dojo)

				# e, Q_vector = brain.train_3(sess, diamond_net, zombie_net, explore_net)

				# e, Q_vector = brain.train(extra_net, sess)

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

				state = new_state

				cumulative_reward += reward

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					avg_error += e
					avg_reward += cumulative_reward 
					cumulative_reward = 0

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				
				current_time = math.floor(time.time()-start_time)
				print("Ep:", episode,
					"\tavg t: {0:.3f}".format(avg_time/print_episode),
					"\tavg score: {0:.3f}".format(avg_score/print_episode),
					"\tErr {0:.3f}".format(avg_error/print_episode),
					"\tavg_reward {0:.3f}".format(avg_reward/print_episode), # avg cumulative reward
					"\tepsilon {0:.3f}".format(brain.EPSILON),
					end="")
				print_readable_time(current_time)

				# Save the model's weights and biases to .npz file
				model.save(sess, name=MODEL_NAME_save)
				# diamond_net.save(sess, name=DIAMOND_MODEL_NAME+"")
				# zombie_net.save(sess, name=ZOMBIE_MODEL_NAME+"")
				# explore_net.save(sess, name=EXPLORE_MODEL_NAME+"")
				# extra_net.save(sess, name=EXTRA_MODEL_NAME+"")

				# save_path = saver.save(sess, MODEL_PATH_SAVE)

				s = sess.run(merged_summary, feed_dict={model.input: state, model.actions: Dojo_vector, score:avg_score/print_episode, avg_t:avg_time/print_episode, epsilon:brain.EPSILON, avg_r:avg_reward/print_episode})
				writer.add_summary(s, episode)

				avg_time = 0
				avg_score = 0
				avg_error = 0
				avg_reward = 0

		model.save(sess, verbose=True, name=MODEL_NAME_save)
		# diamond_net.save(sess, verbose=True, name=DIAMOND_MODEL_NAME+"")
		# zombie_net.save(sess, verbose=True, name=ZOMBIE_MODEL_NAME+"")
		# explore_net.save(sess, verbose=True, name=EXPLORE_MODEL_NAME+"")
		# extra_net.save(sess, verbose=True, name=EXTRA_MODEL_NAME+"")

		# save_path = saver.save(sess, MODEL_PATH_SAVE)
		# print("Model saved in path: %s" % save_path)

		writer.close()
		histogram.plot()


# Run the given model
def run():

	MODEL_NAME = "explore15_input6"

	FOLDER = "Best_Dojos"

	MODEL_PATH_SAVE = "./Models/Tensorflow/"+FOLDER+"/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 32
	LOCAL_GRID_SIZE = 15
	MAP_NUMBER = 0
	RANDOMIZE_MAPS = False

	# MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
	MAP_PATH = None
	MAP_PATH = "./Maps/Grid{}/impossible_map1.txt".format(GRID_SIZE, MAP_NUMBER)

	print("\n ---- Running the Deep Q Network ----- \n")

	RENDER_TO_SCREEN = False
	RENDER_TO_SCREEN = True

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80, 
					  max_time = 60,
					  food_count = 0,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  history = 40,
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = Network(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, load=True, path="./Models/Tensorflow/"+FOLDER+"/", trainable = False)

	brain = Brain(epsilon=0.0, action_space = env.number_of_actions())

	model.setup(brain)

	avg_time = 0
	avg_score = 0
	avg_reward = 0
	cumulative_reward = 0

	# Number of episodes
	print_episode = 100
	total_episodes = 100

	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

	# Begin session
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_SAVE)
			print("Model restored.")
		else:
			sess.run(init)

		print("")

		for episode in range(total_episodes):
			
			if RANDOMIZE_MAPS:
				MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, np.random.randint(10))
				env.set_map(MAP_PATH)

			# state, info = env.reset()
			state, info = env.quick_reset()
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

				cumulative_reward += reward

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					avg_reward += cumulative_reward 
					cumulative_reward = 0

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				
				print("Ep:", episode,
					"\tavg t: {0:.3f}".format(avg_time/print_episode),
					"\tavg score: {0:.3f}".format(avg_score/print_episode),
					"\tavg_reward {0:.3f}".format(avg_reward/print_episode), # avg cumulative reward
					"\tepsilon {0:.3f}".format(brain.EPSILON),
					end="\n")

				avg_time = 0
				avg_score = 0
				avg_reward = 0


# Run Meta Network with fixed Dojo networks
def run_MetaNetwork():

	print("\n ---- Running the Meta Network ----- \n")

	MODEL_NAME = "meta15_input6_4_unfrozen"
	DIAMOND_MODEL_NAME = "diamond15_input6_best_unfrozen4_300k"
	ZOMBIE_MODEL_NAME = "zombie15_input6_best_unfrozen4_300k"
	EXPLORE_MODEL_NAME = "explore15_input6_best_unfrozen4_300k"

	MODEL_PATH_SAVE = "./Models/Tensorflow/Meta/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

	LOGDIR = "./Logs/"+MODEL_NAME

	USE_SAVED_MODEL_FILE = False

	GRID_SIZE = 10
	LOCAL_GRID_SIZE = 15
	MAP_PATH = None

	RANDOMIZE_MAPS = True

	RENDER_TO_SCREEN = False
	RENDER_TO_SCREEN = True

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE,
					  local_size = LOCAL_GRID_SIZE,
					  rate = 80, 
					  max_time = 100,
					  food_count = 10,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 2,
					  history = 40, 
					  action_space = 5,
					  map_path = MAP_PATH)

	if RENDER_TO_SCREEN:
		env.prerender()

	model = MetaNetwork(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, path="./Models/Tensorflow/Best_Meta/", load=True,  trainable = False)

	diamond_net = Network(local_size=LOCAL_GRID_SIZE, name=DIAMOND_MODEL_NAME, path="./Models/Tensorflow/Best_Dojos/", load=True, trainable = False)

	zombie_net = Network(local_size=LOCAL_GRID_SIZE, name=ZOMBIE_MODEL_NAME, path="./Models/Tensorflow/Best_Dojos/", load=True, trainable = False)

	explore_net = Network(local_size=LOCAL_GRID_SIZE, name=EXPLORE_MODEL_NAME, path="./Models/Tensorflow/Best_Dojos/", load=True, trainable = False)

	brain = Brain(epsilon=0.0, action_space=3)

	model.setup(brain)
	diamond_net.setup(brain)
	zombie_net.setup(brain)
	explore_net.setup(brain)

	avg_time = 0
	avg_score = 0
	avg_reward = 0
	cumulative_reward = 0

	# Number of episodes
	print_episode = 100
	total_episodes = 100

	saver = tf.train.Saver()

	# Initialising all variables (weights and biases)
	init = tf.global_variables_initializer()

 	# GPU capabilities
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

	# Begin session
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		if USE_SAVED_MODEL_FILE:
			saver.restore(sess, MODEL_PATH_SAVE)
			print("Model restored.")
		else:
			sess.run(init)

		start_time = time.time()

		print("")

		for episode in range(total_episodes):

			if RANDOMIZE_MAPS:
				# Make a random map 0: lava, 1: obstacle
				MAP_PATH = "./Maps/Grid10/map{}.txt".format(np.random.randint(10))
				env.set_map(MAP_PATH)

			state, info = env.reset()
			done = False

			if RENDER_TO_SCREEN:
				env.render()

			while not done:

				dojo = brain.choose_action(state, sess, model)
				# print(dojo)

				if dojo == 0:
					dojo_state = state
					# dojo_state = np.delete(dojo_state, 2, 0)# Take out the zombie layer
					# dojo_state = np.delete(dojo_state, 2, 0)# Take out the history layer
					action = brain.choose_dojo(dojo_state, sess, diamond_net, env.number_of_actions(), 0.0)
				elif dojo == 1:
					dojo_state = state
					# dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					# dojo_state = np.delete(dojo_state, 2, 0)# Take out the history layer
					action = brain.choose_dojo(dojo_state, sess, zombie_net, env.number_of_actions(), 0.0)
				elif dojo == 2:
					dojo_state = state
					# dojo_state = np.delete(dojo_state, 1, 0)# Take out the diamond layer
					# dojo_state = np.delete(dojo_state, 1, 0)# Take out the zombie layer
					action = brain.choose_dojo(dojo_state, sess, explore_net, env.number_of_actions(), 0.0)

				# print(action)

				# Update environment with by performing action
				new_state, reward, done, info = env.step(action)
				# print(new_state)

				state = new_state

				cumulative_reward += reward

				if RENDER_TO_SCREEN:
					env.render()

				if done:
					avg_time += info["time"]
					avg_score += info["score"]
					avg_reward += cumulative_reward 
					cumulative_reward = 0

			if (episode%print_episode == 0 and episode != 0) or (episode == total_episodes-1):
				
				current_time = math.floor(time.time()-start_time)
				print("Ep:", episode,
					"\tavg t: {0:.3f}".format(avg_time/print_episode),
					"\tavg score: {0:.3f}".format(avg_score/print_episode),
					"\tavg_reward {0:.3f}".format(avg_reward/print_episode), # avg cumulative reward
					"\tepsilon {0:.3f}".format(brain.EPSILON),
					end="")
				print_readable_time(current_time)

				avg_time = 0
				avg_score = 0
				avg_reward = 0


# Play the game
def play():
	print("\n ----- Playing the game -----\n")

	GRID_SIZE = 16
	LOCAL_GRID_SIZE = 15 # for printing out the state

	MAP_NUMBER = 0
	# MAP_NUMBER = np.random.randint(5)
	# MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
	# MAP_PATH = None
	MAP_PATH = "./Maps/Grid{}/impossible_map{}.txt".format(GRID_SIZE, MAP_NUMBER)

	env = Environment(wrap = False, 
					  grid_size = GRID_SIZE, 
					  local_size = LOCAL_GRID_SIZE,
					  rate = 100,
					  food_count = 0,
					  stick_count = 0,
					  obstacle_count = 0,
					  lava_count = 0,
					  zombie_count = 0,
					  history = 100,
					  action_space = 5,
					  map_path = MAP_PATH)

	env.play()


# Main function 
if __name__ == '__main__':

	# train()

	train_MetaNetwork()

	# run()

	# run_MetaNetwork()

	# play()
 