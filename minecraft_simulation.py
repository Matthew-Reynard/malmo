import os
import numpy as np
from model import DeepQNetwork, Agent
from Malmo_Environment import Environment
from utils import plotLearning

# Train the model. (or run it to see how the training went)
def train():
	print()
	print("RUNNING THE MINECRAFT SIMULATION")
	print()

	RENDER = False
	LOAD_MODEL = False
	start_eps = 0.8

	WRAP = False
	GRID_SIZE = 8
	LOCAL_GRID_SIZE = 9 # Has to be an odd number (I think...)
	SEED = 1
	FOOD_COUNT = 5
	OBSTACLE_COUNT = 0
	# MAP_PATH = "./Maps/Grid{}/map2.txt".format(GRID_SIZE)
	MAP_PATH = None
	
	env = Environment(wrap = WRAP, 
					  grid_size = GRID_SIZE, 
					  rate = 80, 
					  max_time = 60,
					  food_count = FOOD_COUNT,
					  obstacle_count = OBSTACLE_COUNT,
					  lava_count = 5,
					  zombie_count = 0,
					  action_space = 5,
					  map_path = MAP_PATH)

	brain = Agent(gamma = 0.99, epsilon = start_eps, alpha = 0.001, maxMemorySize = 10000, replace = 10)

	if LOAD_MODEL:
		try:
		    path = "./Models/Torch/my_model.pth"
		    brain.load_model(path)
		    print("Model loaded from path:", path)
		    print()
		    brain.EPSILON = 0.05
		except Exception:
		    print('Could not load model')
		    print('Press <ENTER> to continue with random initialision')
		    print()
		    input()
		    # quit()
	
	if RENDER: env.prerender()

	games_played = 0

	print("INITIALISING REPLAY MEMORY")

	while brain.memCntr < brain.memSize:
		obs, _ = env.reset()
		observation = env.local_state_vector_3D()
		done = False

		if RENDER: env.render() # Render first screen
		while not done:

			action = brain.chooseAction(observation)

			observation_, reward, done, info = env.step(action)
			observation_ = env.local_state_vector_3D()
			# print(reward)
			if done:
				# reward = -1
				games_played += 1
			brain.storeTransition(observation, action, reward, done, observation_)

			observation = observation_
			if RENDER: env.render()
	
	print("Done initialising replay memory. Played {} games".format(games_played))

	scores = []
	epsHistory = []
	numGames = 100000
	print_episode = 10
	batch_size = 8

	avg_score = 0
	avg_time = 0
	avg_loss = 0

	print()
	print("TRAINING MODEL")
	print()

	for i in range(numGames):
		epsHistory.append(brain.EPSILON)
		done = False
		obs, _ = env.reset()
		observation = env.local_state_vector_3D()
		score = 0
		lastAction = 0

		if RENDER: env.render() # Render first screen
		while not done:
			action = brain.chooseAction(observation)

			observation_, reward, done, info = env.step(action)
			
			observation_ = env.local_state_vector_3D()
			score += reward

			# print(observation_)

			brain.storeTransition(observation, action, reward, done, observation_)

			observation = observation_
			loss = brain.learn(batch_size)
			lastAction = action
			if RENDER: env.render()

		avg_score += info["score"]
		avg_time += info["time"]
		avg_loss += loss.item()


		if i%print_episode == 0 and not i==0 or i == numGames-1:
			print("Episode", i, 
				"\tepsilon: %.4f" %brain.EPSILON,
				"\tavg time: {0:.3f}".format(avg_time/print_episode), 
				"\tavg score: {0:.3f}".format(avg_score/print_episode), 
				"\tavg loss: {0:.3f}".format(avg_loss/print_episode))
			brain.save_model("./Models/Torch/my_model{}.pth".format(i))
			avg_loss = 0
			avg_score = 0
			avg_time = 0

		scores.append(score)
		# print("score:", score)

	brain.save_model("./Models/Torch/my_model.pth")

	x = [i+1 for i in range(numGames)]

	fileName = str(numGames) + 'Games' + 'Gamma' + str(brain.GAMMA) + 'Alpha' + str(brain.ALPHA) + 'Memory' + str(brain.memSize) + '.png'

	# plotLearning(x, scores, epsHistory, fileName)

# Just play the game. (for debugging)
def play():

	print()
	print("PLAYING THE MINECRAFT SIMULATION")
	print()

	GRID_SIZE = 8
	LOCAL_GRID_SIZE = 9 # Has to be an odd number (I think...)
	SEED = 1
	WRAP = False
	FOOD_COUNT = 3
	OBSTACLE_COUNT = 0
	# MAP_PATH = "./Maps/Grid{}/map2.txt".format(GRID_SIZE)
	MAP_PATH = None
	
	env = Environment(wrap = WRAP, 
					  grid_size = GRID_SIZE, 
					  rate = 100, 
					  food_count = FOOD_COUNT,
					  obstacle_count = OBSTACLE_COUNT,
					  lava_count = 10,
					  zombie_count = 0,
					  action_space = 5,
					  map_path = MAP_PATH)

	env.play()


if __name__ == '__main__':
	
	train()

	# play()
