import os
import numpy as np
from model import DeepQNetwork, Agent
from Malmo_Environment import Environment
from utils import plotLearning

if __name__ == '__main__':
	print()
	print("RUNNING THE MINECRAFT SIMULATION")
	print()

	RENDER = False
	LOAD_MODEL = True
	start_eps = 0.8

	GRID_SIZE = 6
	LOCAL_GRID_SIZE = 9 # Has to be an odd number (I think...)
	SEED = 1
	WRAP = False
	FOOD_COUNT = 5
	OBSTACLE_COUNT = 0
	# MAP_PATH = "./Maps/Grid{}/map2.txt".format(GRID_SIZE)
	MAP_PATH = None
	
	env = Environment(wrap = WRAP, 
					  grid_size = GRID_SIZE, 
					  rate = 80, 
					  max_time = 50,
					  food_count = FOOD_COUNT,
					  obstacle_count = OBSTACLE_COUNT,
					  zombie_count = 0,
					  action_space = 5,
					  map_path = MAP_PATH)

	brain = Agent(gamma = 0.99, epsilon = start_eps, alpha = 0.003, maxMemorySize = 10000, replace = None)

	# env.play()

	if LOAD_MODEL:
		try:
		    path = "./Models/Torch/my_model.pth"
		    brain.load_model(path)
		    print("Model loaded from path:", path)
		    print()
		    brain.EPSILON = 0.1
		except Exception:
		    print('Could not load model, continue with random initialision (y/n):')
		    print()
		    # input()
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
			# print(observation_)
			if done:
				# reward = -1
				games_played += 1
			brain.storeTransition(observation, action, reward, observation_)

			observation = observation_
			if RENDER: env.render()
	
	print("Done initialising replay memory. Played {} games".format(games_played))

	scores = []
	epsHistory = []
	numGames = 10000
	print_episode = 100
	batch_size = 32

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

			brain.storeTransition(observation, action, reward, observation_)

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

	plotLearning(x, scores, epsHistory, fileName)


