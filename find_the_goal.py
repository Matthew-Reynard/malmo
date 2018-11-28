#!/usr/bin/env python
# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000

from __future__ import print_function

import os, sys
sys.path.insert(0, os.getcwd()) 

from builtins import range
from malmo import MalmoPython
import time
import json
import random
import numpy as np
import torch
# import tensorflow as tf
from model import DeepQNetwork, Agent

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# import malmo.minecraftbootstrap
# malmo.minecraftbootstrap.set_malmo_xsd_path()

if "MALMO_XSD_PATH" not in os.environ:
	os.environ["MALMO_XSD_PATH"] = "/home/matthew/Malmo/Schemas"

if "MALMO_MINECRAFT_ROOT" not in os.environ:
	os.environ["MALMO_MINECRAFT_ROOT"] = "/home/matthew/Malmo/Minecraft"

if "JAVA_HOME" not in os.environ:
	os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"


# Get the agent state
def get_state(grid):
    '''Get the state in the same way snake was'''

    state = torch.zeros([3, 9, 9], dtype=torch.float32)
    
    state[0,4,4] = 1
    
    for i in range(9*9):
        if grid[i] == "gold_block":
            state[1, 8-int(i/9), 8-int(i%9)] = 1
        if grid[i] == "cobblestone" or grid[i] == "stone":
            state[2, 8-int(i/9), 8-int(i%9)] = 1

    return state.numpy()
    # needs to be put to cpu before you can convert it to numpy array

    # DEBUGGING
    # print(state.numpy())

# Reset the Minecraft environment, return the world state
def reset(agent, max_retries=3):
    for retry in range(max_retries):
        try:
            agent.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2.5)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission running ", end=' ')
    
    return world_state


action_space = ["move 1", "move -1", "strafe 1", "strafe -1"]
# action_space = ["move 0", "move 1", "move -1", "strafe 0", "strafe 1", "strafe -1"]

brain = Agent(gamma = 0.99, epsilon = 1.0, alpha = 0.003, maxMemorySize = 100, replace = None)

# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# -- set up the mission -- #
mission_file = './find_the_goal.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

# Python code alterations to the environment
# my_mission.drawBlock(0, 110, 0, "stone")

my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3

try:
    path = "./Models/Torch/my_model.pth"
    brain.load_model(path)
    print("Model loaded from path:", path)
except Exception:
    print('Could not load model, continue with random initialision (y/n):')
    input()
    # quit()

print(my_mission.getSummary())

# INITIALISING REPLAY MEMORY
while brain.memCntr < brain.memSize:
    world_state = reset(agent_host)

    getting_obs = True
    while getting_obs:
        world_state = agent_host.getWorldState()
        # print(world_state)
        if world_state.number_of_observations_since_last_state > 0:
            getting_obs = False
        
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor9x9', 0)
    state = get_state(grid)

    while world_state.is_mission_running: 
        print("-", end="")
        time.sleep(0.1)

        # action based on current state
        # print(state)
        action_taken = np.random.choice(len(action_space))
        agent_host.sendCommand(action_space[action_taken])

        world_state = agent_host.getWorldState()

        for error in world_state.errors:
            print("Error:", error.text)

        # Have we received any observations
        if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floor9x9', 0)
            
            # Get new_state & reward
            new_state = get_state(grid)
            reward = world_state.rewards[-1].getValue()
            
            # Store the transitions
            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # brain.storeTransition(state, torch.Tensor(action_taken, dtype=torch.int32).to(device), torch.Tensor(reward,dtype=torch.float32).to(device), new_state)
            brain.storeTransition(state, action_taken, reward, new_state)

            # Not neccesrily true, might be a few observations after
            state = new_state

print("Done initialising memory")

# scores = []
epsHistory = []
num_episodes = 1000
batch_size = 1

avg_score = 0
avg_loss = 0

# TRAINING
for i in range(num_episodes):

    epsHistory.append(brain.EPSILON)
    score = 0

    # reset env
    world_state = reset(agent_host)
    # print(world_state)

    # for i in range(len(my_mission.getListOfCommandHandlers(0))):
    #     print(my_mission.getListOfCommandHandlers(0)[i])

    # for action in my_mission.getAllowedCommands(0,"DiscreteMovement"):
    #     print(action)

    getting_obs = True
    while getting_obs:
        world_state = agent_host.getWorldState()
        # print(world_state)
        if world_state.number_of_observations_since_last_state > 0:
            getting_obs = False
        
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor9x9', 0)
    state = get_state(grid)

    # Loop until mission ends
    while world_state.is_mission_running: 
        print("-", end="")
        time.sleep(0.1)

        # action based on current state
        # print(state)
        # action_taken = np.random.choice(len(action_space))
        # agent_host.sendCommand(action_space[action_taken])

        action = brain.chooseAction(state)
        agent_host.sendCommand(action_space[action])

        world_state = agent_host.getWorldState()

        for error in world_state.errors:
            print("Error:", error.text)

        # Have we received any observations
        if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floor9x9', 0)
            
            # Get new_state & reward
            new_state = get_state(grid)
            reward = world_state.rewards[-1].getValue()
            score += reward
            
            # Store the transitions
            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # brain.storeTransition(state, torch.Tensor(action_taken, dtype=torch.int32).to(device), torch.Tensor(reward,dtype=torch.float32).to(device), new_state)
            brain.storeTransition(state, action, reward, new_state)

            # Not neccesrily true, might be a few observations after
            state = new_state

            loss = brain.learn(batch_size)

            avg_score += score
            avg_loss += loss.item()

    print("Episode", i, 
        "\tepsilon: %.4f" %brain.EPSILON,
        "\tavg score", avg_score/1,
        "avg loss:", avg_loss/1)

    avg_score = 0
    avg_loss = 0
            	
    print()	
    print("Mission {} ended".format(i+1))
    brain.save_model("./Models/Torch/my_model{}.pth".format(i))
    # Mission has ended.

print()
print("Done")

brain.save_model("./Models/Torch/my_model.pth")
