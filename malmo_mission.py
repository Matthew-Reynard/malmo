#!/usr/bin/env python
# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000

from __future__ import print_function

import os, sys
sys.path.insert(0, os.getcwd()) 

import tensorflow as tf
import numpy as np
import time
import json
import math
import csv
# from builtins import range # Not sure if this is used
from malmo import MalmoPython

# Custom imports
from DQN import Network, MetaNetwork
from Agent import Brain
from utils import getch, print_readable_time
from my_mission import missionXML

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# import malmo.minecraftbootstrap
# malmo.minecraftbootstrap.set_malmo_xsd_path()

if sys.platform == 'linux': 
    if "MALMO_XSD_PATH" not in os.environ:
        os.environ["MALMO_XSD_PATH"] = "/home/matthew/Malmo/Schemas"

    if "MALMO_MINECRAFT_ROOT" not in os.environ:
        os.environ["MALMO_MINECRAFT_ROOT"] = "/home/matthew/Malmo/Minecraft"

    if "JAVA_HOME" not in os.environ:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"


if sys.platform == 'win32':

    pass


MODEL_NAME = "default15_input6_adam_300k"
# DIAMOND_MODEL_NAME = "diamond15"
# ZOMBIE_MODEL_NAME = "zombie15"
# EXPLORE_MODEL_NAME = "explore15"

FOLDER = "Best_Default"

MODEL_CHECKPOINT = "./Models/Tensorflow/"+FOLDER+"/"+MODEL_NAME+"/"+MODEL_NAME+".ckpt"

# This is for viewing the model and summaries in Tensorboard
LOGDIR = "./Logs/"+MODEL_NAME

# Parameters
GRID_SIZE = 10
LOCAL_GRID_SIZE = 15
MAP_NUMBER = 7
MAP_PATH = "./Maps/Grid{}/map{}.txt".format(GRID_SIZE, MAP_NUMBER)
# MAP_PATH = None


def get_state(steve, diamonds, zombies, grid):
    '''
    Get the state and convert it to the one-hot grid
    NOTE: NEED TO FIX THE DISCREPANCIES BETWEEN THE COORD SYSTEMS
    '''
    LGS = LOCAL_GRID_SIZE

    local_pos = int((LOCAL_GRID_SIZE-1)/2)

    state = np.zeros([6, LGS, LGS], dtype=np.float32)
    
    state[0,int((LGS-1)/2),int((LGS-1)/2)] = 1

    # Diamonds
    for pos in diamonds:
        x_diamond = local_pos+int(pos[0])-int(steve[0])
        y_diamond = local_pos+int(pos[1])-int(steve[1])

        if x_diamond < LOCAL_GRID_SIZE and x_diamond >= 0 and y_diamond < LOCAL_GRID_SIZE and y_diamond >= 0:
            state[1, LGS-1-y_diamond, LGS-1-x_diamond] = 1
    
    # Zombies
    for pos in zombies:
        x_zombie = local_pos+int(pos[0])-int(steve[0])
        y_zombie = local_pos+int(pos[1])-int(steve[1])

        if x_zombie < LOCAL_GRID_SIZE and x_zombie >= 0 and y_zombie < LOCAL_GRID_SIZE and y_zombie >= 0:
            state[2, LGS-1-y_zombie, LGS-1-x_zombie] = 1

    # print("grid:", grid)

    # Lava
    for i in range(LGS*LGS):
        if grid[i] == "lava" or grid[i] == "flowing_lava":
            state[4, LGS-1-int(i/LGS), LGS-1-int(i%LGS)] = 1

    # Obstacles
    for i in range(LGS*LGS):
        if grid[i+225] == "stone":
            state[5, LGS-1-int(i/LGS), LGS-1-int(i%LGS)] = 1

    # DEBUGGING - needs to be put to cpu before you can convert it to numpy array
    # print(state.numpy())
    return state


def reset(agent, mission, record, max_retries=3):
    '''
    Reset the Minecraft environment, return the world state
    '''
    for retry in range(max_retries):
        try:
            agent.startMission(mission, record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2.5)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)

    print()
    print("Mission running ", end='\n')
    
    return world_state


def reset_map(grid_size, map_path):
    '''
    Returns environment map of obstacles and lava
    '''
    env_map = []
    
    if map_path != None:
        # Read the map in from the text file
        with open(map_path, 'r') as csvfile:
            matrixreader = csv.reader(csvfile, delimiter=' ')
            
            for row in matrixreader:
                env_map.append(row)

        # DEBUGGING - print the map to console
        # for j in range(grid_size):
        #     for i in range(grid_size):
        #         print(env_map[j][i],end=" ")
        #     print()

    return env_map


def setupMinecraft():
    '''
    Setup the Minecraft environment
    NOTE: action space relies heavily on the coordinate system and minecraft has a weird coord system
    '''
    # 0: up, 1: up, 2: down, 3: left, 4: right
    action_space = ["move 0", "move 1", "move -1", "strafe -1", "strafe 1"]

    # Create default Malmo objects:
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse(sys.argv)
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    # Set up the mission
    mission_file = './mission.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    # Force reset of the environment, generate a brand new world every episode
    my_mission.forceWorldReset()

    # Python code for alterations to the environment
    my_mission.drawCuboid(-1, 106, -1, GRID_SIZE, 106, GRID_SIZE, "lava")
    my_mission.drawCuboid(-1, 107, -1, GRID_SIZE, 107, GRID_SIZE, "obsidian")
    # my_mission.drawCuboid(0, 108, 0, GRID_SIZE-1, 110, GRID_SIZE-1, "air") # makes steve move

    # Generating the map
    gridmap = reset_map(GRID_SIZE, MAP_PATH)
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE):
            if gridmap[j][i] == '1':
                my_mission.drawBlock(i, 108, j, "stone")
                my_mission.drawBlock(i, 109, j, "stone")

    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE):
            if gridmap[j][i] == '2':
                my_mission.drawBlock(i, 107, j, "lava")
                pass

    # Placing diamonds on map
    diamond_spots = [(4,6), (0,0), (5,1), (9,2), (7,8), (0,9), (7,4), (8,0), (1,6), (8,6)]

    for d in diamond_spots:
        my_mission.drawItem(d[0], 109, d[1], "diamond")

    # Extra aesthetics
    my_mission.drawBlock(-1, 111, -1, "torch")
    my_mission.drawBlock(-1, 111, GRID_SIZE, "torch")
    my_mission.drawBlock(GRID_SIZE, 111, -1, "torch")
    my_mission.drawBlock(GRID_SIZE, 111, GRID_SIZE, "torch")

    # Idea for another mission
    # my_mission.drawLine(0, 107, 8, 15, 107, 8, "netherrack")
    # my_mission.drawBlock(8, 108, 8, "fire")

    # Can't add a door properly, only adding half a door?
    # my_mission.drawBlock(11, 108, 6, "wooden_door")
    # my_mission.drawBlock(11, 109, 6, "wooden_door")

    # Placing Steve in the map
    x = np.random.randint(0,9) + 0.5
    z = np.random.randint(0,9) + 0.5
    # my_mission.startAt(x, 108, z)
    my_mission.startAt(4.5, 108, 3.5)

    my_mission_record = MalmoPython.MissionRecordSpec()

    print(my_mission.getSummary())

    return agent_host, my_mission, my_mission_record, action_space


def runMission(train = False, load_model = False):
    '''
    Run or train Deep Model
    Training method still needs to be added
    '''

    # Global timer - multi purpose
    start_time = time.time()

    print("\n ---- Running the Deep Q Network ----- \n")

    USE_SAVED_MODEL_FILE = False

    # agent_host, my_mission, my_mission_record, action_space = setupMinecraft()

    model = Network(local_size=LOCAL_GRID_SIZE, name=MODEL_NAME, path="./Models/Tensorflow/"+FOLDER+"/", load=load_model, trainable=train)

    brain = Brain(epsilon=0.1, action_space = 5)

    model.setup(brain)
    
    tf.summary.scalar('error', tf.squeeze(model.error))

    avg_time = 0
    avg_score = 0
    avg_error = 0
    avg_reward = 0
    cumulative_reward = 0

    print_episode = 1
    total_episodes = 10

    # Saving model capabilities
    saver = tf.train.Saver()

    # Initialising all variables (weights and biases)
    init = tf.global_variables_initializer()

    # Adds a summary graph of the error over time
    merged_summary = tf.summary.merge_all()

    # Tensorboard capabilties
    writer = tf.summary.FileWriter(LOGDIR)

    # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

    # Begin Session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Restore the model, to keep training
        if USE_SAVED_MODEL_FILE:
            saver.restore(sess, MODEL_CHECKPOINT)
            print("Model restored.")
        else:
            # Initialize global variables
            sess.run(init)

        # Tensorboard graph
        writer.add_graph(sess.graph)

        print("\nProgram took {0:.4f} seconds to initialise\n".format(time.time()-start_time))
        start_time = time.time()

        # Running mission
        for episode in range(total_episodes):
            agent_host, my_mission, my_mission_record, action_space = setupMinecraft()
            world_state = reset(agent_host, my_mission, my_mission_record)
            score = 0
            done = False
            craft_sword = False

            # Getting first observation
            while True:
                world_state = agent_host.getWorldState()
                if world_state.number_of_observations_since_last_state > 0:
                    break
                
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            # grid = observations.get(u'floor9x9', 0)
            grid = observations.get(u'floor15x15', 0)
            score = observations.get(u'Hotbar_8_size', 0)
            nearby_entites = observations.get(u'nearby_entites', 0)
            diamonds = []
            zombies = []
            steve_pos = (0,0)
            steve_life = 20

            for entity in nearby_entites:
                if entity["name"] == "diamond":
                    diamonds.append((entity["x"], entity["z"]))
                if entity["name"] == "steve":
                    steve_pos = ((entity["x"], entity["z"]))
                    steve_life = entity["life"]
                if entity["name"] == "Zombie":
                    zombies.append((entity["x"], entity["z"]))
            
            state = get_state(steve_pos, diamonds, zombies, grid)

            # brain.linear_epsilon_decay(total_episodes, episode, start=0.3, end=0.05, percentage=0.5)

            world_state = agent_host.getWorldState()
            while world_state.is_mission_running and not done: 
                print("-", end="")
                time.sleep(0.01)

                action = brain.choose_action(state, sess, model)
                # print("action:", action_space[action])

                if craft_sword:
                    agent_host.sendCommand("craft diamond_sword")
                    done = True
                else:
                    agent_host.sendCommand(action_space[action])

                time.sleep(0.2)
                
                world_state = agent_host.getWorldState()

                for error in world_state.errors:
                    print("Error:", error.text)

                # Have we received any observations?
                if world_state.number_of_observations_since_last_state > 0:
                # if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

                    msg = world_state.observations[-1].text
                    observations = json.loads(msg)
                    # print("\n\n", observations, "\n\n")

                    grid = observations.get(u'floor15x15', 0)
                    score = observations.get(u'Hotbar_8_size', 0)
                    nearby_entites = observations.get(u'nearby_entites', 0)
                    diamonds = []
                    zombies = []

                    for entity in nearby_entites:
                        if entity["name"] == "diamond":
                            diamonds.append((entity["x"], entity["z"]))
                        if entity["name"] == "Steve":
                            steve_pos = ((entity["x"], entity["z"]))
                            steve_life = entity["life"]
                        if entity["name"] == "Zombie":
                            zombies.append((entity["x"], entity["z"]))

                    # Debugging - print the state
                    for i in range(6):
                        print(state[i])
                        print()

                    new_state = get_state(steve_pos, diamonds, zombies, grid)

                    # reward = world_state.rewards[-1].getValue()
                    # score += reward

                    # brain.store_transition(state, action, reward, done, new_state)

                    # e, Q_vector = brain.train(model, sess)
                    
                    state = new_state

                    # cumulative_reward += reward

                    # print(score)
                    if score >= 6:
                        craft_sword = True

                    if steve_life != 20:
                        done = True

                    # if done:
                    #     avg_time += info["time"]
                    #     avg_score += info["score"]
                    #     avg_error += e
                    #     avg_reward += cumulative_reward 
                    #     cumulative_reward = 0


            if (episode % print_episode == 0 and episode != 0) or (episode == total_episodes-1):
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
                # model.save(sess)
                # save_path = saver.save(sess, MODEL_PATH_SAVE)

                # s = sess.run(merged_summary, feed_dict={model.input: state, model.actions: Q_vector, score:avg_score/print_episode, avg_t:avg_time/print_episode, epsilon:brain.EPSILON, avg_r:avg_reward/print_episode})
                # writer.add_summary(s, episode)

                avg_time = 0
                avg_score = 0
                avg_error = 0
                avg_reward = 0


        # model.save(sess, verbose=True)

        # save_path = saver.save(sess, MODEL_CHECKPOINT)
        # print("Model saved in path: %s" % save_path)

        writer.close()


def play():
    '''
    Play the game
    '''
    print("\n ----- Playing the game -----\n")

    agent_host, my_mission, my_mission_record, action_space = setupMinecraft()

    world_state = reset(agent_host, my_mission, my_mission_record)

    # DEBUGGING:
    # print("\nList of Commands handlers:")
    # for i in range(len(my_mission.getListOfCommandHandlers(0))):
    #     print(my_mission.getListOfCommandHandlers(0)[i])

    # print("\nAllowed commands:")
    # for action in my_mission.getAllowedCommands(0, "DiscreteMovement"):
    #     print(action)

    # Getting first observation
    while True:
        world_state = agent_host.getWorldState()
        # print(world_state)
        if world_state.number_of_observations_since_last_state > 0:
            break
        
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    # print("\nFirst observation:\n", observations, "\n")
    grid = observations.get(u'floor15x15', 0)
    score = observations.get(u'Hotbar_8_size', 0)
    nearby_entites = observations.get(u'nearby_entites', 0)
    diamonds = []
    zombies = []
    steve_pos = (0,0)
    steve_life = 20

    for entity in nearby_entites:
        if entity["name"] == "diamond":
            diamonds.append((entity["x"], entity["z"]))
        if entity["name"] == "Steve":
            steve_pos = ((entity["x"], entity["z"]))
            steve_life = entity["life"]
        if entity["name"] == "Zombie":
            zombies.append((entity["x"], entity["z"]))
    
    state = get_state(steve_pos, diamonds, zombies, grid)

    # Debugging - print the state
    # for i in range(6):
    #     print(state[i])
    #     print()

    done = False

    world_state = agent_host.getWorldState()
    while world_state.is_mission_running and not done: 
        print("-", end="")
        time.sleep(0.1)

        steve_pos = player_controls(agent_host, steve_pos=steve_pos)
        print(steve_pos)
        time.sleep(0.15)

        world_state = agent_host.getWorldState()

        for error in world_state.errors:
            print("Error:", error.text)

        # Have we received any observations
        # print(world_state.number_of_observations_since_last_state)
        if world_state.number_of_observations_since_last_state > 0:
        # if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            
            # print("\n\n", observations, "\n\n")

            grid = observations.get(u'floor15x15', 0)
            score = observations.get(u'Hotbar_8_size', 0)
            nearby_entites = observations.get(u'nearby_entites', 0)
            diamonds = []
            zombies = []
            # steve_pos = (0,0)
            steve_life = 20

            for entity in nearby_entites:
                if entity["name"] == "diamond":
                    diamonds.append((entity["x"], entity["z"]))
                if entity["name"] == "Steve":
                    steve_pos2 = ((entity["x"], entity["z"]))
                    steve_life = entity["life"]
                if entity["name"] == "Zombie":
                    zombies.append((entity["x"], entity["z"]))

            print(steve_pos2)
            print()

            # Get new_state & reward
            new_state = get_state(steve_pos, diamonds, zombies, grid)

            state = new_state

            if steve_life != 20:
                done = True


def player_controls(agent_host, steve_pos=None):
    '''
    Controls for play to play the game in minecraft with discrete controls
    NOTE: Returns steves position for synchronisation purposes. Not an ideal fix
    '''
    button_delay = 0.01

    # print("Steve pos:", steve_pos)
    sx = steve_pos[0]
    sy = steve_pos[1]

    steve_pos = list(steve_pos)

    char = getch()

    if (char == "q"):
        print("Stop!")
        exit(0)

    if (char == "w"):
        print("Up pressed")
        time.sleep(button_delay)
        # agent_host.sendCommand("move 1")
        agent_host.sendCommand("tp {} 108 {}".format(np.floor(sx) + 0.5, np.floor(sy) + 1.5))
        steve_pos[1] = steve_pos[1]+1

    elif (char == "s"):
        print("Down pressed")
        time.sleep(button_delay)
        # agent_host.sendCommand("move -1")
        agent_host.sendCommand("tp {} 108 {}".format(np.floor(sx) + 0.5, np.floor(sy) - 0.5))
        steve_pos[1] = steve_pos[1]-1

    elif (char == "a"):
        print("Left pressed")
        time.sleep(button_delay)
        # agent_host.sendCommand("strafe -1")
        agent_host.sendCommand("tp {} 108 {}".format(np.floor(sx) + 1.5, np.floor(sy) + 0.5))
        steve_pos[0] = steve_pos[0]+1

    elif (char == "d"):
        print("Right pressed")
        time.sleep(button_delay)
        # agent_host.sendCommand("strafe 1")
        agent_host.sendCommand("tp {} 108 {}".format(np.floor(sx) - 0.5, np.floor(sy) + 0.5))
        steve_pos[0] = steve_pos[0]-1

    elif (char == " "):
        print("Jump pressed")
        time.sleep(button_delay)
        agent_host.sendCommand("jumpmove 1")
        # agent_host.sendCommand("tp 1.5 115 2.5")
        # time.sleep(0.05)
        # agent_host.sendCommand("jump 0")

    elif (char == "i"):
        print("Look Up pressed")
        time.sleep(button_delay)
        agent_host.sendCommand("look -1")

    elif (char == "k"):
        print("Look Down pressed")
        time.sleep(button_delay)
        agent_host.sendCommand("look 1")

    elif (char == "j"):
        print("Look Left pressed")
        time.sleep(button_delay)
        agent_host.sendCommand("turn -1")

    elif (char == "l"):
        print("Look Right pressed")
        time.sleep(button_delay)
        agent_host.sendCommand("turn 1")

    elif (char == "e"):
        print("Use pressed")
        time.sleep(button_delay)
        agent_host.sendCommand("use 1")

    elif (char == "r"):
        print("Break pressed")
        time.sleep(button_delay)
        # agent_host.sendCommand("craft diamond_pickaxe")
        agent_host.sendCommand("attack 1")

    return (steve_pos[0], steve_pos[1])


if __name__ == '__main__':

    runMission(train=False, load_model=True)
    
    # play()
