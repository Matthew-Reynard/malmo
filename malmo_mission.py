#!/usr/bin/env python
# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000

from __future__ import print_function

import os, sys
sys.path.insert(0, os.getcwd()) 

import tensorflow as tf
import numpy as np
from builtins import range
from malmo import MalmoPython
import time
import json
import random
import math
import csv
# import torch
# # from model import DeepQNetwork, Agent
from utils import getch

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


# GLOBAL VARIABLES
# Paths
MODEL_PATH_SAVE = "./Models/Tensorflow/model_0.ckpt"
MODEL_PATH_LOAD = "./Models/Tensorflow/model_0.ckpt"

# Not text files. Some of these arrays are 4D which can't be neatly written to a text file.
W_conv1_textfile_path_save = "./Data/W_conv1.npy"
b_conv1_textfile_path_save = "./Data/b_conv1.npy"

W_conv2_textfile_path_save = "./Data/W_conv2.npy"
b_conv2_textfile_path_save = "./Data/b_conv2.npy"

W_fc_textfile_path_save = "./Data/W_fc.npy"
b_fc_textfile_path_save = "./Data/b_fc.npy"

W_out_textfile_path_save = "./Data/W_out.npy"
b_out_textfile_path_save = "./Data/b_out.npy" 

# This is for viewing the model and summaries in Tensorboard
LOGDIR = "./Logs/log0"

# Parameters
GRID_SIZE = 16
LOCAL_GRID_SIZE = 9 # Has to be an odd number (I think...)
SEED = 1
WRAP = False
FOOD_COUNT = 0
OBSTACLE_COUNT = 0
MAP_PATH = "./Maps/Grid{}/map2.txt".format(GRID_SIZE)

REPLAY_MEMORY = 25000

# Number of hidden layers, nodes, channels, etc. 

n_input_channels = 3 # Using history
# n_input_channels = 4 # Using history

# these still need to be added to the code
n_out_channels_conv1 = 16
n_out_channels_conv2 = 32
n_out_fc = 256

filter1_size = 3
filter2_size = 3

# Number of actions - Up, down, left, right
n_actions = 5

# input - shape is included to minimise unseen errors
x = tf.placeholder(tf.float32, [n_input_channels, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE], name="Input")

# output
y = tf.placeholder(tf.float32, [1, n_actions], name="Output")


# Get the agent state
# NOTE: NEED TO FIX THE DISCREPANCIES BETWEEN THE COORD SYSTEMS
def get_state(grid, steve, diamonds):
    '''Get the state in the same way snake was'''
    LGS = LOCAL_GRID_SIZE

    local_pos = int((LOCAL_GRID_SIZE-1)/2)

    state = np.zeros([3, LGS, LGS], dtype=np.float32)
    
    state[0,int((LGS-1)/2),int((LGS-1)/2)] = 1

    # Diamonds
    for pos in diamonds:
        x_diamond = local_pos+int(pos[0])-int(steve[0])
        y_diamond = local_pos+int(pos[1])-int(steve[1])

        if x_diamond < LOCAL_GRID_SIZE and x_diamond >= 0 and y_diamond < LOCAL_GRID_SIZE and y_diamond >= 0:
            state[1, LGS-1-y_diamond, LGS-1-x_diamond] = 1
    
    # Obstacles
    for i in range(LGS*LGS):
        if grid[i] == "cobblestone" or grid[i] == "air":
            state[2, LGS-1-int(i/LGS), LGS-1-int(i%LGS)] = 1

    # print(state)

    return state
    # needs to be put to cpu before you can convert it to numpy array

    # DEBUGGING
    # print(state.numpy())


# Reset the Minecraft environment, return the world state
def reset(agent, mission, record, max_retries=3):
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
    print("Mission running ", end=' ')
    
    return world_state

# Setup the minecraft environment
def setupMinecraft():
    # Cant stop, can only move
    action_space = ["move 0", "move 1", "move -1", "strafe 1", "strafe -1"]
    #action_space = ["move 0", "move 1", "move -1", "strafe 0", "strafe 1", "strafe -1"]

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
    mission_file = './mission.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)

    # Python code alterations to the environment
    # my_mission.drawBlock(0, 110, 0, "air")

    gridmap = reset_map(GRID_SIZE, MAP_PATH)

    my_mission.drawCuboid(-4, 106, -4, GRID_SIZE+3, 106, GRID_SIZE+3, "lava")
    my_mission.drawCuboid(0, 107, 0, GRID_SIZE-1, 107, GRID_SIZE-1, "stone")
    # my_mission.drawCuboid(0, 108, 0, GRID_SIZE-1, 110, GRID_SIZE-1, "air")


    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE):
            if gridmap[j][i] == '1':
                my_mission.drawBlock(i, 108, j, "stone")
                my_mission.drawBlock(i, 109, j, "stone")

    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE):
            if gridmap[j][i] == '2':
                my_mission.drawBlock(i, 107, j, "air")
                # pass

    # my_mission.drawLine(0, 107, 8, 15, 107, 8, "netherrack")

    my_mission.drawItem(14-5, 109, 14-4, "diamond")
    my_mission.drawItem(14-9, 109, 14-3, "diamond")
    my_mission.drawItem(14-9, 109, 14-7, "diamond")

    my_mission.drawBlock(-1, 109, GRID_SIZE, "torch")
    my_mission.drawBlock(GRID_SIZE, 109, GRID_SIZE, "torch")
    my_mission.drawBlock(-1, 109, -1, "torch")
    my_mission.drawBlock(GRID_SIZE, 109, -1, "torch")

    # my_mission.drawBlock(8, 108, 8, "fire")

    # my_mission.drawBlock(11, 108, 6, "wooden_door")
    # my_mission.drawBlock(11, 109, 6, "wooden_door")

    my_mission_record = MalmoPython.MissionRecordSpec()

    print(my_mission.getSummary())

    return agent_host, my_mission, my_mission_record, action_space

# print(agent_host)

# 2D convolution
def conv2d(x, W, name = None):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding ="VALID", name = name)


# Max pooling
# strides changed from 2 to 1
def maxpool2d(x, name = None):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID', name = name)


# Input -> Conv layer & max pooling -> Conv layer & max pooling -> Fully connected -> Output (Action-value)
def createDeepModel(data, load_variables = False):

    # Model structure
    if load_variables:
        W_conv1 = np.load(W_conv1_textfile_path_save).astype('float32')
        W_conv2 = np.load(W_conv2_textfile_path_save).astype('float32')
        W_fc = np.load(W_fc_textfile_path_save).astype('float32')
        W_out = np.load(W_out_textfile_path_save).astype('float32')

        b_conv1 = np.load(b_conv1_textfile_path_save).astype('float32')
        b_conv2 = np.load(b_conv2_textfile_path_save).astype('float32')
        b_fc = np.load(b_fc_textfile_path_save).astype('float32')
        b_out = np.load(b_out_textfile_path_save).astype('float32')

        weights = {'W_conv1':tf.Variable(W_conv1, name = 'W_conv1'),
                   'W_conv2':tf.Variable(W_conv2, name = 'W_conv2'),
                   'W_fc':tf.Variable(W_fc, name = 'W_fc'),
                   'W_out':tf.Variable(W_out, name = 'W_out')}

        biases = {'b_conv1':tf.Variable(b_conv1, name = 'b_conv1'),
                  'b_conv2':tf.Variable(b_conv2, name = 'b_conv2'),
                  'b_fc':tf.Variable(b_fc, name = 'b_fc'),
                  'b_out':tf.Variable(b_out, name = 'b_out')}
    
    else:
        '''
        Different Initialisations for the weights (the biases can be initialised a constant)

        tf.random_normal([3, 3, n_input_channels, 16], mean=0.0, stddev=1.0, seed=SEED)
        tf.random_uniform([3, 3, n_input_channels, 16], minval=-1, maxval=1, seed=SEED)
        tf.truncated_normal([3, 3, n_input_channels, 16], mean=0, stddev=1.0, seed=SEED)

        use xavier initialiser
        use tf.get_variable()

        '''
        
        weights = {'W_conv1':tf.Variable(tf.truncated_normal([3, 3, n_input_channels, n_out_channels_conv1], mean=0, stddev=1.0, seed=0), name = 'W_conv1'),
                   'W_conv2':tf.Variable(tf.truncated_normal([3, 3, n_out_channels_conv1, n_out_channels_conv2], mean=0, stddev=1.0, seed=1), name = 'W_conv2'),
                   'W_fc':tf.Variable(tf.truncated_normal([4*4*n_out_channels_conv2, n_out_fc], mean=0, stddev=1.0, seed=2), name = 'W_fc'),
                   'W_out':tf.Variable(tf.truncated_normal([n_out_fc, n_actions], mean=0, stddev=1.0, seed=3), name = 'W_out')}

        biases = {'b_conv1':tf.Variable(tf.constant(0.1, shape=[n_out_channels_conv1]), name = 'b_conv1'),
                  'b_conv2':tf.Variable(tf.constant(0.1, shape=[n_out_channels_conv2]), name = 'b_conv2'),
                  'b_fc':tf.Variable(tf.constant(0.1, shape=[n_out_fc]), name = 'b_fc'),
                  'b_out':tf.Variable(tf.constant(0.1, shape=[n_actions]), name = 'b_out')}

    # Model operations
    x = tf.reshape(data, shape=[-1, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE, n_input_channels])

    conv1 = conv2d(x, weights['W_conv1'], name = 'conv1')
    # conv1 = maxpool2d(conv1, name = 'max_pool1')

    conv2 = conv2d(conv1, weights['W_conv2'], name = 'conv2')
    conv2 = maxpool2d(conv2, name = 'max_pool2')

    # print(weights['W_out'].shape)
    fc = tf.reshape(conv2,[-1, 4*4*n_out_channels_conv2])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['W_out']) + biases['b_out']
    output = tf.nn.l2_normalize(output)


    return output, weights, biases


# Run or train Deep Model
def runMission(train = False, load = False):

    # Used to see how long model takes to train - model needs to be optimized!
    start_time = time.time()

    print("\n ---- Running the Deep Neural Network ----- \n")

    # True - Load model from modelpath_load; False - Initialise random weights
    USE_SAVED_MODEL_FILE = False

    # First we need our environment
    agent_host, my_mission, my_mission_record, action_space = setupMinecraft()

    # Hyper-parameters
    alpha = 0.001  # Learning rate, i.e. which fraction of the Q values should be updated
    gamma = 0.99  # Discount factor, i.e. to which extent the algorithm considers possible future rewards
    epsilon = 0.2  # Probability to choose random action instead of best action

    epsilon_function = False
    epsilon_start = 0.3
    epsilon_end = 0.1
    epsilon_percentage = 0.5 # in decimal

    alpha_function = False
    alpha_start = 0.01
    alpha_end = 0.003
    alpha_percentage = 0.9 # in decimal

    # Trajectory
    tau = []

    # Create NN model
    with tf.name_scope('Model'):
        Q_values, weights, biases = createDeepModel(x, load_variables = load)

    # Error / Loss function 
    # reduce_max -> it reduces the [1,4] tensor to a scalar of the max value
    with tf.name_scope('Error'):

        # test
        error = tf.losses.mean_squared_error(labels=Q_values, predictions=y)

        # error = tf.reduce_max(tf.square(tf.subtract(Q_values, y)), axis=1)
        # error = tf.reduce_max(tf.square(Q_values - y), axis=1)
    
    tf.summary.scalar('error', tf.squeeze(error))

    # Gradient descent optimizer - minimizes error/loss function
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(error)
        # optimizer = tf.train.AdamOptimizer(alpha).minimize(error)

    # The next states action-value [1,4] tensor, reduced to a scalar of the max value
    with tf.name_scope('Max_y_prime'):
        y_prime_max = tf.reduce_max(y, axis=1)

    # Action at time t, the index of the max value in the action-value tensor (Made a global variable)
    with tf.name_scope('Max_action'):
        action_t = tf.argmax(y, axis=1)

    avg_time = 0
    avg_score = 0
    avg_error = 0

    # error plot
    # errors = []

    print_episode = 1
    total_episodes = 100000

    # Saving model capabilities
    saver = tf.train.Saver()

    # Initialising all variables (weights and biases)
    init = tf.global_variables_initializer()

    # Adds a summary graph of the error over time
    merged_summary = tf.summary.merge_all()

    # Tensorboard capabilties
    writer = tf.summary.FileWriter(LOGDIR)

    # Session can start running
    with tf.Session() as sess:

        # Restore the model, to keep training
        if USE_SAVED_MODEL_FILE:
            saver.restore(sess, MODEL_PATH_LOAD)
            print("Model restored.")

        # Initialize global variables
        sess.run(init)

        # Tensorboard graph
        writer.add_graph(sess.graph)

        print("\nProgram took {0:.4f} seconds to initialise\n".format(time.time()-start_time))
        start_time = time.time()



        # Training
        for episode in range(total_episodes):
            world_state = reset(agent_host, my_mission, my_mission_record)
            score = 0
            #done = False

            # Getting first obseration
            getting_obs = True
            while getting_obs:
                world_state = agent_host.getWorldState()
                # print(world_state)
                if world_state.number_of_observations_since_last_state > 0:
                    getting_obs = False
                
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floor9x9', 0)
            score = observations.get(u'Hotbar_2_size', 0)
            nearby_entites = observations.get(u'nearby_entites', 0)
            diamonds = []
            steve_pos = (0,0)

            for entity in nearby_entites:
                if entity["name"] == "diamond":
                    diamonds.append((entity["x"], entity["z"]))
                if entity["name"] == "steve":
                    steve_pos = ((entity["x"], entity["z"]))
            state = get_state(grid, steve_pos, diamonds)

            # Linear function for alpha
            if alpha_function:
                alpha = (-alpha_start / (alpha_percentage*total_episodes)) * episode + (alpha_start+alpha_end)
                if alpha < alpha_end: 
                    alpha = alpha_end

            # Linear function for epsilon
            if epsilon_function:
                epsilon = (-(epsilon_start-epsilon_end)/ (epsilon_percentage*total_episodes)) * episode + (epsilon_start)
                if epsilon < epsilon_end: 
                    epsilon = epsilon_end

            world_state = agent_host.getWorldState()
            while world_state.is_mission_running: 
                print("-", end="")
                time.sleep(0.1)
                

                # Retrieve the Q values from the NN in vector form
                # Q_vector = sess.run(Q_values, feed_dict={x: state})
                # print("Qvector", Q_vector) # DEBUGGING

                # Deciding one which action to take
                # if np.random.rand() <= epsilon:
                #     action = np.random.choice(len(action_space))
                # else:
                #     # "action" is the max value of the Q values (output vector of NN)
                #     action = sess.run(action_t, feed_dict={y: Q_vector})
                #     action = action[0]

                # action = np.random.choice(len(action_space))

                # print("action:", action_space[action])
                # agent_host.sendCommand(action_space[action])
                # agent_host.sendCommand("move 1")
                
                world_state = agent_host.getWorldState()

                for error in world_state.errors:
                    print("Error:", error.text)

                # Have we received any observations
                if world_state.number_of_observations_since_last_state > 0:
                # if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

                    msg = world_state.observations[-1].text
                    observations = json.loads(msg)
                    # print("\n\n", observations, "\n\n")

                    grid = observations.get(u'floor9x9', 0)

                    score = observations.get(u'Hotbar_2_size', 0)

                    nearby_entites = observations.get(u'nearby_entites', 0)
                    diamonds = []

                    for entity in nearby_entites:
                        if entity["name"] == "diamond":
                            diamonds.append((entity["x"], entity["z"]))
                        if entity["name"] == "Steve":
                            steve_pos = ((entity["x"], entity["z"]))

                    # print(state)
                    # Get new_state & reward
                    new_state = get_state(grid, steve_pos, diamonds)
                    # reward = world_state.rewards[-1].getValue()
                    # score += reward

                    # Update environment by performing action
                    # new_state, reward, done, info = env.step(action)

                    
                    # IF TRAINING
                    '''
                    ## Standard training with learning after every step

                    # Q_vector = sess.run(Q_values, feed_dict={x: state})
                    # if final state of the episode
                    # print("Q_vector:", Q_vector)
                    if done:
                        Q_vector[:,action] = reward
                        # print("Reward:", reward)
                    else:
                        # Gathering the now current state's action-value vector
                        y_prime = sess.run(Q_values, feed_dict={x: new_state})

                        # Equation for training
                        maxq = sess.run(y_prime_max, feed_dict={y: y_prime})

                        # RL Equation
                        Q_vector[:,action] = reward + (gamma * maxq)

                    _, e = sess.run([optimizer, error], feed_dict={x: state, y: Q_vector})
                    
                    '''

                    '''
                    ## Training using replay memory

                    # Update trajectory (Update replay memory)
                    if len(tau) < REPLAY_MEMORY:
                        tau.append(Trajectory(state, action, reward, new_state, done))
                    else:
                        # print("tau is now full")
                        tau.pop(0)
                        tau.append(Trajectory(state, action, reward, new_state, done))
                    
                    # Choose a random step from the replay memory
                    random_tau = np.random.randint(0, len(tau))

                    # Get the Q vector of the training step
                    Q_vector = sess.run(Q_values, feed_dict={x: tau[random_tau].state})
                    
                    # If terminating state of episode
                    if tau[random_tau].done:
                        # Set the chosen action's current value to the reward value
                        Q_vector[:,tau[random_tau].action] = tau[random_tau].reward
                    else:
                        # Gets the Q vector of the new state
                        y_prime = sess.run(Q_values, feed_dict={x: tau[random_tau].new_state})

                        # Getting the best action value
                        maxq = sess.run(y_prime_max, feed_dict={y: y_prime})

                        # RL DQN Training Equation
                        Q_vector[:,tau[random_tau].action] = tau[random_tau].reward + (gamma * maxq)

                    _, e = sess.run([optimizer, error], feed_dict={x: tau[random_tau].state, y: Q_vector})
                    
                    '''

                    # Add to the error list, to show the plot at the end of training - RAM OVERLOAD!!!
                    # errors.append(e)
                    
                    state = new_state


                    # if done:
                    #     avg_time += info["time"]
                    #     avg_score += info["score"]
                    #     avg_error += e


            if (episode % print_episode == 0 and episode != 0) or (episode == total_episodes-1):
                current_time = math.floor(time.time()-start_time)
                print("Ep:", episode, 
                    "\tavg t: {0:.3f}".format(avg_time/print_episode), 
                    "\tavg score: {0:.3f}".format(avg_score/print_episode), 
                    "\tErr {0:.3f}".format(avg_error/print_episode), 
                    "\tepsilon {0:.3f}".format(epsilon), 
                    #"\ttime {0:.0f}:{1:.0f}".format(current_time/60, current_time%60),
                    end="")
                if current_time%60<10:
                    if math.floor((current_time/60)%60)<10:
                        print("\ttime {0:.0f}:0{1:.0f}:0{2:.0f}".format(math.floor((current_time/60)/60), math.floor((current_time/60)%60), current_time%60))
                    else:
                        print("\ttime {0:.0f}:{1:.0f}:0{2:.0f}".format(math.floor((current_time/60)/60), math.floor((current_time/60)%60), current_time%60))
                else:
                    if math.floor((current_time/60)%60)<10:
                        print("\ttime {0:.0f}:0{1:.0f}:{2:.0f}".format(math.floor((current_time/60)/60), math.floor((current_time/60)%60), current_time%60))
                    else:
                        print("\ttime {0:.0f}:{1:.0f}:{2:.0f}".format(math.floor((current_time/60)/60), math.floor((current_time/60)%60), current_time%60))
                avg_time = 0
                avg_score = 0
                avg_error = 0

                # Save the model's weights and biases to .npy files (can't save 4D array to text file)
                W_conv1 = np.array(sess.run(weights['W_conv1']))
                W_conv2 = np.array(sess.run(weights['W_conv2']))
                W_fc = np.array(sess.run(weights['W_fc']))
                W_out = np.array(sess.run(weights['W_out']))

                b_conv1 = np.array(sess.run(biases['b_conv1']))
                b_conv2 = np.array(sess.run(biases['b_conv2']))
                b_fc = np.array(sess.run(biases['b_fc']))
                b_out = np.array(sess.run(biases['b_out']))

                np.save(W_conv1_textfile_path_save, W_conv1.astype(np.float32))
                np.save(W_conv2_textfile_path_save, W_conv2.astype(np.float32))
                np.save(W_fc_textfile_path_save, W_fc.astype(np.float32))
                np.save(W_out_textfile_path_save, W_out.astype(np.float32))

                np.save(b_conv1_textfile_path_save, b_conv1.astype(np.float32))
                np.save(b_conv2_textfile_path_save, b_conv2.astype(np.float32))
                np.save(b_fc_textfile_path_save, b_fc.astype(np.float32))
                np.save(b_out_textfile_path_save, b_out.astype(np.float32))

                s = sess.run(merged_summary, feed_dict={x: state, y: Q_vector})
                writer.add_summary(s, episode)

        save_path = saver.save(sess, MODEL_PATH_SAVE)
        print("Model saved in path: %s" % save_path)

    # DISPLAY THE ERROR GRAPH
    # plt.plot([np.mean(errors[i:i+500]) for i in range(len(errors) - 500)])
    # plt.savefig("./Images/errors.png")
    # plt.show()


# Play the game
def play():
    print("\n ----- Playing the game -----\n")

    agent_host, my_mission, my_mission_record, action_space = setupMinecraft()

    world_state = reset(agent_host, my_mission, my_mission_record)

    # Getting first obseration
    getting_obs = True
    while getting_obs:
        world_state = agent_host.getWorldState()
        # print(world_state)
        if world_state.number_of_observations_since_last_state > 0:
            getting_obs = False
        
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'floor9x9', 0)
    score = observations.get(u'Hotbar_2_size', 0)
    nearby_entites = observations.get(u'nearby_entites', 0)
    diamonds = []
    steve_pos = (0,0)

    for entity in nearby_entites:
        if entity["name"] == "diamond":
            diamonds.append((entity["x"], entity["z"]))
        if entity["name"] == "steve":
            steve_pos = ((entity["x"], entity["z"]))
    state = get_state(grid, steve_pos, diamonds)

    button_delay = 0.01

    # done = False

    world_state = agent_host.getWorldState()
    while world_state.is_mission_running:# and not done: 
        print("-", end="")
        time.sleep(0.05)

        char = getch()

        if (char == "p"):
            print("Stop!")
            exit(0)

        if (char == "w"):
            print("Up pressed")
            time.sleep(button_delay)
            agent_host.sendCommand("move 1")

        elif (char == "s"):
            print("Down pressed")
            time.sleep(button_delay)
            agent_host.sendCommand("move -1")

        elif (char == "a"):
            print("Left pressed")
            time.sleep(button_delay)
            agent_host.sendCommand("strafe -1")

        elif (char == "d"):
            print("Right pressed")
            time.sleep(button_delay)
            agent_host.sendCommand("strafe 1")

        elif (char == " "):
            print("Jump pressed")
            time.sleep(button_delay)
            agent_host.sendCommand("jumpmove 1")
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

        

        world_state = agent_host.getWorldState()

        for error in world_state.errors:
            print("Error:", error.text)

        # Have we received any observations
        if world_state.number_of_observations_since_last_state > 0:
        # if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            # print("\n\n", observations, "\n\n")

            grid = observations.get(u'floor9x9', 0)

            score = observations.get(u'Hotbar_2_size', 0)

            nearby_entites = observations.get(u'nearby_entites', 0)
            diamonds = []

            for entity in nearby_entites:
                if entity["name"] == "diamond":
                    diamonds.append((entity["x"], entity["z"]))
                if entity["name"] == "Steve":
                    steve_pos = ((entity["x"], entity["z"]))

            # print(state)
            # Get new_state & reward
            new_state = get_state(grid, steve_pos, diamonds)

            state = new_state


def reset_map(grid_size, map_path):
    # self.array.clear()

    map1 = []
    
    # Read the map in from the text file
    with open(map_path, 'r') as csvfile:
        matrixreader = csv.reader(csvfile, delimiter=' ')
        
        for row in matrixreader:
            map1.append(row)

    # for j in range(grid_size):
    #     for i in range(grid_size):
    #         print(map1[j][i],end=" ")
    #     print()

    return map1

# # Load model
# try:
#     path = "./Models/Torch/my_model.pth"
#     brain.load_model(path)
#     print("Model loaded from path:", path)
# except Exception:
#     print('Could not load model, continue with random initialision (y/n):')
#     input()
#     # quit()


# INITIALISING REPLAY MEMORY

# while brain.memCntr < brain.memSize:
#     world_state = reset(agent_host)

#     getting_obs = True
#     while getting_obs:
#         world_state = agent_host.getWorldState()
#         # print(world_state)
#         if world_state.number_of_observations_since_last_state > 0:
#             getting_obs = False
        
#     msg = world_state.observations[-1].text
#     observations = json.loads(msg)
#     grid = observations.get(u'floor9x9', 0)
#     state = get_state(grid)

#     while world_state.is_mission_running: 
#         print("-", end="")
#         time.sleep(0.1)

#         # action based on current state
#         # print(state)
#         # action_taken = np.random.choice(len(action_space))
#         # agent_host.sendCommand(action_space[action_taken])
#         agent_host.sendCommand("move 0") # Send a nothing command for testing

#         # world_state = agent_host.getWorldState()
#         world_state = agent_host.peekWorldState()

#         for error in world_state.errors:
#             print("Error:", error.text)

#         # Have we received any observations
#         if world_state.number_of_observations_since_last_state > 0:
#         # if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

#             msg = world_state.observations[-1].text
#             observations = json.loads(msg)
#             grid = observations.get(u'floor9x9', 0)

#             print("\n\n\n", observations, "\n\n\n")

#             # Print normally
#             # for i in range(9):
#             #     for j in range(9): 
#             #         print("{0:12s}".format(grid[j+(i*9)]), end=" ")
#             #     print("\n")

#             # Print in easier to see direction
#             for i in range(8, -1, -1):
#                 for j in range(8, -1, -1): 
#                     print("{0:12s}".format(grid[j+(i*9)]), end=" ")
#                 print("\n")

#             # Get new_state & reward
#             new_state = get_state(grid)
#             # reward = world_state.rewards[-1].getValue()
            
#             # Store the transitions
#             # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#             # brain.storeTransition(state, torch.Tensor(action_taken, dtype=torch.int32).to(device), torch.Tensor(reward,dtype=torch.float32).to(device), new_state)
#             # brain.storeTransition(state, action_taken, reward, False, new_state)

#             # Not neccesrily true, might be a few observations after
#             state = new_state

# print("Done initialising memory")


# # scores = []
# epsHistory = []
# num_episodes = 0 # Not training if 0
# batch_size = 1

# avg_score = 0
# avg_loss = 0

# # TRAINING
# for i in range(num_episodes):

#     epsHistory.append(brain.EPSILON)
#     score = 0

#     # reset env
#     world_state = reset(agent_host)
#     # print(world_state)

#     # for i in range(len(my_mission.getListOfCommandHandlers(0))):
#     #     print(my_mission.getListOfCommandHandlers(0)[i])

#     # for action in my_mission.getAllowedCommands(0,"DiscreteMovement"):
#     #     print(action)


#     # Loop until mission ends
#     while world_state.is_mission_running: 
#         print("-", end="")
#         time.sleep(0.1)

#         # action based on current state
#         # print(state)
#         # action_taken = np.random.choice(len(action_space))
#         # agent_host.sendCommand(action_space[action_taken])

#         action = brain.chooseAction(state)
#         agent_host.sendCommand(action_space[action])

#         world_state = agent_host.getWorldState()

#         for error in world_state.errors:
#             print("Error:", error.text)

#         # Have we received any observations
#         if world_state.number_of_observations_since_last_state > 0 and world_state.number_of_rewards_since_last_state > 0:

#             msg = world_state.observations[-1].text
#             observations = json.loads(msg)
#             grid = observations.get(u'floor9x9', 0)
            
#             # Get new_state & reward
#             new_state = get_state(grid)
#             reward = world_state.rewards[-1].getValue()
#             score += reward
            
#             # Store the transitions
#             # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#             # brain.storeTransition(state, torch.Tensor(action_taken, dtype=torch.int32).to(device), torch.Tensor(reward,dtype=torch.float32).to(device), new_state)
#             brain.storeTransition(state, action, reward, new_state)

#             # Not neccesrily true, might be a few observations after
#             state = new_state

#             loss = brain.learn(batch_size)

#             avg_score += score
#             avg_loss += loss.item()

#     print("Episode", i, 
#         "\tepsilon: %.4f" %brain.EPSILON,
#         "\tavg score", avg_score/1,
#         "avg loss:", avg_loss/1)

#     avg_score = 0
#     avg_loss = 0
            	
#     print()	
#     print("Mission {} ended".format(i+1))
#     # Mission has ended.

# print()
# print("Done")




if __name__ == '__main__':

    # runMission(train = False, load = False)
    
    play()

    # reset_map(16, "./Maps/Grid16/map3.txt")
