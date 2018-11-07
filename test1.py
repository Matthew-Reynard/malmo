#!/usr/bin/env python
# $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000

# MarLo-CliffWalking-v0
# MarLo-FindTheGoal-v0

import malmo.minecraftbootstrap
# import random
# import time
# import numpy as np
# import tensorflow as tf
# import time # Used to measure how long training takes
import os

# if "MALMO_XSD_PATH" not in os.environ:
os.environ["MALMO_XSD_PATH"] = "/home/matthew/Malmo/Schemas"

# if "MALMO_MINECRAFT_ROOT" not in os.environ:
os.environ["MALMO_MINECRAFT_ROOT"] = "/home/matthew/Malmo/Minecraft"

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

# X_ = -4
# Y_ = 0
# Z_ = -4
# X = 4
# Y = 0
# Z = 4

# observe_grid = [X_,Y_,Z_,X,Y,Z]
# client_pool = [('127.0.0.1', 10000)]

# help(malmo)

# malmo.run_mission.run()

# join_tokens = marlo.make('MarLo-FindTheGoal-v0',
# # join_tokens = marlo.make("./templates/mission.xml",
#                           params={
#                             "client_pool": client_pool,
#                             "tick_length": 20,
#                             "observeGrid": observe_grid,
# 	                          "kill_clients_after_num_rounds": 150
#                           })
# # As this is a single agent scenario,
# # there will just be a single token
# assert len(join_tokens) == 1
# join_token = join_tokens[0]

# env = marlo.init(join_token)

# print("action:", env.action_space)
# print("action:", env.action_names)

# observation = env.reset()

# done = False

# while not done:
#   _action = env.action_space.sample()
#   obs, reward, done, info = env.step(_action)
#   print("reward:", reward)
#   print("done:", done)
#   print("info", info)
# env.close()