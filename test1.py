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

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# import malmo.minecraftbootstrap
# malmo.minecraftbootstrap.set_malmo_xsd_path()

# import random
# import numpy as np
# import tensorflow as tf

if "MALMO_XSD_PATH" not in os.environ:
	os.environ["MALMO_XSD_PATH"] = "/home/matthew/Malmo/Schemas"

if "MALMO_MINECRAFT_ROOT" not in os.environ:
	os.environ["MALMO_MINECRAFT_ROOT"] = "/home/matthew/Malmo/Minecraft"

if "JAVA_HOME" not in os.environ:
	os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"


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

my_mission.drawBlock(0, 110, 0, "stone")

my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

my_mission.drawBlock(0, 110, 0, "stone")

# Loop until mission ends:
while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    if world_state.number_of_observations_since_last_state > 0: # Have any observations come in?
        msg = world_state.observations[-1].text                 # Yes, so get the text
        observations = json.loads(msg)                          # and parse the JSON
        grid = observations.get(u'floor3x3', 0)                 # and get the grid we asked for
        # ADD SOME CODE HERE TO SAVE YOUR AGENT

        if grid[4] == "redstone_block":
        	# agent_host.sendCommand("jump 1")
        	agent_xPos = int(observations.get(u"XPos", 0))
        	agent_yPos = int(observations.get(u"YPos", 0))
        	agent_zPos = int(observations.get(u"ZPos", 0))
        	my_mission.drawBlock(agent_xPos, agent_yPos, agent_zPos, "stone")
        	my_mission.drawBlock(random.randint(0,10), agent_yPos, random.randint(0,10), "redstone_block")
        	
        	

print()
print("Mission ended")
# Mission has ended.
