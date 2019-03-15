import sys, os

if sys.platform == 'linux': 
	import termios, tty

if sys.platform == 'win32': 
    import msvcrt

import numpy as np
import matplotlib.pyplot as plt 
import time
import math


# Used for zombie A* algorithm
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


class Trajectory():
    
    def __init__(self, state, action, reward, new_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done = done


# Just for fun, wanted to implement something similar while training
def print_rotation():
	for i in range(1000):
		print("\r|", end="\r")
		time.sleep(0.1)
		print("\r/", end="\r")
		time.sleep(0.1)
		print("\r-", end="\r")
		time.sleep(0.1)
		print("\r\\", end="")
		time.sleep(0.1)
		print("", end="\b")
		time.sleep(0.01)


def plotLearning(x, scores, epsilons, filename):
	fig=plt.figure()
	ax=fig.add_subplot(111, label="1")
	ax2=fig.add_subplot(111, label="2", frame_on=False)

	ax.plot(x, epsilons, color="C0")
	ax.set_xlabel("Game", color="C0")
	ax.set_ylabel("Epsilon", color="C0")
	ax.tick_params(axis='x', colors="C0")
	ax.tick_params(axis='y', colors="C0")

	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])

	ax2.scatter(x, running_avg, color="C1")
	#ax2.xaxis.tick_top()
	ax2.axes.get_xaxis().set_visible(False)    
	ax2.yaxis.tick_right()
	#ax2.set_xlabel('x label 2', color="C1") 
	ax2.set_ylabel('Score', color="C1")       
	#ax2.xaxis.set_label_position('top') 
	ax2.yaxis.set_label_position('right') 
	#ax2.tick_params(axis='x', colors="C1")
	ax2.tick_params(axis='y', colors="C1")

	plt.savefig(filename)


def createGrid(grid_size, obstacles_array, scale):

	row = []
	grid = []

	for i in range(grid_size):
		row.append(0)

	for i in range(grid_size):
		grid.append(row)

	a = np.array(grid)

	for i in range(len(obstacles_array)):
		a[int(obstacles_array[i][0]/scale)][int(obstacles_array[i][1]/scale)] = 1

	return a


if sys.platform == 'linux': 
	def getch():
	    fd = sys.stdin.fileno()
	    old_settings = termios.tcgetattr(fd)
	    try:
	        tty.setraw(sys.stdin.fileno())
	        ch = sys.stdin.read(1)
	 
	    finally:
	        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	    return ch


if sys.platform == 'win32':
    def getch():
        return msvcrt.getch().decode('UTF-8')


def print_readable_time(current_time):
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


def custom_epsilon(episodes, peaks, scale, end):

    epdivp = int(episodes/peaks)

    t = np.linspace(1, episodes, episodes)
    x = np.zeros(episodes)

    # print(t)

    n = -1

    for i in range(len(t)):
        if i%(epdivp) == 0:
            n += 1

        x[i] = np.exp((peaks*scale*n)-i/(episodes/(peaks*peaks)))
    
    plt.plot(t, x)

    plt.show()
