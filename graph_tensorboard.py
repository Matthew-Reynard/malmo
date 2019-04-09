import numpy as np
import matplotlib.pyplot as plt
import csv


'''
Read in tensorboard csv file, and place data into array
'''

score = []
epsilon = []
episode = []

score2 = []
epsilon2 = []
episode2 = []

# Avg time
# with open("Data/run_.-tag-average_time.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	score = []
# 	episode = []
# 	for i, row in enumerate(matrixreader):
# 		if i > 0:
# 			score.append(float(row[2]))
# 			episode.append(float(row[1]))

# Avg score
with open("Data/meta9_1m_epsilon/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score.append(float(row[2]))
			episode.append(1+float(row[1]))
			amount_of_episodes = 1+float(row[1])

# Epsilon
with open("Data/meta9_1m_epsilon/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon.append(float(row[2]))

# CONTINUE

with open("Data/Meta_9_1M/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score2.append(float(row[2]))
			episode2.append(amount_of_episodes + float(row[1]))

# Epsilon
with open("Data/Meta_9_1M/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon2.append(float(row[2]))


fig, ax1 = plt.subplots()

plt.ylim(0, 3.0)

# color = 'red'
red1 = 'tab:red'
blue1 = 'tab:blue'
red2 = 'red'
blue2 = 'blue'

ax1.set_xlabel('episode')
ax1.set_ylabel('average episode score', color=blue2)
ax1.plot(episode, score, color=blue2)
ax1.plot(episode, score2, color=red2)
ax1.tick_params(axis='y', labelcolor=blue2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x and y axis

# color = 'blue'
ax2.set_xlabel('episode')
ax2.set_ylabel('epsilon (\u03B5)', color=red2)  # we already handled the x-label with ax1
ax2.plot(episode, epsilon, color=blue1, linestyle='dashed')
ax2.plot(episode, epsilon2, color=red1, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=red2)

plt.title("Meta model", fontsize=12)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("Graphs/meta9_1m_both.pdf")
plt.show()
