import numpy as np
import matplotlib.pyplot as plt
import csv


'''
Read in tensorboard csv file, and place data into array
'''

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
with open("Data/run_.-tag-score (meta9).csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	score = []
	episode = []
	for i, row in enumerate(matrixreader):
		if i > 0:
			score.append(float(row[2]))
			episode.append(float(row[1]))

# Epsilon
with open("Data/run_.-tag-epsilon (meta9).csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	epsilon = []
	episode1 = []
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon.append(float(row[2]))
			episode1.append(float(row[1]))


fig, ax1 = plt.subplots()

plt.ylim(0, 2.5)

# color = 'red'
color = 'tab:red'
ax1.set_xlabel('episode')
ax1.set_ylabel('average episode score', color=color)
ax1.plot(episode, score, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x and y axis

# color = 'blue'
color = 'tab:blue'
ax2.set_xlabel('episode')
ax2.set_ylabel('epsilon (\u03B5)', color=color)  # we already handled the x-label with ax1
ax2.plot(episode1, epsilon, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Meta model", fontsize=12)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("Graphs/Meta9.pdf")
plt.show()