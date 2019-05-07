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
with open("Data/default9_input5_best_epgreedy/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score.append(float(row[2]))
			episode.append(1+float(row[1]))
			amount_of_episodes = 1+float(row[1])

# Epsilon
with open("Data/default9_input5_best_epgreedy/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon.append(float(row[2]))

# CONTINUE

# with open("Data/meta_local15_input6_2M_unfrozen/run_.-tag-score.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	for i, row in enumerate(matrixreader):
# 		if i > 0:
# 			score.append(float(row[2]))
# 			episode.append(amount_of_episodes + float(row[1]))

# # Epsilon
# with open("Data/meta_local15_input6_2M_unfrozen/run_.-tag-epsilon.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	for i, row in enumerate(matrixreader):
# 		if i > 0:
# 			epsilon.append(float(row[2]))


fig, ax1 = plt.subplots()


# color = 'red'
red1 = 'tab:red'
blue1 = 'tab:blue'
red2 = 'red'
blue2 = 'blue'

ax1.set_ylim(0, 10.0)
ax1.set_xlabel('episode')
ax1.set_ylabel('average score per episode', color=red2)
p1 = ax1.plot(episode, score, color=red2)
# ax1.plot(episode, score, color=blue2)
# ax1.plot(episode, score2, color=red2)
ax1.tick_params(axis='y', labelcolor=red2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x and y axis
ax2.set_ylim(0, 1.05)
ax2.set_yticks(np.linspace(0.05, 0.05, 1), minor=True)

# color = 'blue'
ax2.set_xlabel('episode')
ax2.set_ylabel('epsilon value (\u03B5)', color=blue2)  # we already handled the x-label with ax1
p2 = ax2.plot(episode, epsilon, color=blue2, linestyle='dashed')
# ax2.plot(episode, epsilon2, color=red1, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=blue2)

plt.title("Dojo model (3 dojos, complex env, unfrozen, randomly initialised)", fontsize=12)
plt.legend((p1[0], p2[0]), ("score", "epsilon"), loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.autofmt_xdate()
plt.savefig("Graphs/testing.pdf")
plt.show()
