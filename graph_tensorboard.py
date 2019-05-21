import numpy as np
import matplotlib.pyplot as plt
import csv

'''
Read in tensorboard csv file, and place data into array
'''

score = []
epsilon = []
episode = []
moves = []

score2 = []
epsilon2 = []
episode2 = []
moves2 = []

DATA_FOLDER = "default15_input6_100k"
DATA_FOLDER_cont = "default15_input6_200k"
DATA_FOLDER_cont_again = "default15_input6_300k"

ADDITIONAL_DATA_FOLDER = "meta15_input6_unfrozen_100k_cointoss"
ADDITIONAL_DATA_FOLDER_cont = "meta15_input6_unfrozen_300k_cointoss"
# ADDITIONAL_DATA_FOLDER_cont_again = "meta15_input6_4_unfrozen_300k"

# Avg score
with open("Data/"+DATA_FOLDER+"/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score.append(float(row[2]))
			episode.append(1+float(row[1]))
			amount_of_episodes = 1+float(row[1])

# Epsilon
with open("Data/"+DATA_FOLDER+"/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon.append(float(row[2]))

# Average time
with open("Data/"+DATA_FOLDER+"/run_.-tag-average_time.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			moves.append(float(row[2]))


############################# CONTINUE

with open("Data/"+DATA_FOLDER_cont+"/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score.append(float(row[2]))
			episode.append(amount_of_episodes + float(row[1]))
			amount_of_episodes_ = amount_of_episodes + float(row[1])
# Epsilon
with open("Data/"+DATA_FOLDER_cont+"/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon.append(float(row[2]))

# Average time
with open("Data/"+DATA_FOLDER_cont+"/run_.-tag-average_time.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			moves.append(float(row[2]))

############################# CONTINUE again

with open("Data/"+DATA_FOLDER_cont_again+"/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score.append(float(row[2]))
			episode.append(amount_of_episodes_ + float(row[1]))

# Epsilon
with open("Data/"+DATA_FOLDER_cont_again+"/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon.append(float(row[2]))

# Average time
with open("Data/"+DATA_FOLDER_cont_again+"/run_.-tag-average_time.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			moves.append(float(row[2]))


############################ ADDITIONAL GRAPH ####################################

with open("Data/"+ADDITIONAL_DATA_FOLDER+"/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score2.append(float(row[2]))
			episode2.append(1+float(row[1]))
			amount_of_episodes2 = 1+float(row[1])

# Epsilon
with open("Data/"+ADDITIONAL_DATA_FOLDER+"/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon2.append(float(row[2]))

with open("Data/"+ADDITIONAL_DATA_FOLDER+"/run_.-tag-average_time.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			moves2.append(float(row[2]))

############################# ADDITIONAL CONTINUE

with open("Data/"+ADDITIONAL_DATA_FOLDER_cont+"/run_.-tag-score.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			score2.append(float(row[2]))
			episode2.append(amount_of_episodes2 + float(row[1]))
			amount_of_episodes2_ = amount_of_episodes2+float(row[1])

# Epsilon
with open("Data/"+ADDITIONAL_DATA_FOLDER_cont+"/run_.-tag-epsilon.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			epsilon2.append(float(row[2]))

# Average time
with open("Data/"+ADDITIONAL_DATA_FOLDER_cont+"/run_.-tag-average_time.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	for i, row in enumerate(matrixreader):
		if i > 0:
			moves2.append(float(row[2]))

############################# ADDITIONAL CONTINUE again

# with open("Data/"+ADDITIONAL_DATA_FOLDER_cont_again+"/run_.-tag-score.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	for i, row in enumerate(matrixreader):
# 		if i > 0:
# 			score2.append(float(row[2]))
# 			episode2.append(amount_of_episodes2_ + float(row[1]))

# # Epsilon
# with open("Data/"+ADDITIONAL_DATA_FOLDER_cont_again+"/run_.-tag-epsilon.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	for i, row in enumerate(matrixreader):
# 		if i > 0:
# 			epsilon2.append(float(row[2]))

# # Average time
# with open("Data/"+ADDITIONAL_DATA_FOLDER_cont_again+"/run_.-tag-average_time.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	for i, row in enumerate(matrixreader):
# 		if i > 0:
# 			moves2.append(float(row[2]))


# print(len(score))
# print(len(score2))
# print(len(episode))
# print(len(episode2))


''' (PRESS CTRL + / to change between the graphs)

########################## PLOT INDIVIDUAL GRAPH #####################

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
ax2.set_ylim(0, 100.05)
ax2.set_yticks(np.linspace(0.05, 0.05, 1), minor=True)

# color = 'blue'
ax2.set_xlabel('episode')
# ax2.set_ylabel('epsilon value (\u03B5)', color=blue2)  # we already handled the x-label with ax1
# p2 = ax2.plot(episode, epsilon, color=blue2, linestyle='dashed')
ax2.set_ylabel('average number of moves per episode', color=blue2)  # we already handled the x-label with ax1
p2 = ax2.plot(episode, moves, color=blue2, linestyle='dashed')
# ax2.plot(episode, epsilon2, color=red1, linestyle='dashed')
ax2.tick_params(axis='y', labelcolor=blue2)

plt.title("Dojo Network (complex env, cointoss experiment)", fontsize=12)
# plt.legend((p1[0], p2[0]), ("score", "epsilon"), loc="upper right")
plt.legend((p1[0], p2[0]), ("score", "moves"), loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.autofmt_xdate()
plt.savefig("Graphs/meta15_input6_cointoss_300k.pdf")
plt.show()


'''

############################## PLOT COMPARISON GRAPH############################

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
p2 = ax1.plot(episode, score2, color=blue2, linestyle='dashed')
ax1.tick_params(axis='y', labelcolor=red2)

ax1.set_xlabel('episode')

plt.title("Standard and Dojo (cointoss) Networks (complex env, unfrozen)", fontsize=12)

plt.legend((p1[0], p2[0]), ("Standard", "Dojo cointoss"), loc="upper right")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
# fig.autofmt_xdate()
plt.savefig("Graphs/Comparison_300k_cointoss_vs_standard.pdf")
plt.show()

# '''