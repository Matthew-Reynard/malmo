print("TESTING")

import numpy as np

from math import pi
import sys
from utils import custom_epsilon
import matplotlib.pyplot as plt
# import getch
import csv

# a = np.array([[[0,0],
# 			   [0,0]],
# 			  [[0,0],
# 			   [0,0]],
# 			  [[0,0],
# 			   [0,0]]])


# print(a)

# for row in a[2]:
# 	for col in row:
# 		a[2][col] = 1

# a[1] = 1

# print(a)

# b = [3,0,1,2,3,4,5,6,7]
# b.append(9)
# b.pop(0)
# print(b)
# memSize = 10
# memCntr = 0
# memory=[]

# for i in range(20):
#     if memCntr < memSize:
#         memory.append([0,memCntr])
#     else:
#         memory[memCntr%memSize] = [0,memCntr]
#     memCntr += 1
#     print(memory)

# from my_mission import missionXML

# print(missionXML)

# myfile = open("mission2.xml", "w")  
# myfile.write(mydata)  

# grid = 9


# # for x in range(grid):
# # 	print(np.exp((x*1.41)-grid))

# for x in range(grid):
# 	print(np.tan(x))

# print(np.tan(pi/4))

# y = [0,1,2,3,4,5,6,7,8,9]

# z = list(map(lambda x: ((x/grid)*(pi/2))-pi/4, y))

# print(z)

# for x in z:
# 	print(np.tan(x))

# def test():

# state = np.zeros((4,3,3))

# state[1] = 1
# state[2] = 2
# state[3] = 3
# state = np.delete(state, 1, 0)

# print(state)


# custom_epsilon(1000, 5)

with open("test.csv", 'r') as csvfile:
	matrixreader = csv.reader(csvfile, delimiter=',')
	a = []
	b = []
	for i, row in enumerate(matrixreader):
		# a.append(row)
		if i > 0:
			print(row[2])
			print()
			a.append(float(row[2]))
			b.append(float(row[1]))



print(a)

plt.plot(b,a)
plt.show()

