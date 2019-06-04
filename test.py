print("TESTING")

import numpy as np

from math import pi
import sys
from utils import custom_epsilon
import matplotlib.pyplot as plt
# import getch
import csv

# s = np.array([[[0,0],
#       [0,0]],
#      [[1,1],
#       [1,1]],
#      [[2,2],
#       [2,2]]])

# s[0] = 1

# print(s)

# t = []

# size = 10

# cntr = 0

# for i in range(100):
# 	print(t)
# 	if cntr < size:
# 		t.append(i**2)
# 	else:
# 		t[cntr%size] = i**2
# 	print(t[cntr%size])
# 	cntr = cntr + 1

# 	print(t[cntr%size-1])
	
# 	print()

# b = [1,3,0]

# a = [[10,20],[30,40],[50,60],[70,80],[90,0]]
# c = []
# for each in b:
# 	c.append(a[each])
# # c = a[b]

# print(c)

# print(np.random.choice(a[:], 5, replace=False))







# custom_epsilon(100000, 2, 0.8, 0.05)
# start = 1
# end = 0.00
# percentage = 1
# total = 1000

# for episode in range(total+1):

#     e = (-(start-end)/ (percentage*total)) * episode + (start)
        
#     if e < end: 
#         e = end

#     print(e)


# a = np.array([[[0,0],
#              [0,0]],
#             [[0,0],
#              [0,0]],
#             [[0,0],
#              [0,0]]])


# print(a)

# for row in a[2]:
#   for col in row:
#       a[2][col] = 1

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
# #     print(np.exp((x*1.41)-grid))

# for x in range(grid):
#   print(np.tan(x))

# print(np.tan(pi/4))

# y = [0,1,2,3,4,5,6,7,8,9]

# z = list(map(lambda x: ((x/grid)*(pi/2))-pi/4, y))

# print(z)

# for x in z:
#   print(np.tan(x))

# def test():

# state = np.zeros((4,3,3))

# state[1] = 1
# state[2] = 2
# state[3] = 3
# state = np.delete(state, 1, 0)

# print(state)



# with open("test.csv", 'r') as csvfile:
# 	matrixreader = csv.reader(csvfile, delimiter=',')
# 	a = []
# 	b = []
# 	for i, row in enumerate(matrixreader):
# 		# a.append(row)
# 		if i > 0:
# 			print(row[2])
# 			print()
# 			a.append(float(row[2]))
# 			b.append(float(row[1]))



# print(a)

# plt.plot(b,a)
# plt.show()

# import tensorflow as tf

# x = tf.placeholder(tf.float32, [])

# # output
# y = tf.placeholder(tf.float32, [4, ])

# # a=tf.Variable(1.0)

# b=tf.Variable(0.0)

# # error = tf.losses.mean_squared_error(labels=a, predictions=b)

# tf.summary.scalar('a', y)

# # tf.summary.scalar('b', b)

# # merged_summary = tf.summary.merge_all()
# init = tf.global_variables_initializer()
# # writer = tf.summary.FileWriter("./Logs/test/")

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# start = 0.5
# end = 0.05
# percentage = 0.5
# total = 100
#     # Begin session
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     sess.run(init)

#     # writer.add_graph(sess.graph)
#     a = 0

#     actions = [0.1, 1.8, 0.1, 0.0]

#     a = tf.nn.l2_normalize(actions)

#     a_ = sess.run(a, feed_dict={y: actions})

#     print(a_)
    # for episode in range(total):

    	# a = tf.nn.softmax()

        # a = (-(start-end)/ (percentage*total)) * episode + (start)
        # if a < end:
        #     a = end
        # b = b + 2
        # s = sess.run(merged_summary, feed_dict={y: a})
        # writer.add_summary(s, episode)

    # writer.close()