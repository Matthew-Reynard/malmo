print("TESTING")

import numpy as np 

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

from my_mission import missionXML

print(missionXML)

# myfile = open("mission2.xml", "w")  
# myfile.write(mydata)  