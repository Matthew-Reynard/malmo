print("TESTING")

import numpy as np 

a = np.array([[[0,0],
			   [0,0]],
			  [[0,0],
			   [0,0]],
			  [[0,0],
			   [0,0]]])


print(a)

# for row in a[2]:
# 	for col in row:
# 		a[2][col] = 1

a[1] = 1

print(a)
