import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
Data = si.loadmat('result.mat')

Z = Data['Z']

p = Data['p']

# index=0

# for z,c in zip(Z,p):
# 	if c<9.738119e-11:
# 		plt.plot(index,c,color="red")
# 	else:
# 		plt.plot(index,c,color="green")
# 	index+=1
# plt.show()
for index in range(0,len(Z)):
	if(p[index]<9.738119e-191):
		print(index)
		t = np.linspace(0, 2, 64, endpoint=False)
		plt.plot(t, Z[index], 'g-', linewidth=2, label='filtered data')
		plt.xlabel('Time [sec]')
		plt.legend()
		plt.show()
