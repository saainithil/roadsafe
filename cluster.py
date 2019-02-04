from sklearn.cluster import KMeans,DBSCAN
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
X, Y, Z = np.loadtxt('filtereddata/XYZ_norm1.txt',skiprows=0,unpack=True)
Data = np.column_stack((X,Y,Z))
#Z = np.hsplit(Data,np.array([80])) 

#kmeans =KMeans(n_clusters=2,random_state=42,max_iter=10000,n_init=100).fit(Z[0])

# epsilon = np.arange(0.1,2,0.025)

# for e in epsilon:

# 	dbscan = DBSCAN(eps=e,algorithm='kd_tree', min_samples=2).fit(Data)
# 	print(e)
# 	print(dbscan.labels_)

dbscan = DBSCAN(eps = 0.874,algorithm='kd_tree',min_samples=2).fit(Data)
print(dbscan.labels_)
print(Data.shape)



# print(Z)
# pred = Data[1]


pred = np.column_stack((Data,dbscan.labels_))
print(pred.shape)


for index in range(0,len(Z)):
	if(dbscan.labels_[index]<0):
		print(index)
		t = np.linspace(0, 2, 64, endpoint=False)
		plt.plot(t, Z[index-32:index+32], 'g-', linewidth=2, label='filtered data')
		plt.xlabel('Time [sec]')
		plt.legend()
		plt.show()


#np.savetxt('filtereddata/predictxyz1.txt',pred,fmt='%1.15g')

print("Prediction data ready")