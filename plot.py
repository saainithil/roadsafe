import numpy as np
import scipy.io as si
import progressbar
import re
from filter import butter_lowpass_filter,plotdata
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import norm,multivariate_normal


def normalize(Z,min=-3,max=3):

	scaler = MinMaxScaler(feature_range=(min,max))

	Z = Z.reshape(len(Z),1)

	scaler = scaler.fit(Z)

	print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))


	normalise = scaler.transform(Z)

	return normalise


		
def datafeeder(Z,window):
	#numpy array initialisation to hold the Z - axis data. 
	Zdata = np.array([])
	#Code for progress bar
	bar =progressbar.ProgressBar(maxval=len(Z),widgets=[progressbar.Bar('*','[',']'),' ',progressbar.Percentage()])
	bar.start()

	for index in range(0,len(Z)-window+1):
		#splitting the Z axis data into 64 taps each
		data=np.array(Z[index:index+window])
		
		#Finding maximum, minimum and average so that the Z axis readings are normalised using speed. 
		# maximum = np.amax(data)
		# minimum = np.amin(data)

		# average = np.average(data)

		# data = (data - average) / (maximum -minimum)

		#data = np.around(data,decimals=1)
		
		#original = data
		
		# Function call to lowpass filter
		#data = butter_lowpass_filter(data, cutoff=5,fs= 32,order= 6)
		
		#plotdata(original,filtered=data,fs=32,t=T[index:index+80]+MM[index:index+80]*60)
		
		#append each 64 taps to Zdata array
		Zdata=np.append(Zdata,data)
		
		Zdata=np.append(Zdata,longitude[index])
		Zdata=np.append(Zdata,latitude[index])
		
		#update progress bar by 1 every in iteration
		bar.update(index+1)
	bar.finish()
	# Reshape the Zdata array into equal sized 66(64 Zaxis readings + latitude and longitude) column rows each
	Zdata = (Zdata.reshape(index+1,98))



	Zdata = np.hsplit(Zdata,np.array([96])) 

	Z = Zdata[0]

	# Location = Zdata[1]


	print(Z.shape)
	#Save the filtered data to a text file.
	np.savetxt('filtereddata/predictdata5.txt',Z,fmt='%1.15g')

	#Save the filtered data to a mat file
	#si.savemat('modeldata1.mat',{'Z':Z})
	print("Data written to File.")



latitude,longitude,X,Y,Z,S,HH,MM,T = np.loadtxt('originaldata/bump12.txt',skiprows=2,unpack=True)

S = np.around(Z,decimals=2) 



original = Z



T= HH * 60 * 60 + MM * 60 +T


X = butter_lowpass_filter(X, cutoff=5,fs= 32,order= 6)

Y = butter_lowpass_filter(Y, cutoff=5,fs= 32,order= 6)

Z = butter_lowpass_filter(Z, cutoff=5,fs= 32,order= 6)

#plotdata(original,filtered=Z,fs=32,t=T)
D = np.column_stack((X,Y,Z))
X_norm = normalize(X,-2,2)

Y_norm = normalize(Y,-2,2)

Z_norm = normalize(Z,-2,2)

delta = norm.logpdf(Z_norm,loc=np.mean(Z_norm, dtype=np.float64,axis=0),scale=np.std(Z_norm, dtype=np.float64,axis=0))
delta3 =multivariate_normal.logpdf(D,mean=np.mean(D, dtype=np.float64,axis=0),cov=np.std(D, dtype=np.float64,axis=0))

XYZ_norm = np.column_stack((X_norm,Y_norm,Z_norm))

#xnp.savetxt('filtereddata/XYZ_norm1.txt',XYZ_norm,fmt='%1.15g')

# for index in range(0,len(Z)):
# 	if(delta[index]<-5):
# 		print(index)
# 		t = np.linspace(0, 2, 64, endpoint=False)
# 		plt.plot(t, Z[index-32:index+32], 'g-', linewidth=2, label='filtered data')
# 		plt.xlabel('Time [sec]')
# 		plt.legend()
# 		plt.show()

		
plotdata(delta3,XYZ_norm,fs=32,t=T)


#datafeeder(Z_norm,96)

