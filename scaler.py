from sklearn.preprocessing import MinMaxScaler
import numpy as np
from filter import plotdata


latitude,longitude,X,Y,Z,S,HH,MM,T = np.loadtxt('originaldata/readin8.txt',skiprows=2,unpack=True)

scaler = MinMaxScaler(feature_range=(-5,5))

Z = Z.reshape(len(Z),1)

scaler = scaler.fit(Z)

print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))


normalise = scaler.transform(Z)


inversed = scaler.inverse_transform(normalise)


T= HH * 60 * 60 + MM * 60 +T

plotdata(Z,filtered=normalise,fs=32,t=T)

for i in range(100,200):
	print(normalise[i],inversed[i],Z[i])

