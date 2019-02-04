import numpy as np
import scipy.io as si

trainZ = np.loadtxt('filtereddata/XYZ_norm.txt',skiprows=0,unpack=False)

predZ = np.loadtxt('filtereddata/predictxyz.txt',skiprows=0,unpack=False)

Zdata = np.hsplit(predZ,np.array([3])) 

Zval = Zdata[0]

yval = Zdata[1]

print(trainZ.shape)

print(Zval.shape)

print(yval.shape)

si.savemat('modeldata2.mat',{'Z':trainZ,'Zval':Zval,'yval':yval})
print("Octave data ready!")
