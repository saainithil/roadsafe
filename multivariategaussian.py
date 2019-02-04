import numpy as np
import scipy.io as si
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import progressbar

def gaussian_variate(X,mu,sigma2):
	
	if(sigma2.ndim==1):
		sigma2 = np.diag(sigma2)


	pval = multivariate_normal.logpdf(X,mean=mu,cov=sigma2)

	return pval

def estimate_gaussian(X):
	m , n = X.shape

	mu = (np.sum(X,axis=0)/m) 
	#mu = np.mean(X, dtype=np.float64,axis=0)
	

	sigma2 = np.sum(np.square(np.subtract(X,mu)),axis=0)/m
	#sigma2 = np.std(X, dtype=np.float64,axis=0)
	
	return mu, sigma2


def load_data():
	Data = si.loadmat('modeldata2.mat')
	Z = Data['Z']
	Zval = Data['Zval']
	yval = Data['yval']
	yval=yval.reshape(len(yval),)
	
	return Z,Zval,yval

def select_threshold(yval,pval):
	bestepsilon = 0
	bestF1 = 0
	F1 = 0
	
	step = np.linspace(max(pval),min(pval),100000,endpoint=True,retstep=False)
	
	yval=yval<0
	
	index=0
	bar =progressbar.ProgressBar(maxval=len(step),widgets=[progressbar.Bar('*','[',']'),' ',progressbar.Percentage()])
	bar.start()

	for epsilon in step:
		prediction = pval<epsilon
		tp = np.sum(np.logical_and(prediction==True,yval==True))
		fp = np.sum(np.logical_and(prediction==True,yval==False))
		fn = np.sum(np.logical_and(prediction==False,yval==True))
		precision=0
		recall=0
		if(tp+fp):
			precision =  tp / ( tp + fp )
		if(tp+fn):	 
			recall =  tp / ( tp + fn )
		F1=0
		if(precision+recall):
			F1 = 2 * precision * recall / (precision + recall)

		if(F1>bestF1):
			bestF1=F1
			bestepsilon=epsilon
			etp = tp
			efp = fp
			efn = fn
		bar.update(index+1)
		index=index+1
	bar.finish()
	return bestepsilon,bestF1,etp,efp,efn
	


Z ,Zval, yval = load_data()



mu, sigma2 = estimate_gaussian(Z)

p = gaussian_variate(Z,mu,sigma2)

pval = gaussian_variate(Zval,mu,sigma2)

epsilon, F1 ,tp,fp,fn = select_threshold(yval,pval)
print("Best Epsilon value:",epsilon)
print("Best F1 score on Cross Validation data-set:",F1)
print("True positive:",tp,"False postive:",fp,"False Negative:",fn)
print("# Outliers found:",np.sum(p<epsilon))

#subsequence matching
# for index in range(0,len(Zval)):
# 	if(p[index]<epsilon):
# 		print(index)
# 		t = np.linspace(0, 2, 96, endpoint=False)
# 		plt.plot(t, Zval[index], 'g-', linewidth=2, label='filtered data')
# 		plt.xlabel('Time [sec]')
# 		plt.legend()
# 		plt.show()


#sequence matching

Data = np.hsplit(Z,np.array([1]))
X = Data[0]
Data = Data[1]
Data = np.hsplit(Data,np.array([1]))
Y = Data[0]
Z = Data[1] 
index=0
while(index<len(Z)):
	if(pval[index]<epsilon):
		print(index)
		t = np.linspace(0, 2, 64, endpoint=False)
		plt.plot(t, Z[index-32:index+32], 'g-', linewidth=2, label='filtered data')
		plt.xlabel('Time [sec]')
		plt.legend()
		plt.show()
		index=index+32
	index=index+1
