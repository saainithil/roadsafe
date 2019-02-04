def movingavg(Sample ,window):
	zmoving=[]
	movingvalues=0
	for index in range(0,window):
		movingvalues+=Sample[index]
	zmoving.append(movingvalues/window)
	
	size=len(Sample)-window+1
	
	xaxis=[i for i in range(0,len(Sample))]

	for index in range(1,size):
		movingvalues+=Sample[index+window-1]-Sample[index-1]
		zmoving.append(movingvalues/(window/2 +1))
	size=len(Sample)
	start=size-window+1
	for index in range(start,size):
			zmoving.append(Sample[index])
	plt.plot(xaxis,Sample[0:len(Sample)],'r',zmoving[0:len(Sample)],'b')
	plt.show()
	

plt.figure(1)
plt.plot(xaxis[0:len(Z)],Z[0:len(Z)],'g')
plt.show()
plt.close()


print(len(latitude))
print(len(Z))	


movingavg(Z[0:64],4)
