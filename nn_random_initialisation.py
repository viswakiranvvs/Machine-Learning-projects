import numpy as np
import random
n1=20        #Hidden layer units
n0=3500      #Input layer units
n2=1         #output layer units
W1=(np.random.randn(n1,n0)*00.1)   #random initialisation to eliminate similarity
b1=np.zeros((n1,1))        #zeros for bias terms
##print(W1.shape)
W2=(np.random.randn(n2,n1)*00.1)
b2=0
params={'w1':W1,'b1':b1,'w2':W2,'b2':b2}  ##parameters are stored in a dictonary and saved in a file

f=open('randomini','wb')   ##opening and saving in file randomini
np.save(f,params)
f.close()
