import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

##defining activation functions

def sigmoid(Z):
    a=1/(1+np.exp(-Z))
    return a

##regularisation parameter, iterations and training examples
lambd=0.09
itr=850
m=80

features=3500      ##(70*50) pixel photos

##Loading data from saved files
f=open('x_train.txt','rb')
X_train=np.load(f).T
X_train=X_train/255    ## feature scaling
#print(X_train.shape)
f.close()
f=open('y_train.txt','rb')
Y_train=np.load(f)
f.close()
#print(Y_train.shape)

##learning rate
alpha=0.5

##Initialising weights and bias to zeros
W=(np.zeros((features,1)))
b=0

for i in range(0,itr):
    ##Forward prop
    Z=np.dot(W.T,X_train)+b
    A=sigmoid(Z)

    ##computing cost
    cost=-(np.dot(Y_train,(np.log(A)).T)+np.dot(1-Y_train,(np.log(1-A)).T)+lambd*(np.sum(np.square(W),axis=0,keepdims=True)))/m ##using regularisation

    #printing cost after every time
    print(cost)

    ##Back prop
    dZ=A-Y_train
    dW=np.dot(X_train,dZ.T)/m
    db=np.sum(dZ,keepdims=True,axis=1)/m
    W=W-(lambd*alpha*dW)
    b=b-(lambd*alpha*db)

##retrieving test set values    
A=A>0.5
print(sum(sum(A==Y_train)))
f=open('x_test.txt','rb')    
X_test=np.load(f).T
f.close()
f=open('Y_test.txt','rb')
Y_test=np.load(f)
f.close()

##Forward prop on test set
Z_test=np.dot(W.T,X_test)+b
A_test=(sigmoid(Z_test)>0.5)
print(sum(sum(A_test==Y_test)))
