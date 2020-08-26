##importing required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


## Defining required functions
def sigmoid(Z):       #sigmoid activation function
    a=1/(1+np.exp(-Z))
    return a
def sigderivative(Z):
    a=sigmoid(Z)
    return a*(1-a)
def leakyRelu(Z):    #leaky ReLu activation function
    a=np.maximum(0.01*Z,Z)    
    return a
def derivative_leakyReLu(Z):
    x=Z
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            if Z[i,j]>=0:
                Z[i,j]=1
            else:
                Z[i,j]=0.01
    return Z            



#lambd=5 ##regularisation parameter

## fixing iterations, training examples, learning rate, test exampls
itr=550
m=80
alpha=0.05
m_test=20


## Getting training set and test set from save files by x_initialisation
f=open('x_train.txt','rb')
X_train=np.load(f).T    # of shape (features, training exampls)

## Dividing by maximum value
X_train=X_train/255

f.close()

## Getting training set and test set from save files by y_initialisation
f=open('y_train.txt','rb')
Y_train=np.load(f) # of shape(1,m_train)
f.close()
#print(Y_train.shape)

n1=20     #Hidden layer units
n0=3500   #Input layer units
n2=1    #output layer units

print(str(alpha)+' '+str(itr)+' '+str(n1))
## These are initiased already and stored in files due to differences in every time

##W1=(np.random.randn(n1,n0)*0.1)
####print(W)
##b1=0
##print(W1.shape)
##W2=(np.random.randn(n2,n1)*0.1)
##b2=0

## Retrieving values from randomini file which is saved by random initialisation.py file
f=open('randomini','rb')
params=np.load(f,allow_pickle=True).item()
W1=params['w1']
b1=params['b1']
W2=params['w2']
b2=params['b2']
f.close()

## Training the set
for i in range(0,itr):
    ##Forward Prop
    
    Z1=np.dot(W1,X_train)+b1     ##Z1
    A1=leakyRelu(Z1)             ##A1
    Z2=np.dot(W2,A1)+b2         ##Z2
    A2=sigmoid(Z2)              ##A2
    cost=-(np.dot(Y_train,(np.log(A2)).T)+np.dot(1-Y_train,(np.log(1-A2)).T))/m    ##calculating cost every time
    if i%500==0:    ##printing cost after every 500 iterations
        print(cost)
        #print(cost>0)
        
    ##BAck Prop    
    dZ2=A2-Y_train         
    dW2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,keepdims=True,axis=1)/m
    dZ1=np.dot(W2.T,dZ2)*derivative_leakyReLu(Z1)   ##since we used leakyrelu for activation so, derivative in back prop     
    dW1=np.dot(dZ1,X_train.T)/m
    db1=np.sum(dZ1,keepdims=True,axis=1)/m
    
    W1=W1-(alpha*dW1)   
    b1=b1-(alpha*db1)
    W2=W2-(alpha*dW2)
    b2=b2-(alpha*db2)

## Checking weights on training set    
A2=A2>0.5
#printing cost of training set
print('train cost= '+str(cost))
#printing correctly classified examples
print(sum(sum(A2==Y_train)))

## checking weights on test set

#Retrieving values from test set
f=open('x_test.txt','rb')    
X_test=np.load(f).T
X_test=X_test/255
f.close()
f=open('Y_test.txt','rb')
Y_test=np.load(f)
f.close()


##Forward prop for test set with trained weights

Z_test=np.dot(W1,X_test)+b1
A_test=(sigmoid(Z_test))
Z_t2=np.dot(W2,A_test)+b2
A_t2=sigmoid(Z_t2)

##calculating and printing test set cost
cost=-(np.dot(Y_test,(np.log(A_t2)).T)+np.dot(1-Y_test,(np.log(1-A_t2)).T))/m_test
print('test cost= '+str(cost))
A_t2=A_t2>0.5
print(sum(sum(A_t2==Y_test)))



## saving trained weights in file
params_after_training={'w1':W1,'b1':b1,'w2':W2,'b2':b2}
f=open('nn_params_after_good_training.txt','wb')
np.save(f,params)
f.close()



## checking on real time examples
e=int(input('enter for checkng on real time examples:'))

## verifying with real time examples
while(e):
    image=cv2.VideoCapture(0)    #capturing video
    if image.isOpened():
       ret,frame=image.read()
    else:
       ret=False
    while ret:
      ret,frame=image.read()
      frame=cv2.resize(frame,(50,70),0)
      frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      cv2.imshow("ima",frame)
      k=cv2.waitKey(1)
      if k==13:
        cv2.imwrite('verify.jpg',frame)      #when enter is pressed, photo will be saved and trained
        image.release()
        cv2.destroyAllWindows()
        break
    ##print(frame.shape)
    ##print(W.shape)

    
    #Forward prop for real time photos
    X=frame.reshape((1,3500))
    X=X.T/255
    Z_test=np.dot(W1,X)+b1
    A_test=(sigmoid(Z_test))
    Z_t2=np.dot(W2,A_test)+b2
    A_t2=sigmoid(Z_t2)
    A_t2=A_t2>0.5
    if A_t2: 
        print('human')           #prints the result
    else:
        print('Not human')
    e=int(input('enter:'))    
