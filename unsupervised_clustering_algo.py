import numpy as np
import matplotlib.pyplot as plt
import random
import math

def distance(a,b,c,d):
    return math.hypot((a-c),(b-d))




## sample examples
x1=[1,2,3,4,2,6,3,0,4,7,8,5,9,6,0,3,6,8,10,8,10,12,7,4,10,15,8,7,12,14,13,9]
y1=[1,2,3,2,4,1,2,1,3,4,5,7,6,7,5,9,7,8,12,7,11,14,9,5,3,2,15,12,1,3,4,5]

## initialising centre assignments to zeros
c=np.zeros((len(x1)-1,1))

##initialising number of centroids
n_c=4


##set iterations
iterations=20


##random initialisation of centroids
mu_x=np.random.uniform(low=min(x1), high=max(x1), size=(n_c,1) )
mu_y=np.random.uniform(low=min(y1), high=max(y1), size=(n_c,1) )

## creating list of centraoid points
mu_x_set=[mu_x]
mu_y_set=[mu_y]
##plotting data and initial centroids

plt.scatter(mu_x,mu_y,c='r',marker='x')
##plt.scatter(mu_x,mu_y,c='b',marker='x')
plt.legend(['Initial First centroid', 'Initial Second centroid'], loc='upper right')
plt.scatter(x1,y1,c='g')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.title("Before training")
##plt.scatter(x2,y2,c='g')
plt.show()


<<<<<<< HEAD
for i in range(0,iterations):     ## 3 iterations enough for this to learn
=======
for i in range(0,20):     
>>>>>>> e9bbe6b6b619c6a704506ccb4a8a66c99d42d404

    #Centroid assignment step
    for j in range(0,len(x1)-1):      ##checking distance of x1 y1 datasets
        
        ##a=distance(mu_x,mu_y,x1[j],y1[j])
        a=np.sqrt(np.square((mu_x-x1[j]))+np.square((mu_y-y1[j])))
        #print(a,b)
        mm=np.where(a==min(a))
        c[j,0]=mm[0][0]
        
    ## centroid updation step

    ##for first 18 examples
    k=0    
    while k<n_c:
        a=0
        b=0
        e=0
        for j in range(0,len(x1)-1):
            if c[j,0]==k:     
              a=a+x1[j]
              b=b+y1[j]
              e=e+1
        if e==0:
            e=1
        mu_x[k,0]=a/e
        mu_y[k,0]=b/e
        k=k+1


     ##appending lists
    mu_x_set.append(mu_x)
    mu_y_set.append(mu_y)

k=0
col=['gray','r','c','g','orange']
while k<n_c:
    n_x=[]
    n_y=[]
    for j in range(0,len(x1)-1):
        if c[j,0]==k:
            n_x.append(x1[j])
            n_y.append(y1[j])
    plt.scatter(n_x,n_y,c=col[k%5])
    plt.scatter(mu_x[k,0],mu_y[k,0],c=col[k%5],marker='D')
    
    k=k+1


plt.xlabel('X1')
plt.ylabel('Y1')
plt.title("After training")
plt.show()


##for i in range(0,iterations):
##    plt.plot(mu_x_set[i],mu_y_set[i],'b--')


    
##plotting data, centroids and travel of centroids
##plt.scatter(mu_x,mu_y,c='r',marker='D')
##plt.scatter(mu2_x,mu2_y,c='b',marker='D')
##plt.legend(['Final First centroid', 'Final Second centroid'], loc='upper right')    
##plt.scatter(mu1x[0],mu1y[0],c='r',marker='x')
##plt.scatter(mu2x[0],mu2y[0],c='b',marker='x')
####plt.legend(['Initial First centroid', 'Initaial Second centroid'])
##plt.scatter(n_x1,n_y1,c='r')
##plt.scatter(n_x2,n_y2,c='b')
####plt.scatter(x2,y2,c='g')
##plt.plot(mu1x,mu1y,'g--')
##plt.plot(mu2x,mu2y,'g--')
    

    
        
        
    

