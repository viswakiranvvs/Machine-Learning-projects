import numpy as np
import matplotlib.pyplot as plt
import random
import math

def distance(a,b,c,d):
    return math.hypot((a-c),(b-d))

## initialising centre assignments to zeros
c=np.zeros((18,1))

## sample examples
x1=[1,2,3,4,2,6,3,0,4]
y1=[2,3,1,2,4,1,2,1,3]
x2=[7,8,5,9,6,0,3,6,8]
y2=[4,5,7,6,7,8,9,7,8]

##random initialisation of centroids
mu1_x=random.randrange(min(x1),max(x1))
mu1_y=random.randrange(min(y1),max(y1))
mu2_x=random.randrange(min(x2),max(x2))
mu2_y=random.randrange(min(y2),max(y2))

## creating list of centraoid points
mu1x=[mu1_x]
mu1y=[mu1_y]
mu2x=[mu2_x]
mu2y=[mu2_y]

##plotting data and initial centroids
plt.scatter(x1,y1,c='r')
plt.scatter(mu1_x,mu1_y,c='r',marker='x')
plt.scatter(x2,y2,c='b')
plt.scatter(mu2_x,mu2_y,c='b',marker='x')
plt.show()


for i in range(0,3):     ## 3 iterations enough for this to learn

    #Centroid assignment step
    for j in range(0,9):      ##checking distance of x1 y1 datasets
        a=distance(mu1_x,mu1_y,x1[j],y1[j])
        b=distance(mu2_x,mu2_y,x1[j],y1[j])
        #print(a,b)
        if a>=b:
            c[j,0]=2
        else:
            c[j,0]=1
        
    for j in range(0,9):       ##checking distance of x2 y2 datasets
        a=distance(mu1_x,mu1_y,x2[j],y2[j])
        b=distance(mu2_x,mu2_y,x2[j],y2[j])
        #print(a,b)
        if a>b:
            c[j+9,0]=2
        else:
            c[j+9,0]=1
           
    ## centroid updation step
    a=0
    b=0
    n=0
    d=0
    e=0
    f=0

    ##for first 9 examples
    for j in range(0,9):
        if c[j,0]==1:     
            a=a+x1[j]
            b=b+y1[j]
            e=e+1
        else:             
            n=n+x2[j]
            d=d+y2[j]
            f=f+1

    ## for next 9 examples        
    for j in range(9,18):
        if c[j,0]==1:      
            a=a+x1[j%9]
            b=b+y1[j%9]
            e=e+1
        else:
            n=n+x2[j%9]
            d=d+y2[j%9]
            f=f+1


    ##updating centroid        
    mu1_x=a/e
    mu1_y=b/e
    mu2_x=n/f
    mu2_y=d/f

    ##appending lists
    mu1x.append(mu1_x)
    mu1y.append(mu1_y)
    mu2x.append(mu2_x)
    mu2y.append(mu2_y)


##plotting data, centroids and travel of centroids    
plt.scatter(x1,y1,c='r')
plt.scatter(mu1_x,mu1_y,c='r',marker='x')
plt.scatter(x2,y2,c='b')
plt.scatter(mu2_x,mu2_y,c='b',marker='x')
plt.plot(mu1x,mu1y,'g--')
plt.scatter(mu1x[0],mu1y[0],c='r',marker='x')
plt.scatter(mu2x[0],mu2y[0],c='b',marker='x')
plt.plot(mu2x,mu2y,'g--')
plt.show()    

    
        
        
    
