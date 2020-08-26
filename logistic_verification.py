import cv2
import numpy as np
import matplotlib.pyplot as plt
#Loading trained prameters and bias term
f=open('W_params.txt','rb')
W=np.load(f)
f.close()
f=open('b_param.txt','rb')
b=np.load(f)
f.close()
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
while 1:
    image=cv2.VideoCapture(0)
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
        cv2.imwrite('verify.jpg',frame)
        image.release()
        cv2.destroyAllWindows()
        break
    ##print(frame.shape)
    ##print(W.shape)
    X=frame.reshape((1,3500))
    X=X/255
    Z=np.dot(X,W)+b
    A=sigmoid(Z)
    if A>0.5:
        print("Human can be seen in this picture")
    else:
       print("No human")
    frame=cv2.imread('verify.jpg',0)    
    plt.imshow(frame)
    plt.show()


