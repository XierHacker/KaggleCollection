from __future__ import  print_function,division
import numpy as np
import matplotlib.pyplot as plt

def gen(x):
    y=np.sin(2*np.pi*x)+np.random.normal(loc=0,scale=0.01,size=100)
    print (y.shape)
    return y

x=np.linspace(start=-1,stop=1,num=100)
y=gen(x)

#print (x)
#print (x.shape)

def getgrad(w,b,x,y):
    #g store the value
    g=np.zeros(shape=(3,))
    B=0
    loss=0
    for i in range(100):
        X=np.power([x[i],x[i],x[i]],[1,2,3])
        loss+=(np.dot(w,X)+b-y[i])*(np.dot(w,X)+b-y[i])
      #  print ("X:",X)
        g[0]+=2*(np.dot(w,X)+b-y[i])*x[i]
        g[1] += 2 * (np.dot(w, X) + b - y[i]) * x[i]*x[i]
        g[2] += 2 * (np.dot(w, X) + b - y[i]) * x[i]*x[i]*x[i]
       # g[3] += 2 * (np.dot(w, X) + b - y[i]) * x[i] * x[i] * x[i]*x[i]
        #g[4] += 2 * (np.dot(w, X) + b - y[i]) * x[i] * x[i] * x[i] * x[i]*x[i]
        #g[5] += 2 * (np.dot(w, X) + b - y[i]) * x[i] * x[i] * x[i] * x[i]*x[i]*x[i]
        #g[6] += 2 * (np.dot(w, X) + b - y[i]) * x[i] * x[i] * x[i] * x[i]*x[i]*x[i]*x[i]
       # print ("g:",g)
        B+=2 * (np.dot(w, X) + b - y[i])
       # print ("b:",B)
    print ("loss:",loss/100)
    return g/100,B/100

def gradDescent(w,b,x,y,epoch=10000, rate=0.1):
    w_temp=w
    b_temp=b
    iter = 1
    while (iter <= epoch):
        print("epoch:", iter)
        temp_grad, temp_b = getgrad(w_temp, b_temp, x, y)
        w_temp = w_temp - rate * temp_grad
        print("w:", w_temp)
        b_temp = b_temp - rate * temp_b
        print("b:", b_temp)
        iter += 1
    return w_temp,b_temp


w_init=np.array([0,0,0])
b_init=0
new_w,new_b=gradDescent(w_init,b_init,x,y)


def getf(w,x,b):
    f = np.zeros(shape=(100))
    for i in range(100):
        X = np.power([x[i], x[i], x[i]], [1, 2, 3])
        f[i] =np.dot(w, X) + b
    return f

f=getf(new_w,x,new_b)
print (f)

plt.plot(x,y,"+",x,f,"*")
plt.show()








