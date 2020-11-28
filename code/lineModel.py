import numpy as np
import pandas as pd

x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]

def getGradient(b, w):
    b_grad = 0.0
    w_grad = 0.0
    for i in range(10):
        b_grad += (-2.0) * (y_data[i] - (w * x_data[i] + b))
        w_grad += (-2.0*x_data[i])*(y_data[i]-(b+w*x_data[i]))
    return (b_grad, w_grad)


def gradient(b, w):
    b_grad = (-2.0) * (y_data[0] - (w * x_data[0] + b))
    w_grad = (-2.0*x_data[0])*(y_data[0]-(b+w*x_data[0]))
    return (b_grad, w_grad)

b = -120 # initial b
w = -4 # initial w
lr = 1e5
b_grad = 0.0
w_grad = 0.0
(b_grad,w_grad) = gradient(b,w)

while(abs(b_grad)>0.00001 or abs(w_grad)>0.00001):
    print("b: "+str(b)+"\t\t\t w: "+str(w)+"\n"+"b_grad: "+str(b_grad)+"\t\t\t w_grad: "+str(w_grad)+"\n")
    b -= lr*b_grad
    w -= lr*w_grad
    (b_grad,w_grad) = getGradient(b,w)

print("the function will be y_data="+str(b)+"+"+str(w)+"*x_data")

error=0.0
for i in range(10):
    error += abs(y_data[i]-(b+w*x_data[i]))
average_error=error/10
print("the average error is "+str(average_error))

