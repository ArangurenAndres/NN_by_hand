import numpy as np
import math

x = [1,-1]
w_1 = [[1,1,1],[-1,-1,-1]]
b = [0,0,0]

## Compute first linear output
z=[]
for j in range(3):
    z_t=sum(w_1[i][j]*x[i]+b[j] for i in range(2))
    z.append(z_t)
print(f"Z linear output equals: {z}")

## Compute activation of linear output
h = []
for i in range(3):
    h_t = 1/(1+math.exp(-z[j]))
    h.append(h_t)
print(f"H activation function applied to Z equals to: {h}")

## Compute hidden layer linear output
w_2 = [[1,-1,-1],[1,-1,-1]]
b_2 = [0,0]
o = []
for k in range(2):
    o_t = sum(w_2[k][j]*h[j]+b_2[k] for j in range(3))
    o.append(o_t)
print(f"Second linear output O: {o}")

##Compute the softmax
exp_o =  list(map(lambda x: math.exp(x),o))
sum_exp = 0
for exp in exp_o:
    sum_exp+=exp
print(sum_exp)
y = []
for i in range(2):
    y_i = math.exp(o[i])/sum_exp
    y.append(y_i)
print(y)

## Compute the loss 
t_class = 0
loss = []
for i in range(2):
    if i==t_class:
        l = -math.log(y[i])
        loss.append(l)
    else:
        l=0
        loss.append(l)
print(f"Loss equals: {loss}")



 
w_2 = [[1,-1,-1],[1,-1,-1]]
do_dh = [[w_2[k][j] for j in range(3)] for k in range(2)]
print(do_dh)








 