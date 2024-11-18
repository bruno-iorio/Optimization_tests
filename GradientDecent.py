import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

Q = np.matrix([[2,0,0],[0,1,0],[0,0,1]])
#this will be an example given for dimention 3;
def f(x):
    return (1/2) * x.T@(Q@x);
def grad_f(x):
    return Q @ x
def hessian_f(x):
    return Q

def GradientDescentNewton(x,f,grad_f,hessian_f,eps=0.001,step = 0.01,max=10000):
    it = 0
    x_k = x
    next = x
    while it < max or (it > 0 and abs(f(next) - f(x_k)) > eps):
        this = x_k
        this_f= f(this) 
        this_grad = grad_f(this) 
        this_hessian = hessian_f(this)
        if (not np.any(this_grad)):
            print("Found Global minimizer")
            return this
        try: 
            next = this - step*np.linalg.inv(this_hessian)@ this_grad
            x_k = next
            it += 1
        except: 
            print("Found global minimizer")
            return this
    print("Approximation of minimizer Found")
    return next

print(GradientDescentNewton(np.matrix([[0,1,0]]).T,f,grad_f,hessian_f))









