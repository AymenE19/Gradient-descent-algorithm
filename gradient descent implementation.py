import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as lp


def gradient(x):
    ''' gradient function takes a variable x (mainly a vector) as input and calculates its gradient at that given point.
        the derivation is calculated via the Forward difference method
    '''
    grad = np.zeros(len(x))    
    dx = x.copy()
    h = 10**-6
    for j in range(len(x)):
        dx[j] = x[j] + h
        grad[j] = (func(dx) - func(x))/h
        dx = x.copy()
    return grad
 
#Rosenbrock Function
def func(x):                     
    return 50*(x[1] - x[0]**2)**2 + (1 - x[0])**2


def step_size(x):      
    ''' for a given iteration k, the goal of a linear search is to calculate the step size of alpha, 
        that leads to decrease the objective function in the direction -∇f(w_k)
        exact linear search finds the optimal step size that maximizes the decreasing of the objective function,
        which can be resource intensive to evaluate.
        another approach consists of finding alpha that is close enough to the optimal one, using backtracking.
    '''           
    alpha = 1                    
    beta = 0.9
    while func(x - alpha*gradient(x)) > (func(x) - 0.5*alpha*lp.norm(gradient(x))**2):
        alpha *= beta
    return alpha
    
x = [0, 0] #initialization for the iterations
x1 = np.array([])
x2 = np.array([])
tol = 10**-7 #Tolerence Value to stop the iteration loop 
pre = -50
k = 0        
print("Initial Guess: ", x)

#Gradient-Descent Algorithm
while abs(func(x) - pre) > tol:
    pre = func(x)
    x1 = np.append(x1, x[0])
    x2 = np.append(x2, x[1])
    alpha = step_size(x)
    x -= alpha*gradient(x)  #Gradient-Descent Algorithm
    k += 1
    
print("Function Value: ", func(x), "at x = ", x, "in", k, "iterations")
print("For Tolerence Value: ", tol)

#Contour Plots
x = np.arange(-0.5, 1.2, 0.02)
y = np.arange(-0.5, 1.2, 0.02)
X, Y = np.meshgrid(x, y)
Z = 50*(Y - X**2)**2 + (1 - X)**2
plt.contour(X, Y, Z, 100)
plt.plot(x1, x2, 'o-')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("$f\,(x_1, x_2) \,= 50(x_2 - x_1^2)^2 + (1 - x_1)^2 $")
plt.grid(True)
plt.show()
