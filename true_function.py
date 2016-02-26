import numpy as np 

def true_function(X, problem):
    Xmin = np.array([-5, 0])
    Xmax = np.array([10, 15])
    X = Xmin + (Xmax - Xmin) * X

    if problem == 1:
        return ((X[1]-(5.1/4/np.pi**2)*X[0]**2 + 5/np.pi*X[0]-6)**2 + 10*(1-1/8/np.pi) * np.cos(X[0]) + 10);