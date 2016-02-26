import numpy as np
from pyDOE import lhs
from kriging import kriging
from true_function import true_function
from ga import ga
from exp_imp import EGO

k = 2
n = 5*2

# sampling plan
X = lhs(k, samples=n)
y = np.zeros((n, 1))

# find true values
for i in range(k):
    y[i] = true_function(X[i], 1)

# create kriging model
kr = kriging(k, X, y)

# train model
kr.train()

# plot prediction
kr.plot_2d()

E = EGO(kr)
MinExpImp = 1e14
infill = 0

while abs(MinExpImp) > 1e-3 and infill < 3*n:
    Xnew, EI = E.next_infill()
    Ynew = true_function(Xnew, 1)
    kr.X = np.vstack((kr.X, Xnew))
    kr.y = np.vstack((kr.y, Ynew))
    infill = infill + 1

    kr.train()
    kr.plot_2d()