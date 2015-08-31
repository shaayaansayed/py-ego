import numpy as np
from numpy import *
import matplotlib
import scipy as sci
from scipy.linalg import *
from true_function import *
from ga import ga
import matplotlib.pyplot as plt

class kriging():

	def __init__(self, nvars, X, y):
		self.nvars = nvars
		self.X = X
		self.y = y

		self.UpperTheta = ones((2, )) * 0.0
		self.LowerTheta = ones((2, )) * -3.0

	def train(self):

		print('GA run for tuning hyperparameters of OBJ. function...')
		self.Theta, self.MinNegLnLik = ga(self.likelihood, self.nvars, self.LowerTheta, self.UpperTheta)
		print('GA run complete.')
		self.U, self.mu = self.likelihood(np.array(self.Theta))[1:]

	def likelihood(self, theta):
		
		X = self.X
		f = self.y
		Theta = 10**theta
		n = size(X,0)
		one = ones((n,1))

		# build correlation matrix
		R = zeros((n, n))
		for i in xrange(n):
			for j in xrange(i+1, n):
				R[i, j] = exp( -sum(Theta * sci.power(abs(X[i, :] - X[j, :]), 2)))

		R = R + R.T + eye(n) + eye(n)*np.finfo(np.float32).eps

		# upper triangular matrix
		U = cholesky(R)

		LnDetPsi = 2 * sum(log(abs(diag(U))));
		mu = (np.dot(one.T, solve(U, solve(U.T, f)))) / (np.dot(one.T, solve(U, solve(U.T, one))))
		SigmaSqr = (np.dot((f - one*mu).T, solve(U, solve(U.T, f-one*mu))))/n
		NegLnLike = -1*(-(n/2)*log(SigmaSqr) - .5*LnDetPsi)

		return NegLnLike, U, mu

	def predict(self, x):

		X = self.X
		f = self.y
		Theta = 10**np.array(self.Theta)
		n = size(X,0)
		one = ones((n,1))
		U = self.U

		r = zeros((n, 1))
		for i in xrange(n):
			r[i] = exp( -sum(Theta * sci.power(abs(X[i, :] - x), 2)))

		y_hat = self.mu + np.dot(r.T, solve(U, solve(U.T, f-one*self.mu)))
		return y_hat

X = np.array([[0.4444, 0.6667], 
    [0.2222,    0.8889], 
    [0.8889,    0.7778], 
    [     0,    0.5556], 
    [0.1111,    0.2222], 
    [0.5556,    0.3333], 
    [0.7778,    0.1111], 
    [1.0000,    0.4444], 
    [0.6667,    1.0000], 
    [0.3333,         0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0]])

y = zeros((X.shape[0], 1))

for i in range(X.shape[0]):
	y[i] = true_function(X[i], 1)

k = kriging(2, X, y)
k.train()

npts = 50
Xplot = np.arange(0, 1, 1.0/npts)
predF = zeros((npts, npts))

for i in range(npts):
	for j in range(npts):
		predF[j, i] = k.predict(np.array([Xplot[i], Xplot[j]]))

plt.figure()
plt.contour(Xplot, Xplot, predF, 50)
plt.show()




