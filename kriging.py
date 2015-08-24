import numpy as np

class kriging():

	def __init__(self, X, y):
		self.X = X
		self.y = y

	def likelihood(theta):

		X = self.X
		f = self.y
		Theta = np.full(theta.shape, 10)**theta
		n = X.size
		one = np.ones(n, 1)

		R = np.zeros(n, n)
		for i in xrange(n):
			for k in xrange(i+1, n):
				R(i, j) = np.exp(-np.sum(np.multiply(Theta, np.abs(X(i, :) - X(k, :))**2)))

		R = R + R.T + np.eye(n) + np.eye(n)*np.finfo(float).eps



