from numpy import *
import scipy as sci
from scipy.linalg import *
from true_function import *

class kriging():

	def __init__(self, X, y):
		self.X = X
		self.y = y

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

X = np.array([[0.4444, 0.6667], 
    [0.2222,    0.8889], 
    [0.8889,    0.7778], 
    [     0,    0.5556], 
    [0.1111,    0.2222], 
    [0.5556,    0.3333], 
    [0.7778,    0.1111], 
    [1.0000,    0.4444], 
    [0.6667,    1.0000], 
    [0.3333,         0]])

y = zeros((X.shape[0], 1))

for i in range(X.shape[0]):
	y[i] = true_function(X[i], 1)

k = kriging(X, y)
theta = 1.0e-03 * np.array([-0.0973, -0.3948])
print(k.likelihood(theta))





