import numpy as np
import scipy as sci
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt
from ga import ga

class kriging():

    def __init__(self, nvars, X, y):
        self.nvars = nvars
        self.X = X
        self.y = y

        self.UpperTheta = np.ones((2, )) * 0.0
        self.LowerTheta = np.ones((2, )) * -3.0

    def train(self):

        print('GA run for tuning hyperparameters of OBJ. function...')
        self.Theta, self.MinNegLnLik = ga(self.likelihood, self.nvars, self.LowerTheta, self.UpperTheta)
        print('GA run complete.')
        self.U, self.mu, self.SigmaSqr = self.likelihood(self.Theta)[1:]

    def likelihood(self, theta):
        
        X = self.X
        f = self.y
        Theta = 10**theta
        n = np.size(X,0)
        one = np.ones((n,1))

        # build correlation matrix
        R = np.zeros((n, n))
        for i in xrange(n):
            for j in xrange(i+1, n):
                R[i, j] = np.exp( -sum(Theta * sci.power(abs(X[i, :] - X[j, :]), 2)))

        R = R + R.T + np.eye(n) + np.eye(n)*np.finfo(np.float32).eps

        # upper triangular matrix
        U = cholesky(R)

        LnDetPsi = 2 * sum(np.log(abs(np.diag(U))));
        mu = (np.dot(one.T, solve(U, solve(U.T, f)))) / (np.dot(one.T, solve(U, solve(U.T, one))))
        SigmaSqr = (np.dot((f - one*mu).T, solve(U, solve(U.T, f-one*mu))))/n
        NegLnLike = -1*(-(n/2)*np.log(SigmaSqr) - .5*LnDetPsi)

        return NegLnLike, U, mu, SigmaSqr

    def predict(self, x):

        X = self.X
        f = self.y
        Theta = 10**self.Theta
        n = np.size(X,0)
        one = np.ones((n,1))
        U = self.U

        r = np.zeros((n, 1))
        for i in xrange(n):
            r[i] = np.exp( -sum(Theta * sci.power(abs(X[i, :] - x), 2)))

        y_hat = self.mu + np.dot(r.T, solve(U, solve(U.T, f-one*self.mu)))
        SSqr = abs(self.SigmaSqr * (1 - np.dot(r.T, solve(U, solve(U.T, r)))))
        return y_hat, SSqr

    def plot_2d(self):
        npts = 50
        Xplot = np.arange(0, 1, 1.0/npts)
        predF = np.zeros((npts, npts))

        for i in range(npts):
            for j in range(npts):
                predF[j, i] = self.predict(np.array([Xplot[i], Xplot[j]]))[0]

        plt.figure()
        plt.contour(Xplot, Xplot, predF, 50)
        plt.plot(self.X[:, 0], self.X[:, 1], 'ro')
        plt.show(block=False)