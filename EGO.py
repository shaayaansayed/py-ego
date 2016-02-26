import numpy as np
from ga import ga
from scipy.special import erf

class EGO():
    def __init__(self, kriging):
        self.kriging = kriging
        self.nvars = kriging.nvars
        self.y_best = min(kriging.y)

    def next_infill(self):

        print('GA run for optimizing EI...')
        x, EI = ga(self.exp_imp, self.nvars, np.zeros((self.nvars, )), np.ones((self.nvars, )))
        print('GA run complete.')
        return x, EI

    def exp_imp(self, x):
        y_hat, SSqr = self.kriging.predict(x)

        if SSqr == 0:
            EI = 0
        else:
            EI = (self.y_best - y_hat) * (0.5 + 0.5*erf((self.y_best - y_hat)/np.sqrt(2 * SSqr))) + \
                    np.sqrt(0.5*SSqr/np.pi)*np.exp(-0.5*(self.y_best - y_hat)**2/SSqr)
            EI = -EI

        return EI

    
    