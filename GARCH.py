import numpy as np
import scipy

# Used to calculate the volatility of the returns
class garchOneOne(object):

    def __init__ (self, logReturns):
        self.logReturns = logReturns * 100
        self.sigma_2 = self.garch_filter(self.garch_optimizer())
        self.coefficients = self.garch_optimizer()

    def garch_filter(self, parameters):
        " returns the variance expression of a GARCH(1,1) process"

        # slicing parameters list
        omega, alpha, beta = parameters[0], parameters[1], parameters[2]

        # length of logReturns
        n = len(self.logReturns)

        # initializing empty array
        sigma_2 = np.zeros(n)

        # filling the array, if i == 0 then uses the long_term variance
        for i in range(n):
            if i == 0:
                sigma_2[i] = omega / (1 - alpha - beta)
            else:
                sigma_2[i] = omega + alpha * self.logReturns[i-1]**2 + beta * sigma_2[i-1]
        return sigma_2

    def garch_loglikelihood(self, parameters ):
        # defines the log likelihood sum to be optimized given the parameters
        n = len(self.logReturns)
        sigma_2 = self.garch_filter(parameters)
        loglikelihood = -np.sum(-np.log(sigma_2) - self.logReturns**2 / sigma_2)
        return loglikelihood

    def garch_optimizer(self):
        # optimizes the log likelihood function and returns estimated coefficients
        # parameters initialization
        parameters = [0.1, 0.05, 0.92]
        # optimization
        opt = scipy.optimize.minimize(self.garch_loglikelihood, parameters,
                                      bounds=((0.001, 1), (0.001, 1), (0.001, 1)))
        variance = 0.01**2 * opt.x[0] / (1 - opt.x[1] - opt.x[2])
        # 0.01**2 due to the squared returns
        return np.append(opt.x, variance)


