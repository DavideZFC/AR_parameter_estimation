import numpy as np
from functions.misc.confidence_bounds import bootstrap_ci
from functions.misc.plot_from_dataset import plot_data

class Experiment:
    
    def __init__(self, AR_process, estimator):
        self.ar = AR_process
        self.est = estimator
        self.k = self.ar.k

    def make_experiment(self, seeds, n, sd=1):
        """Simulates the estimation of the parameters of a process of length n, once for every seed."""

        # generate matrix to contain dat
        self.data_matrix = np.zeros((seeds, n))

        # when to start estimating parametes
        first = 2*self.k + 2

        for s in range(seeds):

            # generate process to estimate
            X = self.ar.simulate(n+first, sd=sd)

            for i in range(n):
                # estimate the coeffcients
                v0, v = self.est(X[:(i+first)], self.k)

                # fill data matrix with errors
                self.data_matrix[s,i] = self.ar.mse(v0, v)

        return self.data_matrix
    
    def plot_confidence_region(self, ax, col, label):
        """Plots confidence curves for how much the estimated parametrs can approximate the real ones"""
        low, high = bootstrap_ci(self.data_matrix)
        x = np.linspace(0, len(low), len(low))
        plot_data(x, low, high, ax, col, label)
        return low, high


