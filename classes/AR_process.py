import numpy as np

class AutoregressiveProcess:
    def __init__(self, coefficients, gamma0):
        self.k = len(coefficients)
        self.gamma0 = gamma0
        self.coefficients = coefficients

    def simulate(self, n, sd=1):
        """Simulate an autoregressive process of length n, with given standard deviation."""
        # Initialize the process with zeros
        x = np.zeros(n)

        # Generate random noise
        noise = sd*np.random.normal(size=n)

        # Simulate the process
        for i in range(self.k, n):
            x[i] = np.dot(x[i-self.k:i], self.coefficients) + noise[i] + self.gamma0

        return x
    
    def mse(self, v0, v):
        """Returns the MSE between the estimated values v0, v and the real coefficients."""
        return np.sum((v-self.coefficients)**2) + (v0-self.gamma0)**2