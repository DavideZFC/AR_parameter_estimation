import numpy as np

def fit_AR(time_series, p):
    """
    Fits an AR(p) process using least squares.
    
    Parameters
    ----------
    time_series : numpy array
        The time series to fit.
    p : int
        The order of the AR process to fit.
    
    Returns
    -------
    numpy array
        The coefficients of the AR process.
    """
    
    # Construct the Y matrix and X matrix
    Y = time_series[p:]
    X = np.zeros((len(time_series) - p, p))
    for i in range(p):
        X[:, i] = time_series[p-i-1:-i-1]

    # Add a column of ones to take into account constant term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Use least squares to solve for the AR coefficients
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    beta0 = beta[0]
    beta = beta[1:]
    
    # Return the AR coefficients
    return beta0, beta[::-1]