import numpy as np
from numpy import array

def mean(x: array) -> float:
    '''
    Calculates the mean value of a sample

        Parameters:
            x (array): sample.

        Returns:
            _x (float): mean value of a sample calculated via the regular estimator.
    '''

    return sum(x)/len(x)

def variance(x: array, unbiased: bool = True) -> float:
    '''
    Calculates the biased and unbiased variance of a sample

        Parameters:
            x (array): sample;
            unbiased (bool): boolean value to select biased or unbiased calculation.

        Returns:
            _V (float): variance of a sample calculated via the regular estimator.
    '''
    _x = mean(x)
    if unbiased == True:
        return sum((x - _x)**2)/(len(x)-1)
    else:
        return sum((x - _x)**2)/len(x)

##CAN NOT MAKE THESE TWO STATISTICS MATCH THE ONE FROM SCIPY!!  (ノಠ益ಠ)ノ彡┻━┻

# def skew(x: array) -> float:
#     return sum((x - mean(x))**3)/(len(x)*(variance(x)**3))

# def kurtosis(x: array) -> float:
#     return sum((x - mean(x))**4)/(len(x)*(variance(x)**4)) - 3

def covariance(x: array, y: array) -> float:
    '''
    Calculates the covariance of two samples

        Parameters:
            x (array): first sample;
            y (array): second sample;

        Returns:
            cov (float): covariance of x and y.
    '''
    temp = array([])
    _x = mean(x)
    _y = mean(y)
    if len(x) == len(y):
        for i in range(len(x)):
            temp = np.append(temp, (x[i]-_x)*(y[i]-_y))
        return sum(temp)/len(x)
    else:
        raise ValueError("Arrays have different dimensions")
    
def correlation(x: array, y: array) -> float:
    '''
    Calculates the correlation of two samples

        Parameters:
            x (array): first sample;
            y (array): second sample;

        Returns:
            cov (float): correlation of x and y.
    '''

    if len(x) == len(y):
        return covariance(x, y)/(variance(x)*variance(y))
    else:
        raise ValueError("Arrays have different dimensions")