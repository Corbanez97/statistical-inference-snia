import numpy as np
from numpy import array
from scipy.special import erf
from scipy.integrate import quad
from random import uniform
from math import ceil

def sampler(x: array, pdf_x: array) -> array:
    '''
    Given a sample {x} and P({x}), it creates a distribution fitted to P({x})

        Parameters:
            x (array): original sample with N values;
            pdf_x (array): array with calculated values of P({x}).

        Returns:
            temp (array): sample of N values distributed accordingly to P({x}).
    '''
    x = x/max(x)
    temp = array([])
    for i in range(len(pdf_x)):
        temp = np.append(temp, uniform(ceil(100*pdf_x[i]), x[i] - 0.05, x[i] + 0.05))
    temp = np.random.choice(temp, len(x), replace=False)

    # if len(temp) < len(x):
    #     temp = np.append(temp, uniform(len(x) - len(temp), min(x), max(x)))
    # else:
    #     temp = np.random.choice(temp, len(x), replace=False)

    return temp/max(temp)

def gaussian(x: float, mu: float, sigma: float) -> float:
    '''
    Calculate the value of the gaussian function given its parameters
    
        Parameters: 
            x (float): value of sample;
            mu (float) and sigma (float): constants.
            
        Returns:
            P(x|mu, sigma) (float): Probability value for x, sigma, and mu
                    
    '''
    return (np.exp(((-(x - mu)**2)/ (2 * (sigma**2)))))/ (sigma * (2*np.pi)**(1/2))

def cdf(pdf, x: array, args: tuple) -> array:
    '''
    Generates the cumulative distribution function for a given probability distribution function

        Parameters:
            pdf (function): probability distribution function;
            x (array): domain of calculation;
            args (list): list of parameters for the given distribution.

        Returns:
            cdf (array): cumulative distribution function.
    '''
    temp = ([])
    for i in x:
        temp = np.append(temp, quad(pdf, x[0], i, args)[0])

    return temp/max(temp)

def gaussian_distribution(x: array, mu: float, sigma: float) -> array:
    '''
    Creates a gaussian distribution given a sample
    
        Parameters: 
            x (array): sample;
            mu (float) and sigma (float): expected value and variance respectively.
            
        Returns:
            Distribution (array): Gaussian Distribution.
            
    '''
    temp = array([])
    for i in x:
        temp = np.append(temp, gaussian(i, mu, sigma))

    return sampler(x, temp)

def some_random_function(x: float, x_0: float, sigma: float) -> float:
    '''
    As required on the first assigment, this function calculates a weird value

        Parameters:
            x (float): nth value of a sample;
            x_0 (float): 0th value of a sample;
            sigma (float): variance.

        Returns:
            w_f (float): value of given expression.
    '''
    denominator1 = 2 * (sigma**2)
    denominator2  = (sigma * ((2*np.pi)**(1/2)))

    sec1 = (np.exp((-(x - x_0)**2)/ denominator1)/ denominator2)
    sec2 = (np.exp((-(x + x_0)**2)/ denominator1)/ denominator2)
    sec3 = (erf(x_0/ denominator2))

    return (sec1 - sec2)/(sec3)

def skewed_distribution(x: array, x_0: float, sigma: float) -> array:
    '''
    Creates a skewed distribution
    
        Parameters: 
            x (array): sample;
            mu (float) and sigma (float): expected value and variance respectively.
            
        Returns:
            Distribution (array): Skewed Distribution.
            
    '''
    temp = array([])
    for i in x:
        temp = np.append(temp, some_random_function(i, x_0, sigma))

    return sampler(x, temp)

def interpolate(f: array, x: array) -> array:
    '''
    Linear interpolation of a given function
        
        Parameters:
            f (array): array containing N values f(x);
            x (array): array containing the domain of f with N values.

        Return
            _f (array): array containing 2N - 1 interpolated values of f(x). 
    ''' 
    temp = ([])
    if len(f) != len(x):
        raise ValueError("Dim(f) must be equal to Dim(x).")
    elif len(f) == len(x):
        for i in range(len(x)-1):
            a_i = (f[i+1] - f[i])/(x[i+1] - x[i])
            b_i = f[i] - a_i*x[i]
            temp = np.append(temp, f[i])
            temp = np.append(temp, a_i*uniform(x[i], x[i+1]) + b_i)
    temp = np.append(temp, f[-1])
    
    return temp