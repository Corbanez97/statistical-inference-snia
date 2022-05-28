import numpy as np
from numpy import array
from scipy.special import erf
from scipy.integrate import quad
from random import uniform
from math import ceil, factorial

def uniform_distribution(n: int, x_min: float, x_max: float) -> array:
    '''
    Generates an array with uniformly distributed numbers, i.e., a uniform distribution

        Parameters:
            n (int): number of values in a sample;
            x_min (float): minimum value of the sample;
            x_max (float): maximum value of the sample.

        Returns:
            temp (array): array with uniform distribution.
    '''
    temp = array([])

    for i in range(n):
        temp = np.append(temp, uniform(x_min, x_max))

    return temp

def sampler(x: array, pdf_x: array) -> array:
    '''
    Given a sample {x} and P({x}), it creates a distribution fitted to P({x})

        Parameters:
            x (array): original sample with N values;
            pdf_x (array): array with calculated values of P({x}).

        Returns:
            temp (array): sample of N values distributed accordingly to P({x}).
    '''
    temp = array([])
    for i in range(len(pdf_x)):
        temp = np.append(temp, uniform_distribution(ceil(len(x)*pdf_x[i]), 0, x[i]))

    if len(temp) < len(x):
        temp = np.append(temp, uniform_distribution(len(x)-len(temp), min(x), max(x)))
    else:
        temp = np.random.choice(temp, len(x), replace=False)
    return temp

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

def binomial_coef(n: int, r: int) -> float:
    return(factorial(n))/(factorial(r))*(factorial(n-r))

def binomial_pdf(r: int, p: float, n: int) -> float:
    '''
    Calculates the value of the probability density function of a binomial distribution
    
        Parameters: 
            r (int): sample;
            p (float): probability of a certain event.
            
        Returns:
            Distribution (array): Binomial Distribution.   
    '''
    if p > 1:
        raise ValueError("p must be a value smaller than 1.")
    else:
        return (p**r)*((1-p)**(n-r))*binomial_coef(n, r)

def binomial_distribution(x: array, p: float) -> array:
    '''
    Creates a binomial distribution
    
        Parameters: 
            x (array): sample;
            p (float): probability of a certain event.
            
        Returns:
            Distribution (array): Binomial Distribution.   
    '''
    n = len(x)
    temp = array([])
    for r in x:
        temp = np.append(temp, binomial_pdf(r, p, n))

    return sampler(x, temp)

def linear_interpolation(f1: float, f2: float, x1: float, x2: float, _x: float) -> float:
    '''
    Calculates the linear_interpolation expression
    
        Parameters:
            f1 and f2 (float): functions indexed values;
            x1 and x2 (float): sample indexed values;
            _x (float): mean of sample values.
            
        Returns:
            \bar{P}_{1,2} (float): linearly interpolated value of f1 and f2.
    '''
    return (f1 + (f2 - f1)*(_x - x1)/(x2 - x1))

def fill_sample(x: array) -> array:
    '''
    Creates an array filled with mean values in between of every two values

        Parametes:
            x (array): sample with N values.

        Returns:
            _x (array): filled sample with, I'm guessing, 2N-1 values.
    '''
    temp = array([])
    for i in range(len(x) - 1):
        temp = np.append(temp, x[i]) 
        temp = np.append(temp, np.mean([x[i],x[i+1]]))
    temp = np.append(temp, x[-1])
    
    return temp

def interpolate(f: array, x: array) -> array:
    '''
    Interpolates a distribution and fills a sample value array

        Parameters:
            f (array): distribution calculated from x sample;
            x (array): sample with N values.

        Returns:
            tup (tuple): tuple with interpolated distribution and filled array with matching dimensions.
    '''
    temp = array([])
    temp_x = fill_sample(x)
    for i in range(0, len(temp_x)-1, 2):
        temp = np.append(temp, f[int(i/2)]) 
        temp = np.append(temp, linear_interpolation(f[int(i/2)], f[int(i/2)+1], temp_x[i], temp_x[i+2], temp_x[i+1]))
    temp = np.append(temp, f[-1])
    
    tup = (temp, temp_x)

    return tup