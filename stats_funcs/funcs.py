import numpy as np
from numpy import array
from scipy.special import erf
from scipy.integrate import quad
from math import ceil

def sampler_pdf(x: array, pdf_x: array) -> array:
    '''
    Given a sample {x} and PDF({x}), it creates a distribution fitted to PDF({x}) unsing only
    the probability density function. the first idea that I had.
    This is not the most efficient way to do this...

        Parameters:
            x (array): original sample with N values;
            pdf_x (array): array with calculated values of P({x}).

        Returns:
            temp (array): sample of N values distributed accordingly to P({x}).
    '''
    x = x/max(x)
    temp = array([])
    for i in range(len(pdf_x)):
        temp = np.append(temp, np.random.uniform(x[i] - 0.05, x[i] + 0.05, abs(ceil(100*pdf_x[i]))))
    temp = np.random.choice(temp, len(x), replace=False)

    return temp/max(temp)

def sampler(cdf_x: array, x: array) -> array:
    '''
    Given a sample {x} and CDF({x}), it creates a distribution fitted to PDF({x}) unsing
    inverse transform sampling method.

        Parameters:
            x (array): original sample with N values;
            cdf_x (array): array with calculated values of CDF({x}).

        Returns:
            sample (array): sample of N values distributed accordingly to PDF({x}).
    '''
    unif = np.random.uniform(0, 1, len(cdf_x))

    temp = []
    sample = ([])

    for i in range(len(x)-1):
        a_i = (x[i+1] - x[i])/(cdf_x[i+1] - cdf_x[i])
        b_i = (x[i] - a_i*cdf_x[i])
        temp.append((i, a_i, b_i))

    for i in range(len(temp)-1):
        for j in unif:
            if j>= cdf_x[temp[i][0]] and j<= cdf_x[temp[i+1][0]]:
                # print(j, "is inside the limit [", cdf[temp[i][0]], cdf[temp[i+1][0]], "]")
                # print("Calculating a_i*j + b_i...")
                sample = np.append(sample, temp[i][1]*j + temp[i][2])
                
    return sample

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

def gaussian2d(x: float, mux: float, sigmax: float, y: float, muy: float, sigmay: float, rho: float) -> float:
    '''
    Calculate the value of the gaussian function given its parameters
    
        Parameters: 
            x (float): value of sample;
            mu (float) and sigma (float): constants.
            
        Returns:
            P(x|mu, sigma) (float): Probability value for x, sigma, and mu
                    
    '''
    x_dot = (x - mux)/sigmax
    y_dot = (y - muy)/sigmay
    rho_dot = 1 - (rho**2)
    exp_arg = -(1/(2*rho_dot))*(x_dot**2 + y_dot**2 - 2*rho*x_dot*y_dot)
    denominator = sigmax * sigmay * 2 * np.pi * np.sqrt(rho_dot)
    return np.exp(exp_arg)/denominator

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

def gaussian_distribution(mu: float, sigma: float, size: int, x_min: float = -10, x_max: float = 10) -> array:
    '''
    Creates a gaussian distribution given a sample
    
        Parameters: 
            mu (float) and sigma (float): expected value and variance respectively;
            size (int): dimension of the generated sample;
            x_min (float) and x_max (float): extreme values of the sample.
            
        Returns:
            Distribution (array): Gaussian Distribution.
            
    '''
    x = np.linspace(x_min, x_max, size)
    erf_x = cdf(gaussian, x, (mu, sigma))
    return sampler(erf_x, x)

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

def skewed_distribution(x_0: float, sigma: float, size: int, x_min: float = 0, x_max: float = 10) -> array:
    '''
    Creates a skewed distribution
    
        Parameters: 
            x_0 (float) and sigma (float): expected value and variance respectively;
            size (int): dimension of the generated sample;
            x_min (float) and x_max (float): extreme values of the sample.
            
        Returns:
            Distribution (array): Skewed Distribution.
            
    '''
    x = np.linspace(x_min, x_max, size)
    cdf_x = cdf(some_random_function, x, (x_0, sigma))
    return sampler(cdf_x, x)

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
            temp = np.append(temp, a_i*np.random.uniform(x[i], x[i+1]) + b_i)
    temp = np.append(temp, f[-1])
    
    return temp