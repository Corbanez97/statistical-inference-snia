import pandas as pd
import numpy as np

from scipy.integrate import quad
from scipy.optimize import minimize


from multiprocessing import Pool
from functools import partial
import time
from datetime import datetime

from typing import Dict

import json
import logging
import sys

# Take Omega_k from args and calculate it after

def Dc(z, Omega_l, Omega_m, w):
    Omega_k = 1 - (Omega_l + Omega_m + 9.2e-5)
    def f(z, Omega_l, Omega_m, w):
        return (np.sqrt(Omega_l*(1 + z)**(-3 - 3*w) + Omega_m*(1 + z)**3 + Omega_k*(1 + z)**2 + 9.2e-5*(1 + z)**4))**(-1)
    return quad(f, 0, z, args = (Omega_l, Omega_m, w))[0]

def Dt(z, Omega_l, Omega_m, w):
    Omega_k = 1 - (Omega_l + Omega_m + 9.2e-5)
    if Omega_k > 0:
        return np.sin(np.sqrt(Omega_k)*Dc(z, Omega_l, Omega_m, w))/np.sqrt(Omega_k)
    elif Omega_k == 0:
        return Dc(z, Omega_l, Omega_m, w)
    elif Omega_k:
        return np.sinh(np.sqrt(abs(Omega_k))*Dc(z, Omega_l, Omega_m, w))/np.sqrt(abs(Omega_k))

def Dl(z, Omega_l, Omega_m, w):
    return (1 + z) * Dt(z, Omega_l, Omega_m, w)
               
def mu(z, Omega_l, Omega_m, w, Hubble):
    c = 3e5 #(km/s)
    return 5*np.log10(Dl(z, Omega_l, Omega_m, w)) + 25 + 5*np.log10(c/ Hubble)

def chi2_i(mu_o, sigma_o, z_obs, Omega_l, Omega_m, w, Hubble):
    return ((mu(z_obs, Omega_l, Omega_m, w, Hubble) - mu_o)**2)/sigma_o**2

def chi2(Omega_l, Omega_m, w, Hubble, df): 
    values = []
    for idx in df.index:
        values.append(chi2_i(df.mu[idx], df.sigma[idx], df.z[idx], Omega_l, Omega_m, w, Hubble))
    return sum(values)

def ratio(Omega_l, Omega_m, w, Hubble, df, chi2null):
    return chi2(Omega_l, Omega_m, w, Hubble, df) - chi2null


def cross_product(*linspaces):
    # Create a list to hold the linspaces and their lengths
    linspaces_list = []
    lengths = []

    # Loop through each linspace and get its values and length
    for linspace in linspaces:
        linspaces_list.append(linspace)
        lengths.append(len(linspace))

    # Create a 2D array to hold the cross product
    cross_product_array = np.zeros((np.prod(lengths), len(linspaces)))

    # Loop through each value in the linspaces and add it to the cross product array
    for i in range(len(linspaces)):
        repeat = np.prod(lengths[i+1:])
        cross_product_array[:, i] = np.tile(np.repeat(linspaces[i], repeat), np.prod(lengths[:i]))

    return cross_product_array

def _zip_vars(fixed: dict, x) -> dict:
    full_parameters = ['Omega_l', 'Omega_m', 'w',  'Hubble']
    variable_parameters = []
    for param in full_parameters:
        if param not in list(fixed.keys()):
            variable_parameters.append(param)
            
    _vars = dict(zip(variable_parameters, x))
    return _vars

def _minimize(args):

    dict_kwargs = {
        'fun': h,
        'x0': args[2],
        'args': (args[0], args[1]),
        'method': 'Nelder-Mead',
        'jac': None,
        'hessp': None,
        'hess': None,
        'constraints': (),
        'tol': 1e-3,
        'callback': None,
        'options': None,
    }
    _min = minimize(**dict_kwargs)

    _vars = _zip_vars(args[0], _min.x)

    keys = ['Omega_l', 'Omega_m', 'w', 'Hubble']
    values = [args[0].get(key,  _vars.get(key)) for key in keys]

    return values

def h(x, fixed, df):
    """
    This is a wrapper for the chi2 function. 
    A workaround to the fact that scipy.optimize.minimize uses positional arguments
    
        Given a dict of fixed variables, this fuction enables minimize to pass an array of arguments
    """
    
     # Creates dictionary of variables to be optimized
    _vars = _zip_vars(fixed, x)

    # fixed['df'] = df
    g = partial(chi2, df = df, **fixed) # Creates partial function fixing variables set by `fixed`
        
    # Unzips `_vars` as named args to `g`
    return g(**_vars) 

def generate_grid(config, grid_size):
    
    def generate_free_params(config, grid_size):
        standard_x0 = {'Omega_l':0.74, 'Omega_m':0.26, 'w':-1,  'Hubble':70}
        _linspaces = []
        for free in config['free']:
            _linspaces.append(np.linspace(config['grid'][free][0], config['grid'][free][1], grid_size))
        cross = cross_product(*_linspaces)
        for i in cross:
            fixed = {config['free'][j]: i[j] for j in range(len(config['free']))}
            dict1 = {key: value for key, value in standard_x0.items() if key not in fixed}
            
            standard_x0_values = [value for value in dict1.values()]
            yield fixed, config['df'], standard_x0_values
    
    
    with Pool() as pool:
        results = pool.map(_minimize, generate_free_params(config, grid_size))
        pool.close()
        pool.join()

    for result in results:
        yield  result

if __name__ == '__main__':
    sample = json.load(open('data/snls_snia.json'))
    config = json.load(open('config/likelihood_ratio_config.json'))

    df = pd.DataFrame(sample)
    config['df'] = df

    grid_size = config['grid_sizes'][0]

    start = time.time()

    g = generate_grid(config, grid_size)

    end = time.time()
    
    print(f'Grid with length {grid_size} generated in {end-start} seconds.')
    for i in g:
        print(i)