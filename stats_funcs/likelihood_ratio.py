import pandas as pd
import numpy as np

from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp2d

from multiprocessing import Pool
from functools import partial
import time
from datetime import datetime

import json
import logging
import sys

"""
params = [Omega Lambda, Omega Matter, Hubble's Constant, w]
"""
#----------------- Mathematical Operations -----------------#
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


#----------------- Cosmological and Statistical Functions -----------------#
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


#----------------- Processing Functions -----------------#
def _zip_vars(fixed: dict, x) -> dict:
    keys = ['Omega_l', 'Omega_m', 'w',  'Hubble']
    variable_parameters = []
    for param in keys:
        if param not in list(fixed.keys()):
            variable_parameters.append(param)
            
    _vars = dict(zip(variable_parameters, x))
    return _vars

def get_values(dct, keys):
    common_keys = set(keys) & set(dct.keys())
    filtered_dict = {k: v for k, v in dct.items() if k in common_keys}
    sorted_values = [filtered_dict[k] for k in keys if k in filtered_dict]
    return tuple(sorted_values)

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
        linspaces = []
        for free in config['free']:
            linspaces.append(np.linspace(config['grid'][free][0], config['grid'][free][1], grid_size))
        cross = cross_product(*linspaces)
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

def orchestrator(params, **kwargs):

    result = []
    
    keys = ['Omega_l', 'Omega_m', 'w', 'Hubble']
    params_dict = dict(zip(keys, params))

    common_keys = set(kwargs['free']) & set(params_dict.keys())
    filtered_dict = {k: v for k, v in params_dict.items() if k in common_keys}
    sorted_values = [filtered_dict[k] for k in kwargs['free'] if k in filtered_dict]

    _ratio = ratio(params[0], params[1], params[2], params[3], kwargs['df'], kwargs['chi2null'])

    result = sorted_values + [_ratio]
    return result
    
if __name__ == '__main__':

    logging.basicConfig(filename="log/likelihood_ratio.log", level=logging.INFO)
    logging.info(f'Execution {datetime.today()}')

    config = json.load(open(sys.argv[1]))

    logging.info(f'Script running with following configuration: {config}')

    sample = json.load(open(config['sample_path']))

    df = pd.DataFrame(sample)
    config['df'] = df

    keys = ['Omega_l', 'Omega_m', 'w', 'Hubble']
    config['free'] = sorted(config['free'], key=lambda k: keys.index(k))

    x0 = [0.76, 0.24, -1, 71] #['Omega_l', 'Omega_m', 'w',  'Hubble']

    null = minimize(h, x0 = x0, args = ({}, df), method = 'Nelder-Mead', tol = 1e-6, bounds = ((0,1), (0,1), (-1.5, 0), (0, None)), options = {'maxiter': 10000})
    config['minimum'] = null.x
    logging.info(f'Maximum Likelihood Estimators: {null.x}')


    chi2null = chi2(null.x[0], null.x[1], null.x[2], null.x[3], df)
    config['chi2null'] = chi2null

    for grid_size in config['grid_sizes']:

        start = time.time()

        if config['mode'] == 'single':
            results = []
            for params in generate_grid(config, grid_size):
                results.append(orchestrator(params, **config))
        elif config['mode'] == 'multi':
            with Pool() as pool:
                results = pool.map(partial(orchestrator, **config), generate_grid(config, grid_size))
                pool.close()
                pool.join()

        end = time.time()

        name = ''
        for free_params in config['free']:
            name += f'_{free_params}'
        file_name = str(grid_size) + name + '_likelihood_ratio.json'

        logging.info(f'Grid with length {grid_size} calculated in {end-start} seconds')
        logging.info(f'Saving calculation in the file {file_name}')

        results_array = np.array(results)
        
        out = {value: column for value, column in zip(config['free'] + ['Ratio'], results_array.T)}
        out = {key: value.tolist() for key, value in out.items()}
        
        # f = interp2d(results_array[:,0], results_array[:,1], results_array[:,2], kind =)

        with open('data/' + file_name, 'w') as out_file:
            json.dump(out, out_file)
    
    