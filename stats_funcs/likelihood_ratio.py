import pandas as pd
import numpy as np

from scipy.integrate import quad
from scipy.optimize import minimize

from multiprocessing import Pool
from functools import partial
import time
from datetime import datetime

import json
import logging
import sys

"""
params = [Omega Lambda, Omega Matter, Hubble's Constant, Omega Curvature, Omega Radiation, w]
"""

def Dc(z, Omega_l, Omega_m, Omega_k, w):
    def f(z, Omega_l, Omega_m, Omega_k, w):
        return (np.sqrt(Omega_l*(1 + z)**(-3 - 3*w) + Omega_m*(1 + z)**3 + Omega_k*(1 + z)**2 + 9.2e-5*(1 + z)**4))**(-1)
    return quad(f, 0, z, args = (Omega_l, Omega_m, Omega_k, w))[0]

def Dt(z, Omega_l, Omega_m, Omega_k, w):
    if Omega_k > 0:
        return np.sin(np.sqrt(Omega_k)*Dc(z, Omega_l, Omega_m, Omega_k, w))/np.sqrt(Omega_k)
    elif Omega_k == 0:
        return Dc(z, Omega_l, Omega_m, Omega_k, w)
    elif Omega_k:
        return np.sinh(np.sqrt(abs(Omega_k))*Dc(z, Omega_l, Omega_m, Omega_k, w))/np.sqrt(abs(Omega_k))

def Dl(z, Omega_l, Omega_m, Omega_k, w):
    return (1 + z) * Dt(z, Omega_l, Omega_m, Omega_k, w)
               
def mu(z, Omega_l, Omega_m, Omega_k, w, Hubble):
    c = 3e5 #(km/s)
    return 5*np.log10(Dl(z, Omega_l, Omega_m, Omega_k, w)) + 25 + 5*np.log10(c/ Hubble)

def chi2_i(mu_o, sigma_o, z_obs, Omega_l, Omega_m, Omega_k, w, Hubble):
    return ((mu(z_obs, Omega_l, Omega_m, Omega_k, w, Hubble) - mu_o)**2)/sigma_o**2

def chi2(Omega_l, Omega_m, Omega_k, w, Hubble, df): 
    values = []
    for idx in df.index:
        values.append(chi2_i(df.mu[idx], df.sigma[idx], df.z[idx], Omega_l, Omega_m, Omega_k, w, Hubble))
    return sum(values)

def ratio(Omega_l, Omega_m, Omega_k, w, Hubble, df, chi2null):
    return chi2(Omega_l, Omega_m, Omega_k, w, Hubble, df) - chi2null

def _zip_vars(fixed: dict, x) -> dict:
    full_parameters = ['Omega_l', 'Omega_m', 'Omega_k', 'w',  'Hubble']
    variable_parameters = []
    for param in full_parameters:
        if param not in list(fixed.keys()):
            variable_parameters.append(param)
            
    _vars = dict(zip(variable_parameters, x))
    return _vars

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
    fixed = {}
    x0 = config['minimization_seed']
    for i in np.linspace(config['grid'][config['free'][0]][0], config['grid'][config['free'][0]][1], grid_size):
        fixed[config['free'][0]] = i
        for j in np.linspace(config['grid'][config['free'][1]][0], config['grid'][config['free'][1]][1], grid_size):
            fixed[config['free'][1]] = j
            
            minimum = minimize(h, x0 = x0, args = (fixed, config['df']), method = 'Nelder-Mead', tol = 1e-6)
            x0 = minimum.x
            
            _vars = _zip_vars(fixed, minimum.x)
            
            keys = ['Omega_l', 'Omega_m', 'Omega_k', 'w', 'Hubble']
            values = [fixed.get(key,  _vars.get(key)) for key in keys]
            print(f'Set of parameters to calculate ratio: {values}')
            yield tuple(values)

def orchestrator(params, **kwargs):
    return params, ratio(params[0], params[1], params[2], params[3], params[4], kwargs['df'], kwargs['chi2null'])
    
if __name__ == '__main__':

    logging.basicConfig(filename="log/likelihood_ratio.log", level=logging.INFO)
    logging.info(f'Execution {datetime.today()}')

    config = json.load(open(sys.argv[1]))

    logging.info(f'Script running with following configuration: {config}')

    sample = json.load(open(config['sample_path']))

    df = pd.DataFrame(sample)
    config['df'] = df
    fixed = {}

    x0 = [0.76, 0.24, 0.02, -1, 71] #['Omega_l', 'Omega_m', 'Omega_k', 'w',  'Hubble']

    null = minimize(h, x0 = x0, args = (fixed, df), method = 'Nelder-Mead', tol = 1e-6, options = {'maxiter': 10000})
    config['minimum'] = null.x
    logging.info(f'Maximum Likelihood Estimators: {null.x}')


    chi2null = chi2(null.x[0], null.x[1], null.x[2], null.x[3], null.x[4], df)
    dict_kwargs = {'df': df, 'chi2null': chi2null}

    for grid_size in config['grid_sizes']:

        start = time.time()

        if config['mode'] == 'single':
            results = []
            #for params in generate_grid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), [71]):
            for params in generate_grid(config, grid_size):
                results.append(orchestrator(params, **dict_kwargs))
        elif config['mode'] == 'multi':
            with Pool() as pool:
                #results = pool.map(partial(orchestrator, **dict_kwargs), generate_grid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), [71]))
                results = pool.map(partial(orchestrator, **dict_kwargs), generate_grid(config, grid_size))
                pool.close()
                pool.join()

        end = time.time()
        file_name = str(grid_size) + '_' +  config['free'][0] + '_' + config['free'][1] + '_' + 'likelihood_ratio.json'
        logging.info(f'Grid with length {grid_size} calculated in {end-start} seconds')
        logging.info(f'Saving calculation in the file {file_name}')

        # Brace yourself for some terrible coding!!! ლ(ಠ益ಠლ)
        out = {'Omega_l': np.array(list(np.array(results)[:,0]))[:,0].tolist(),
               'Omega_m': np.array(list(np.array(results)[:,0]))[:,1].tolist(),
               'Omega_k': np.array(list(np.array(results)[:,0]))[:,2].tolist(),
               'w': np.array(list(np.array(results)[:,0]))[:,3].tolist(),
               'Hubble': np.array(list(np.array(results)[:,0]))[:,4].tolist(),
               'Ratio': np.array(results)[:,1].tolist()}

        with open('data/' + file_name, 'w') as out_file:
            json.dump(out, out_file)
    
    