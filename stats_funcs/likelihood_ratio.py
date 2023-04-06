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

def Dc(z, params):
    def f(z, a, b):
        return (np.sqrt(a + b*(1 + z)**3))**(-1)
    return quad(f, 0, z, args = (params[0], params[1]))[0]

def Dt(z, params):
    if len(params) == 3:
        return Dc(z, params)
    else:
        return np.sinh(np.sqrt(params[3])*Dc(z, params))/np.sqrt(params[3])

def Dl(z, params):
    return (1 + z) * Dt(z, params)
               
def mu(z, params):
    c = 3e5 #(km/s)
    return 5*np.log10(Dl(z, params)) + 25 + 5*np.log10(c/ params[2])

def chi2_i(mu_o, sigma_o, z_obs, params):
    return ((mu(z_obs, params) - mu_o)**2)/sigma_o**2

def chi2(params, df): 
    values = []
    for idx in df.index:
        values.append(chi2_i(df.mu[idx], df.sigma[idx], df.z[idx], params))
    return sum(values)

def ratio(params, df, chi2null):
    return chi2(params, df) - chi2null

def generate_grid(x, y, z):
    for i in x:
        for j in y:
            for k in z:
                yield (i, j, k)

def free_params_parser(config, grid_size):
    if "Hubble" not in config['free']:
        _grid = generate_grid(
            np.linspace(config['grid']['Omega_l'][0], config['grid']['Omega_l'][1], grid_size), 
            np.linspace(config['grid']['Omega_m'][0], config['grid']['Omega_m'][1], grid_size), 
            [config['minimum'][2]],
        )
        return _grid
    elif "Omega_l" not in config['free']:
        _grid = generate_grid(
            [config['minimum'][0]], 
            np.linspace(config['grid']['Omega_m'][0], config['grid']['Omega_m'][1], grid_size), 
            np.linspace(config['grid']['Hubble'][0], config['grid']['Hubble'][1], grid_size),
        )
        return _grid
    elif "Omega_m" not in config['free']:
        _grid = generate_grid(
            np.linspace(config['grid']['Omega_l'][0], config['grid']['Omega_l'][1], grid_size),
            [config['minimum'][1]],
            np.linspace(config['grid']['Hubble'][0], config['grid']['Hubble'][1], grid_size),
        )
        return _grid

def orchestrator(params, **kwargs):
    return params, ratio(params, kwargs['df'], kwargs['chi2null'])
    
if __name__ == '__main__':

    logging.basicConfig(filename="log/likelihood_ratio.log", level=logging.INFO)
    logging.info(f'Execution {datetime.today()}')

    config = json.load(open(sys.argv[1]))

    logging.info(f'Script running with following configuration: {config}')

    sample = json.load(open(config['sample_path']))

    df = pd.DataFrame(sample)

    x0 = [0.75, 0.25, 71] # params = [Omega Lambda, Omega Matter, Hubble's Constant, Omega Curvature, Omega Radiation]

    minimum = minimize(chi2, x0 = x0, args = (df), method = 'Nelder-Mead', tol = 1e-6, bounds = ((0,1), (0,1), (0, None)))
    config['minimum'] = minimum.x
    logging.info(f'Maximum Likelihood Estimators: {minimum.x}')


    chi2null = chi2(minimum.x, df)
    dict_kwargs = {'df': df, 'chi2null': chi2null}

    for grid_size in config['grid_sizes']:

        start = time.time()

        if config['mode'] == 'single':
            results = []
            #for params in generate_grid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), [71]):
            for params in free_params_parser(config, grid_size):
                results.append(orchestrator(params, **dict_kwargs))
        elif config['mode'] == 'multi':
            with Pool() as pool:
                #results = pool.map(partial(orchestrator, **dict_kwargs), generate_grid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), [71]))
                results = pool.map(partial(orchestrator, **dict_kwargs), free_params_parser(config, grid_size))
                pool.close()
                pool.join()

        end = time.time()
        file_name = str(grid_size) + '_' +  config['free'][0] + '_' + config['free'][1] + '_' + 'likelihood_ratio.json'
        logging.info(f'Grid with length {grid_size} calculated in {end-start} seconds')
        logging.info(f'Saving calculation in the file {file_name}')

        out = {'Omega_l': np.array(list(np.array(results)[:,0]))[:,0].tolist(),
               'Omega_m': np.array(list(np.array(results)[:,0]))[:,0].tolist(),
               'Hubble': np.array(list(np.array(results)[:,0]))[:,2].tolist(),
               'Ratio': np.array(results)[:,1].tolist()}

        with open('data/' + file_name, 'w') as out_file:
            json.dump(out, out_file)
    
    