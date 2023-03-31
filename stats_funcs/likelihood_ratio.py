import pandas as pd
import numpy as np

from scipy.integrate import quad
from scipy.optimize import minimize

from multiprocessing import Pool
from functools import partial
import time

import json
import pickle
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

def orchestrator(params, **kwargs):
    return params, ratio(params, kwargs['df'], kwargs['chi2null'])
    
if __name__ == '__main__':

    log = {}

    sub = {}

    sample = json.load(open('../data/snls_snia.json'))

    df = pd.DataFrame(sample)

    x0 = [0.75, 0.25, 71] # params = [Omega Lambda, Omega Matter, Hubble's Constant, Omega Curvature, Omega Radiation]

    minimum = minimize(chi2, x0 = x0, args = (df), method = 'Nelder-Mead', tol = 1e-6, bounds = ((0,1), (0,1), (0, None)))
    chi2null = chi2(minimum.x, df)

    dict_kwargs = {'df': df, 'chi2null': chi2null}

    sizes = map(int, sys.argv[1].strip('[]').split(','))
    for grid_size in sizes:

        start = time.time()

        if sys.argv[2] == 'single':
            print('Using singleprocessing method')
            results = []
            for params in generate_grid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), [71]):
                results.append(orchestrator(params, **dict_kwargs))
        elif sys.argv[2] == 'multi':
            print('Using multiprocessing method')
            with Pool() as pool:
                results = pool.map(partial(orchestrator, **dict_kwargs), generate_grid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), [71])
                ) 
                pool.close()
                pool.join()

        end = time.time()

        sub['result'] = results
        sub['time'] = end - start
        
        log[grid_size] = sub

    with open(f'../data/{sys.argv[2]}processing_log.pickle', 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    