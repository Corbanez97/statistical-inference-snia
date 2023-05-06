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
def cartesian_product(*linspaces):
    """
    Compute the cartesian product of a set of one-dimensional arrays.

    Given n one-dimensional arrays, this function returns an array of shape (N, n), where
    N is the product of the lengths of the input arrays. The resulting array is obtained 
    by taking the Cartesian product of the input arrays, with the first array varying 
    fastest and the last array varying slowest.

    Args:
        *linspaces: One or more one-dimensional arrays to be used in the cartesian product.

    Returns:
        A 2D numpy array of shape (N, n), where N is the product of the lengths of the 
        input arrays, and n is the number of input arrays. The i-th row of the array 
        contains the i-th combination of values from the input arrays.
    """
    
    # Create a list to hold the linspaces and their lengths
    linspaces_list = []
    lengths = []

    # Loop through each linspace and get its values and length
    for linspace in linspaces:
        linspaces_list.append(linspace)
        lengths.append(len(linspace))

    # Create a 2D array to hold the cartesian product
    cartesian_product_array = np.zeros((np.prod(lengths), len(linspaces)))

    # Loop through each value in the linspaces and add it to the cartesian product array
    for i in range(len(linspaces)):
        repeat = np.prod(lengths[i+1:])
        cartesian_product_array[:, i] = np.tile(np.repeat(linspaces[i], repeat), np.prod(lengths[:i]))

    return cartesian_product_array


#----------------- Cosmological and Statistical Functions -----------------#
def Dc(z: float, Omega_l: float, Omega_m: float, w: float) -> float:
    """
    Calculates the adimensional comoving distance using the input parameters.

    Parameters:
        z (float): The redshift value.
        Omega_l (float): The value of the cosmological constant.
        Omega_m (float): The value of the matter density parameter.
        w (float): The value of the dark energy equation of state.

    Returns:
        The adimensional comoving distance.
        
    Notes:
        The adimensional comoving distance is the distance that would be traveled 
        by a photon emitted at redshift z, if the universe were not expanding, 
        divided by the Hubble distance. This function calculates the comoving 
        distance by performing a numerical integration of the function f(z, Omega_l, 
        Omega_m, w) over the redshift range [0, z], where f(z, Omega_l, Omega_m, w) 
        is defined as:

        f(z, Omega_l, Omega_m, w) = 1 / sqrt(Omega_l*(1 + z)**(3 + 3*w) + 
                                              Omega_m*(1 + z)**3 + 
                                              Omega_k*(1 + z)**2 + 
                                              9.2e-5*(1 + z)**4)

        where Omega_k = 1 - (Omega_l + Omega_m + 9.2e-5) represents the curvature of 
        the universe. The result is returned as an adimensional quantity, which 
        can be converted to physical units of length by multiplying by the Hubble 
        distance.
    """
    
    Omega_k = 1 - (Omega_l + Omega_m + 9.2e-5)

    def f(z, Omega_l, Omega_m, w):
        return 1/(np.sqrt(Omega_l*(1 + z)**(3 + 3*w) + Omega_m*(1 + z)**3 + Omega_k*(1 + z)**2 + 9.2e-5*(1 + z)**4))
    
    result = quad(f, 0, z, args = (Omega_l, Omega_m, w))[0]

    return result

def Dt(z: float, Omega_l: float, Omega_m: float, w: float) -> float:
    """
    Calculates the temporal comoving distance using the input parameters.

    Parameters:
        z (float): The redshift value.
        Omega_l (float): The value of the cosmological constant.
        Omega_m (float): The value of the matter density parameter.
        w (float): The value of the dark energy equation of state.

    Returns:
        The temporal comoving distance.
        
    Notes:  
        -If Omega_k is greater than 0, the universe is positively curved like a 
        sphere, and the correction factor involves a sine function.

        -If Omega_k is equal to 0, the universe is flat, and no correction factor 
        is necessary.

        -If Omega_k is less than 0, the universe is negatively curved like a saddle, 
        and the correction factor involves a hyperbolic sine function.
    """
    
    Omega_k = 1 - (Omega_l + Omega_m + 9.2e-5) # Omega_l + Omega_m + Omega_r + Omega_k = 1
    
    if Omega_k > 0:
        return np.sin(np.sqrt(Omega_k)*Dc(z, Omega_l, Omega_m, w))/np.sqrt(Omega_k)
    elif Omega_k == 0:
        return Dc(z, Omega_l, Omega_m, w)
    elif Omega_k:
        return np.sinh(np.sqrt(abs(Omega_k))*Dc(z, Omega_l, Omega_m, w))/np.sqrt(abs(Omega_k))

def Dl(z: float, Omega_l: float, Omega_m: float, w: float) -> float:
    """
    Calculates the luminosity distance using the input parameters.

    Parameters:
        z (float): The redshift value.
        Omega_l (float): The value of the cosmological constant.
        Omega_m (float): The value of the matter density parameter.
        w (float): The value of the dark energy equation of state.

    Returns:
        The luminosity distance.
    """
    
    return (1 + z) * Dt(z, Omega_l, Omega_m, w)
               
def mu(z: float, Omega_l: float, Omega_m: float, w: float, Hubble: float) -> float:
    """
    Calculates the distance modulus using the input parameters.

    Parameters:
        z (float): The redshift value.
        Omega_l (float): The value of the cosmological constant.
        Omega_m (float): The value of the matter density parameter.
        w (float): The value of the dark energy equation of state.
        Hubble (float): The Hubble parameter.

    Returns:
        The distance modulus.
    """
    
    c = 3e5 #(km/s)
    return 5*np.log10(Dl(z, Omega_l, Omega_m, w)) + 25 + 5*np.log10(c/ Hubble)

def chi2_i(mu_o: float, sigma_o: float, z_obs: float, Omega_l: float, Omega_m: float, w: float, Hubble: float) -> float:
    """
    Calculates the chi-squared value for a single data point using the input parameters.

    Parameters:
        mu_o (float): The observed distance modulus.
        sigma_o (float): The uncertainty in the observed distance modulus.
        z_obs (float): The observed redshift value.
        Omega_l (float): The value of the cosmological constant.
        Omega_m (float): The value of the matter density parameter.
        w (float): The value of the dark energy equation of state.
        Hubble (float): The Hubble parameter.

    Returns:
        The chi-squared value for a single data point.
    """
    
    _mu = mu(z_obs, Omega_l, Omega_m, w, Hubble)
    return ((_mu - mu_o)**2)/sigma_o**2

def chi2(Omega_l: float, Omega_m: float, w: float, Hubble: float, df): 
    """
    Calculates the chi-squared value for a set of data points using the input parameters.

    Parameters:
        Omega_l (float): The value of the cosmological constant.
        Omega_m (float): The value of the matter density parameter.
        w (float): The value of the dark energy equation of state.
        Hubble (float): The Hubble parameter.
        df (pandas.DataFrame): The dataframe containing the observed redshift values, distance moduli, and uncertainties.

    Returns:
        The chi-squared value for the set of data points.
    """
    values = []
    for idx in df.index:
        values.append(chi2_i(df.mu[idx], df.sigma[idx], df.z[idx], Omega_l, Omega_m, w, Hubble))
    result = sum(values)
    return result

def ratio(Omega_l: float, Omega_m: float, w: float, Hubble: float, df, chi2null: float) -> float:
    """
    Calculates the likelihood ratio test statistic for a given set of cosmological
    parameters and data.

    Parameters:
        Omega_l (float): The cosmological constant.
        Omega_m (float): The density of matter.
        w (float): The dark energy equation of state parameter.
        Hubble (float): The Hubble constant in units of km/s/Mpc.
        df (pandas.DataFrame): A DataFrame with columns for redshift (z), observed
            distance modulus (mu_o), and uncertainty (sigma_o).
        chi2null (float): The chi-squared value for the null model, i.e., the chi-
            squared value for the observed data using the values of the cosmological
            parameters in the null hypothesis.

    Returns:
        float: The difference in chi-squared values between the null and alternative
            models.
    """
    
    return chi2(Omega_l, Omega_m, w, Hubble, df) - chi2null


#----------------- Processing Functions -----------------#
def _zip_vars(fixed: dict, x) -> dict:
    """
    Given a dictionary of fixed parameters and an array of values, this function creates 
    an ordered dictionary estimating the position of each value in the array for the 
    remaining unfixed parameters (in the order Omega_l, Omega_m, w, and Hubble). 

    Args:
        fixed (dict): A dictionary of fixed parameters.
        x: An array of values with length equal to the number of unfixed parameters.

    Returns:
        A dictionary with keys corresponding to unfixed parameters and values corresponding 
        to the corresponding values in x.
    """
    
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
    """
    Minimizes the given function h using the Nelder-Mead method and returns the optimal value of x.

    Parameters:
        args (tuple): A tuple containing the values of the arguments to be passed to the function h.

    Returns:
        x (array): The optimal value of x that minimizes the function h.
    """

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

def h(x, fixed: dict, df) -> float:
    """
    A wrapper for the chi2 function that enables minimize to pass an array of arguments 
    by creating a dictionary of variables to be optimized. 

    Args:
        x: An array of values to be optimized.
        fixed (dict): A dictionary of fixed variables.
        df: A pandas dataframe containing the data to be fitted.

    Returns:
        The value of the chi-squared function for the given parameters.
    """
    
    # Creates dictionary of variables to be optimized
    _vars = _zip_vars(fixed, x)

    g = partial(chi2, df=df, **fixed) # Creates partial function fixing variables set by `fixed`
    # Unzips `_vars` as named args to `g`
    return g(**_vars) 

def generate_grid(config, grid_size):
    """
    Generates a grid of values for the given configuration and grid size, and finds the optimal values of the free parameters
    using the _minimize function.

    Parameters:
        config (dict): A dictionary containing the configuration for the grid search.
        grid_size (int): The size of the grid to be generated.

    Yields:
        result (tuple): The optimal value of x that minimizes the function h for the current set of free parameters.
    """    
    def generate_free_params(config, grid_size):
        standard_x0 = {'Omega_l':0.74, 'Omega_m':0.26, 'w':-1,  'Hubble':70}
        linspaces = []
        for free in config['free']:
            linspaces.append(np.linspace(config['grid'][free][0], config['grid'][free][1], grid_size))
        cartesian = cartesian_product(*linspaces)
        for i in cartesian:

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
    
    