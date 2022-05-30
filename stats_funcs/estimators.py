import numpy as np
from numpy import array
from scipy.stats import mode
 
def one(sample: array) -> float:
    return sum(sample)/len(sample)

def two(sample: array) -> float:
    if len(sample) < 10:
        return one(sample)
    else:
        return sum(sample)/10

def three(sample: array) -> float:
    return sum(sample)/(len(sample) - 1)

def four(sample: array) -> float:
    return 1.8

def five(sample: array) -> float:
    return np.prod(sample)**(1/len(sample))

def six(sample: array) -> float:
    return mode(sample)[0][0]

def seven(sample: array) -> float:
    return (min(sample) + max(sample))/2

def eight(sample: array) -> float:
    if len(sample) % 2 == 0:
        n = int(len(sample)/2)
        s = 0
        for i in range(n):
            s += sample[2*i]
        return s/n
    else:
        s = 0
        n = int((len(sample)-1)/2)
        for i in range():
            s += sample[2*i]
        return s/n

    