'''
General functions to analyse varius types of data.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx