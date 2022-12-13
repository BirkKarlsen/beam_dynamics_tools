'''
Functions to analyse signals from measurements.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np

def reshape_buffer_data(data, n_points):
    turns = len(data)//n_points
    rdata = np.zeros((n_points, turns), dtype=complex)

    for i in range(turns):
        rdata[:, i] = data[n_points * i: n_points * (i + 1)]

    return rdata