'''
Useful mathematical functions.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np


def to_dB(x):
    r'''
    Converts a linear signal to dB.

    :param x: numpy-array signal
    :return: numpy-array signal in dB
    '''
    return 20 * np.log10(x)


def to_linear(x):
    r'''
    Converts a signal in dB to linear.

    :param x: numpy-array signal in dB
    :return: numpy-array signal in linear
    '''
    return 10**(x/20)