'''
Changes the rcParams in matplotlib.pyplot to make nicer looking plots.

Author: Birk Emil Karlsen-Bæck
'''

import matplotlib.pyplot as plt
import os

if 'birkkarlsen-baeck' in os.getcwd():
    plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{fourier}',
            'font.family': 'serif',
            'font.size': 16
        })
