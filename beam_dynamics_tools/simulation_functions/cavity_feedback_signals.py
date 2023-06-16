'''
File to analyse and plot cavity signals in simulations.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import matplotlib.pyplot as plt


def plot_one_cavity_signal(signal, signal_name, CavityLoop, real_and_imag=False):
    r'''Function to plot a single cavity signal in either real and imaginary or
    amplitude and phase.'''

    plt.figure()
    plt.title(signal_name)
    if real_and_imag:
        plt.plot(signal.real, c='r', label='real')
        plt.plot(signal.imag, c='b', label='imag')
    else:
        plt.plot(np.abs(signal), c='r', label='absolute')

    plt.legend()


def plot_all_cl_signals_lhc(LHCCavityLoop, real_and_imag=False):
    r'''Function to plot all signals in LHCCavityLoop in either real and imaginary or
    amplitude and phase.'''

    plot_one_cavity_signal(LHCCavityLoop.V_ANT, 'V antenna', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_FB_IN, 'Feedback in', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_AC_IN, 'AC coupler in', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_AN_IN, 'Analog feedback in', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_AN_OUT, 'Analog feedback out', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_DI_OUT, 'Digital feedback out', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_OTFB, 'One-turn feedback out', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_OTFB_INT, 'FIR filter in', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_FIR_OUT, 'FIR filter out', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_FB_OUT, 'Feedback out', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.V_SWAP_OUT, 'SWAP out', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.I_GEN, 'Generator current', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.I_BEAM, 'RF beam current', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.I_TEST, 'Test current', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.TUNER_INPUT, 'Tuner in', LHCCavityLoop, real_and_imag=real_and_imag)
    plot_one_cavity_signal(LHCCavityLoop.TUNER_INTEGRATED, 'Tuner CIC out', LHCCavityLoop, real_and_imag=real_and_imag)

