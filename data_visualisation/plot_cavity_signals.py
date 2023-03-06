'''
Functions to plot signals from the SPS and LHC Cavity Loops.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt

import data_visualisation.make_plots_pretty

# LHC Cavity Loop
def plot_generator_power(LHCCavityLoop, turn, save_to):
    power = LHCCavityLoop.generator_power()[-LHCCavityLoop.n_coarse:]
    t = LHCCavityLoop.rf_centers

    plt.figure()
    plt.title(f'Generator Power, Turn {turn}')
    plt.plot(t * 1e6, power/1e3)
    plt.xlabel(r'$\Delta t$ [$\mu$s]')
    plt.ylabel(r'Power [kW]')
    plt.savefig(save_to + f'gen_power_{turn}.png')

def plot_cavity_voltage(LHCCavityLoop, turn, save_to):
    t = LHCCavityLoop.rf_centers
    volt = LHCCavityLoop.V_ANT[-LHCCavityLoop.n_coarse:]

    plt.figure()
    plt.title(f'Antenna Voltage, Turn {turn}')
    plt.plot(t * 1e6, np.abs(volt) / 1e6)
    plt.xlabel(r'$\Delta t$ [$\mu$s]')
    plt.ylabel(r'Antenna Voltage [MV]')
    plt.savefig(save_to + f'ant_volt_{turn}.png')

def plot_max_power(power, time, j, save_to):
    fig, ax = plt.subplots()

    ax.set_title('Max Power')

    ax.plot(time[:j], power[:j]/1e3)
    ax.set_xlabel(r'Time since injection [s]')
    ax.set_ylabel(r'Power [kW]')

    fig.savefig(save_to + 'max_power.png')


# SPS Cavity Loop