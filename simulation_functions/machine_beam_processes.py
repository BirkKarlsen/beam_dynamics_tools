'''
Functions to edit the momentum program of BLonD simulations.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.interpolate import interp1d

import data_visualisation.make_plots_pretty
from general_functions.analysis_utilities import find_nearest

from blond.input_parameters.ring_options import load_data, RingOptions


def fetch_momentum_program(fname, C, particle_mass, scale_data=1e9, target_momentum=None):
    r'''
    Fetching momentum program from a CSV-file.

    :param fname: name of momentum program file
    :param C: circumferenc of the machine
    :param particle_mass: mass of particles in accelerator
    :param scale_data: option to scale data
    :return: numpy-array with turn-by-turn momentum value
    '''

    time, data = load_data(fname, ignore=2, delimiter=',')
    data *= scale_data

    if target_momentum is not None:
        data, time = cut_momentum_program(target_momentum, data, time)

    ring_option = RingOptions()
    time_interp, momentum = ring_option.preprocess(mass=particle_mass, circumference=C,
                                                   time=time, momentum=data)

    return np.ascontiguousarray(momentum)


def cut_momentum_program(target_momentum, momentum_program, program_time):
    r'''
    Cut the momentum program short at a chosen momentum value.

    :param target_momentum: Maximum momentum reached
    :param momentum_program: Original momentum program
    :param program_time: Original momentum program in time
    :return: new momentum program and time
    '''

    data_val, data_ind = find_nearest(momentum_program - target_momentum, 0)
    mom_func = interp1d(program_time[data_ind - 3: data_ind + 3],
                        momentum_program[data_ind - 3: data_ind + 3] - target_momentum)

    target_time = bisect(mom_func, program_time[data_ind - 1], program_time[data_ind + 1])

    if data_val < target_momentum:
        momentum_program = np.concatenate((momentum_program[:data_ind], np.array([target_momentum])))
        program_time = np.concatenate((program_time[:data_ind], np.array([target_time])))
    else:
        momentum_program = np.concatenate((momentum_program[:data_ind - 1], np.array([target_momentum])))
        program_time = np.concatenate((program_time[:data_ind - 1], np.array([target_time])))

    return momentum_program, program_time


def plot_momentum_program(momentum_program, momentum_cut, scale_data=1e-9):
    r'''
    Making a plot of the momentum program with a potential cut of the program.

    :param momentum_program: numpy-array of the turn-by-turn momentum
    :param momentum_cut: momentum-value to reach
    :param scale_data: option to scale data
    '''

    plt.figure()
    plt.title('Momentum Program')
    plt.xlabel(r'Turns [-]')
    plt.ylabel(r'Momentum [GeV]')
    plt.plot(momentum_program * scale_data, c='black')
    plt.hlines(momentum_cut * scale_data, 0, len(momentum_program), colors='r')
    plt.xlim((0, len(momentum_program) - 1))
    plt.grid()
