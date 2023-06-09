'''
Analysis of simulations.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import re

from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml

class SimulationAnalysis(object):

    def __init__(self, simulation_name, simulation_dir, simulation_id=0):

        self.simulation_name = simulation_name
        self.simulation_dir = simulation_dir

        self.simulation_parameters = {}
        self.loss_summary = {}

    def get_simulation_parameters(self):
        r'''Gets the values of the parameters used in the simulation from the folder name.'''

        sim_name = self.simulation_name[4:]

        parameter_values = re.findall('[0-9.e-]+', sim_name)

        if 'e' in parameter_values:
            parameter_values.remove('e')
        if '-' in parameter_values:
            parameter_values.remove('-')

        param_name = [sim_name]
        for i in range(len(parameter_values)):
            splits_i = param_name[-1].split(parameter_values[i])
            param_name[-1] = splits_i[0].strip('_')
            param_name.append(splits_i[1].strip('_'))

        if '' in param_name:
            param_name.remove('')

        parameter_values = [float(x) for x in parameter_values]

        for (param, value) in zip(param_name, parameter_values):
            self.simulation_parameters[param] = value

        return self.simulation_parameters

    def get_loss_summary(self):
        r'''Retrieve losses from the loss summary file.'''

        self.loss_summary = fetch_from_yaml('loss_summary.yaml', self.simulation_dir + self.simulation_name + '/')
        return self.loss_summary

    def get_turn_by_turn_quantity(self, filename):
        r'''Retrieves turn-by-turn quantities.'''

        return np.load(self.simulation_dir + '')
