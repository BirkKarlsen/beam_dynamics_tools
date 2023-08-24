
import numpy as np
import os

from beam_dynamics_tools.simulation_functions.result_analysis.result_analysis import SimulationAnalysis
from beam_dynamics_tools.data_management.importing_data import fetch_from_yaml, sort_measurements


class LHCSimulationAnalysis(SimulationAnalysis):

    def __init__(self, simulation_name, simulation_dir, simulation_id=0):

        super().__init__(simulation_name, simulation_dir, simulation_id=simulation_id)

        self.loss_summary = {}
        self.peak_power = None
        self.injection_transients = None
        self.profile_losses = None


    def get_loss_summary(self):
        r'''Retrieve losses from the loss summary file.'''

        self.loss_summary = fetch_from_yaml('loss_summary.yaml', self.simulation_dir + self.simulation_name + '/')
        return self.loss_summary

    def get_peak_power(self):
        self.peak_power = self.get_data('max_power.npy')
        return self.peak_power

    def get_power_transients(self):
        files = []

        for file in os.listdir(self.simulation_dir + self.simulation_name + '/data/'):
            if file.startswith('power_transient'):
                files.append(file)

        self.injection_transients = np.zeros((len(files), 500, 3564))

        for i in range(len(files)):
            self.injection_transients[i, :, :] = self.get_data(files[i])

        return self.injection_transients

    def get_profile_losses(self):
        self.profile_losses = self.get_data('bunch_losses.npy')
        return self.profile_losses



class ParameterScanAnalysis(object):

    def __init__(self, scan_name, scan_dir):

        self.scan_name = scan_name
        self.scan_dir = scan_dir
        self.simulations = []
        self.scanned_parameters = None

    def get_simulations(self):
        r'''Fetch the simulation files.'''

        i = 0
        for directories in os.listdir(self.scan_dir + self.scan_name):
            if not directories.startswith('.'):
                sim_an_i = LHCSimulationAnalysis(directories.strip('/'),
                                                 self.scan_dir + self.scan_name + '/',
                                                 simulation_id=i)
                sim_an_i.get_simulation_parameters()
                self.simulations.append(sim_an_i)
                i += 1

        self.scanned_parameters = self.simulations[0].simulation_parameters.keys()

    def get_parameter_value_from_simulations(self, param):
        values = np.zeros(len(self.simulations))

        for i, simulation in enumerate(self.simulations):
            values[i] = simulation.simulation_parameters[param]

        return values

    def sort_simulations(self, param):
        r'''Sorts simulations after parameter values'''

        def sort_func(simulation):
            return simulation.simulation_parameters[param]

        self.simulations = sort_measurements(self.simulations, sort_func)

    def get_data_from_simulations(self, pname, dims):
        data = np.zeros(dims)

        for i, simulation in enumerate(self.simulations):
            f = getattr(simulation, pname)
            data[i, :] = f()

        return data







