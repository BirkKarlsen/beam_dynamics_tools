r'''
Diagnostics function object to simulations in the SPS.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import matplotlib.pyplot as plt
import os

import beam_dynamics_tools.beam_profiles.bunch_profile_tools as bpt
import beam_dynamics_tools.data_visualisation.plot_profiles as ppr
import beam_dynamics_tools.data_visualisation.plot_cavity_signals as pcs

from beam_dynamics_tools.simulation_functions.diagnostics.diagnostics_base import Diagnostics


class SPSDiagnostics(Diagnostics):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the SPS.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, SPSCavityFeedback, Ring, save_to, get_from,
                 n_bunches, setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000, dt_prfl=500, dt_ld=25):

        super().__init__(RingAndRFTracker, Profile, TotalInducedVoltage, SPSCavityFeedback, Ring, save_to, get_from,
                 n_bunches, dt_cont=dt_cont, dt_beam=dt_beam, dt_cl=dt_cl, dt_prfl=dt_prfl, dt_ld=dt_ld)

        self.init_beam_measurements()
        self.phase_offset = np.zeros((self.n_cont, self.n_bunches))

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        elif setting == 1:
            self.perform_measurements = getattr(self, 'feedforward_measurement')
        elif setting == 2:
            self.perform_measurements = getattr(self, 'fast_measurement')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')

    def init_cavity_loop_measurements(self):
        r'''Method to initialize the measurements of LLRF signals'''
        # Arrays for FF+OTFB tracking
        self.vcav_4sec = np.zeros((self.n_cont, self.cl.OTFB_2.n_coarse), dtype=complex)
        self.vcav_3sec = np.zeros((self.n_cont, self.cl.OTFB_1.n_coarse), dtype=complex)
        self.power_4sec = np.zeros((self.n_cont, self.cl.OTFB_2.n_coarse), dtype=complex)
        self.power_3sec = np.zeros((self.n_cont, self.cl.OTFB_1.n_coarse), dtype=complex)

    def cavity_loop_frequent_measurements(self):
        r'''Method to measure LLRF signals'''
        # Analysis of control system
        self.vcav_3sec[self.ind_cont, :] = self.cl.OTFB_1.V_ANT_COARSE[-self.cl.OTFB_1.n_coarse:]
        self.vcav_4sec[self.ind_cont, :] = self.cl.OTFB_2.V_ANT_COARSE[-self.cl.OTFB_2.n_coarse:]
        self.power_3sec[self.ind_cont, :] = self.cl.OTFB_1.calc_power()[-self.cl.OTFB_1.n_coarse:]
        self.power_4sec[self.ind_cont, :] = self.cl.OTFB_2.calc_power()[-self.cl.OTFB_2.n_coarse:]

    def cavity_loop_infrequency_measurements(self):
        r'''Method to save and plot LLRF signals'''
        # Plot
        pcs.plot_twc_generator_power(self.cl.OTFB_1, self.turn, self.save_to + 'figures/')
        pcs.plot_twc_generator_power(self.cl.OTFB_2, self.turn, self.save_to + 'figures/')
        pcs.plot_twc_gap_voltage(self.cl.OTFB_1, self.turn, self.save_to + 'figures/')
        pcs.plot_twc_gap_voltage(self.cl.OTFB_2, self.turn, self.save_to + 'figures/')

        # Save
        np.save(self.save_to + 'data/' + f'ant_volt_3sec_{self.turn}.npy',
                self.vcav_3sec)
        np.save(self.save_to + 'data/' + f'ant_volt_4sec_{self.turn}.npy',
                self.vcav_4sec)
        np.save(self.save_to + 'data/' + f'gen_power_3sec_{self.turn}.npy',
                self.power_3sec)
        np.save(self.save_to + 'data/' + f'gen_power_4sec_{self.turn}.npy',
                self.power_4sec)

    def standard_measurement(self):
        r'''Standard measurements done in SPS simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            pass

        # Line density measurement
        if self.turn % self.dt_ld == 0:
            self.line_density_measurement()
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            self.beam_frequent_measurements()
            self.bunch_losses[self.ind_cont, :] = self.bunch_intensities[0, :] - \
                                                  self.bunch_intensities[self.ind_cont, :]

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.beam_infrequent_measurements()

        plt.clf()
        plt.cla()
        plt.close()

    def feedforward_measurement(self):
        r'''Measurements for benchmarking the SPS FF.'''

        # Setting up measurement arrays
        if self.turn == 0:
            self.init_cavity_loop_measurements()

        # Line density measurement
        if self.turn % self.dt_ld == 0:
            self.line_density_measurement()
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            # Beam related analysis
            self.beam_frequent_measurements()
            self.bunch_losses[self.ind_cont, :] = self.bunch_intensities[0, :] - \
                                                  self.bunch_intensities[self.ind_cont, :]

            batch_len, n_batches = bpt.find_batch_length(self.bunch_positions[self.ind_cont, :], bunch_spacing=25)

            self.phase_offset[self.ind_cont, :] = bpt.bunch_by_bunch_spacing(self.bunch_positions[self.ind_cont, :],
                                                                             batch_len=batch_len).flatten()

            self.cavity_loop_frequent_measurements()

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.beam_infrequent_measurements()

            # Bunch-by-bunch position variation
            ppr.plot_bunch_phase_offsets(self.phase_offset[self.ind_cont - 1, :], self.turn,
                                         self.save_to + 'figures/')
            np.save(self.save_to + 'data/' + 'phase_offset.npy', self.phase_offset)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.cavity_loop_infrequency_measurements()

        plt.clf()
        plt.cla()
        plt.close()

    def fast_measurement(self):
        r'''Fast measurements done in SPS simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            pass

        # Line density measurement
        if self.turn % self.dt_ld == 0:
            self.line_density_measurement()
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            self.beam_frequent_measurements()
            self.bunch_losses[self.ind_cont, :] = self.bunch_intensities[0, :] - \
                                                  self.bunch_intensities[self.ind_cont, :]

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.beam_infrequent_measurements(plot=False)
