r'''
Diagnostics function object to simulations in the SPS.

Author: Birk Emil Karlsen-Bæck
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

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        elif setting == 1:
            self.perform_measurements = getattr(self, 'feedforward_measurement')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')

    def standard_measurement(self):
        r'''Standard measurements done in SPS simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))

            self.beam_profile = np.zeros((self.n_ld, len(self.profile.n_macroparticles)))

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Line density measurement
        if self.turn % self.dt_ld == 0:
            self.beam_profile[self.ind_ld, :] = self.profile.n_macroparticles * self.tracker.beam.ratio
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:

            bpos, blen, bint = bpt.extract_bunch_parameters(self.profile.bin_centers, self.profile.n_macroparticles,
                                                            heighFactor=1000, distance=500, wind_len=5)
            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1,
                                    self.save_to + 'figures/')

            # Save
            np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)

        plt.clf()
        plt.cla()
        plt.close()

    def feedforward_measurement(self):
        r'''Measurements for benchmarking the SPS FF.'''

        # Setting up measurement arrays
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            # Arrays for profile tracking
            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.beam_profile = np.zeros((self.n_ld, len(self.profile.n_macroparticles)))
            self.phase_offset = np.zeros((self.n_cont, self.n_bunches))

            # Arrays for FF+OTFB tracking
            self.vcav_4sec = np.zeros((self.n_cont, self.cl.OTFB_2.n_coarse), dtype=complex)
            self.vcav_3sec = np.zeros((self.n_cont, self.cl.OTFB_1.n_coarse), dtype=complex)
            self.power_4sec = np.zeros((self.n_cont, self.cl.OTFB_2.n_coarse), dtype=complex)
            self.power_3sec = np.zeros((self.n_cont, self.cl.OTFB_1.n_coarse), dtype=complex)

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Line density measurement
        if self.turn % self.dt_ld == 0:
            self.beam_profile[self.ind_ld, :] = self.profile.n_macroparticles * self.tracker.beam.ratio
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            # Beam related analysis

            bpos, blen, bint = bpt.extract_bunch_parameters(self.profile.bin_centers,
                                                            self.profile.n_macroparticles,
                                                            heighFactor=1000, distance=500, wind_len=5)
            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos
            batch_len, n_batches = bpt.find_batch_length(bpos, bunch_spacing=25)

            self.phase_offset[self.ind_cont, :] = bpt.bunch_by_bunch_spacing(bpos, batch_len=batch_len).flatten()

            # Analysis of control system
            self.vcav_3sec[self.ind_cont, :] = self.cl.OTFB_1.V_ANT[-self.cl.OTFB_1.n_coarse:]
            self.vcav_4sec[self.ind_cont, :] = self.cl.OTFB_2.V_ANT[-self.cl.OTFB_2.n_coarse:]
            self.power_3sec[self.ind_cont, :] = self.cl.OTFB_1.calc_power()[-self.cl.OTFB_1.n_coarse:]
            self.power_4sec[self.ind_cont, :] = self.cl.OTFB_2.calc_power()[-self.cl.OTFB_2.n_coarse:]

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1,
                                  self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1,
                                    self.save_to + 'figures/')
            ppr.plot_bunch_phase_offsets(self.phase_offset[self.ind_cont - 1, :], self.turn,
                                         self.save_to + 'figures/')

            # Save
            np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)
            np.save(self.save_to + 'data/' + 'phase_offset.npy', self.phase_offset)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
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

        plt.clf()
        plt.cla()
        plt.close()