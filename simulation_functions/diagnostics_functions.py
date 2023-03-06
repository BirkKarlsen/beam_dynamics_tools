'''
Diagnostics function object to simulations in the SPS and LHC.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import beam_profiles.bunch_profile_tools as bpt
import data_visualisation.plot_profiles as ppr
import data_visualisation.plot_cavity_signals as pcs

class LHCDiagnostics(object):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the LHC.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, save_to, n_bunches,
                 setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = LHCCavityLoop

        self.save_to = save_to
        self.n_bunches = n_bunches

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.ind_cont = 0
        self.n_cont = int(self.tracker.rf_params.n_turns/self.dt_cont)
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')


    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.perform_measurements()
        self.turn += 1


    def empty_measurement(self):
        pass


    def standard_measurement(self):
        r'''Default measurement rutine for LHC simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.T_rev[0],
                                          self.n_cont)

            self.max_power = np.zeros(self.n_cont)

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.beam_profile = np.zeros((self.n_cont, self.profile.n_slices))

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            self.max_power[self.ind_cont] = np.max(self.cl.generator_power()[-self.cl.n_coarse:])

            self.beam_profile[self.ind_cont, :] = self.profile.n_macroparticles

            bpos, blen = bpt.extract_bunch_position(self.profile.bin_centers, self.profile.n_macroparticles,
                                                    heighFactor=1000)
            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to)
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to)
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to)

            # Save
            np.save(self.save_to + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'bunch_positions.npy', self.bunch_positions)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0:
            # Plot
            pcs.plot_generator_power(self.cl, self.turn, self.save_to)
            pcs.plot_cavity_voltage(self.cl, self.turn, self.save_to)
            pcs.plot_max_power(self.max_power, self.time_turns, self.ind_cont - 1, self.save_to)

            # Save
            np.save(self.save_to + f'gen_power_{self.turn}.npy', self.cl.generator_power()[-self.cl.n_coarse:])
            np.save(self.save_to + f'ant_volt_{self.turn}.npy', self.cl.V_ANT[-self.cl.n_coarse:])

        plt.clf()


class SPSDiagnostics(object):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the SPS.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, SPSCavityFeedback, save_to, n_bunches,
                 setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = SPSCavityFeedback

        self.save_to = save_to
        self.n_bunches = n_bunches

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')


    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.perform_measurements()
        self.turn += 1


    def empty_measurement(self):
        pass


    def standard_measurement(self):
        pass
