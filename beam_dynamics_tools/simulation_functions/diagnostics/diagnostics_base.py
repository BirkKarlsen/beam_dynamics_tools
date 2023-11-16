r'''
Diagnostics function object to get data from simulations.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import beam_dynamics_tools.beam_profiles.bunch_profile_tools as bpt
import beam_dynamics_tools.data_visualisation.plot_profiles as ppr
import os

class Diagnostics(object):
    r'''Object for diagnostics of both the beam and the RF system in simulations.'''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, CavityLoop, Ring, save_to, get_from,
                 n_bunches, dt_cont=1, dt_beam=1000, dt_cl=1000, dt_prfl=500, dt_ld=25):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = CavityLoop
        self.ring = Ring

        self.save_to = save_to
        self.get_from = get_from
        self.n_bunches = n_bunches

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.ind_cont = 0
        self.n_cont = int(self.tracker.rf_params.n_turns / self.dt_cont)
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl
        self.dt_prfl = dt_prfl

        self.dt_ld = dt_ld
        self.ind_ld = 0
        self.n_ld = int(self.tracker.rf_params.n_turns / self.dt_ld)

        self.time_turns = np.linspace(0,
                                      (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                      self.n_cont)

        self.perform_measurements = getattr(self, 'empty_measurement')

        if not os.path.isdir(self.save_to + 'figures/'):
            os.mkdir(self.save_to + 'figures/')

        if not os.path.isdir(self.save_to + 'data/'):
            os.mkdir(self.save_to + 'data/')

    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.reposition_profile_edges()
        self.perform_measurements()
        self.turn += 1

    def reposition_profile_edges(self):
        r'''Function to reposition profile cuts'''
        if self.turn % self.dt_prfl == 0:
            # Modify cuts of the Beam Profile
            self.tracker.beam.statistics()
            self.profile.cut_options.track_cuts(self.tracker.beam)
            self.profile.set_slices_parameters()

    def measure_uncaptured_losses(self):
        r'''Method to measure uncaptured losses.'''

        self.tracker.beam.losses_separatrix(self.ring, self.tracker.rf_params)
        uncaptured_beam = self.tracker.beam.n_macroparticles_lost * self.tracker.beam.ratio

        return uncaptured_beam

    def init_beam_measurements(self):
        r'''Method to initiate the beam related measurements'''
        self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
        self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
        self.bunch_intensities = np.zeros((self.n_cont, self.n_bunches))
        self.bunch_losses = np.zeros((self.n_cont, self.n_bunches))

        self.beam_profile = np.zeros((self.n_ld, len(self.profile.n_macroparticles)))

    def beam_frequent_measurements(self):
        r'''Method to frequency measure the beam parameters'''
        bpos, blen, bint = bpt.extract_bunch_parameters(self.profile.bin_centers, self.profile.n_macroparticles *
                                                        self.tracker.beam.ratio,
                                                        heighFactor=1000 * self.tracker.beam.ratio,
                                                        distance=500,
                                                        wind_len=self.tracker.rf_params.t_rf[0, 0] * 1e9,
                                                        n_bunches=self.n_bunches)

        self.bunch_lengths[self.ind_cont, :] = blen
        self.bunch_positions[self.ind_cont, :] = bpos
        self.bunch_intensities[self.ind_cont, :] = bint

    def beam_infrequent_measurements(self):
        r'''Method to plot bunch parameters and save the data'''
        # Plots
        ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
        ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
        ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

        # Save
        np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
        np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
        np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)
        np.save(self.save_to + 'data/' + 'bunch_intensities.npy', self.bunch_intensities)
        np.save(self.save_to + 'data/' + 'bunch_losses.npy', self.bunch_losses)

    def line_density_measurement(self):
        self.beam_profile[self.ind_ld, :] = self.profile.n_macroparticles * self.tracker.beam.ratio

    def empty_measurement(self):
        r'''Dummy measurement for simulations not needing output.'''
        pass
