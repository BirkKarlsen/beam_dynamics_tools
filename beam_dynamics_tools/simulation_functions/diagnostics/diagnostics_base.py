r'''
Diagnostics function object to get data from simulations.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import beam_dynamics_tools.beam_profiles.bunch_profile_tools as bpt
import beam_dynamics_tools.data_visualisation.plot_profiles as ppr
import os
from typing import TYPE_CHECKING

from beam_dynamics_tools.beam_profiles.beam_phase_estimate import SimulatedBeamPhase

from blond.trackers.tracker import RingAndRFTracker
from blond.beam.profile import Profile
from blond.impedances.impedance import TotalInducedVoltage
from blond.llrf.cavity_feedback import CavityFeedback
from blond.input_parameters.ring import Ring

class Diagnostics(object):
    r'''Object for diagnostics of both the beam and the RF system in simulations.'''

    def __init__(
            self, 
            tracker: RingAndRFTracker, 
            profile: Profile, 
            induced_voltage: TotalInducedVoltage, 
            cavity_loop: CavityFeedback, 
            ring: Ring, 
            save_to: str, 
            get_from: str,
            n_bunches: int, 
            dt_cont: int = 1, 
            dt_beam: int = 1000, 
            dt_cl: int = 1000, 
            dt_prfl: int = 500, 
            dt_ld: int = 25
        ) -> None:
        '''Class to analyse and store information about BLonD simulations.

        Args:
            tracker (RingAndRFTracker):
                RF tracker in the BLonD simulation.
            profile (Profile):
                Profile object in the simulation.
            induced_voltage (TotalInducedVoltage):
                TotalInducedVoltage object in the simulation.
            cavity_loop (CavityFeedback):
                CavityFeedback child class in the simulation.
            ring (Ring):
                Ring class in the simulation.
            save_to (str):
                Directory to save all the data and plots to.
            get_from (str):
                Directory to get related data to the simulation from.
            n_bunches (int):
                Number of bunches in the simulation.
            dt_cont (int):
                Turn interval to store frequently sampled signals, e.g. bunch lengths.
            dt_beam (int):
                Turn interval to store infrequently sampled beam based data.
                For example, save the frequently sampled beam related data or make plots.
            dt_cl (int):
                Turn interval to store infrequently sampled data from the cavity loop.
                For example, make plots, full ring signals and/or save the frequently 
                stored data.
            dt_prfl (int):
                Turn interval to shift the beam profile window as the beam drifts around.
            dt_ld (int):
                Turn interval to store beam profiles.
        '''

        self.turn = 0

        self.tracker = tracker
        self.profile = profile
        self.induced_voltage = induced_voltage
        self.cl = cavity_loop
        self.ring = ring

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

        self.time_turns = np.linspace(
            0, (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
            self.n_cont
        )

        self.phase_measurement = SimulatedBeamPhase(
            profile, tracker, n_bunches=n_bunches, n_bunches_init=n_bunches
        )

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
        self.bunch_phases = np.zeros((self.n_cont, self.n_bunches))

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

        self.phase_measurement.set_number_of_bunches(len(blen[blen > 0]))
        self.bunch_phases[self.ind_cont, :self.phase_measurement.n_bunches] = self.phase_measurement.extract_phases()

    def beam_infrequent_measurements(self, plot=True):
        r'''Method to plot bunch parameters and save the data'''
        # Plots
        if plot:
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

        # Save
        np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
        np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
        np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)
        np.save(self.save_to + 'data/' + 'bunch_intensities.npy', self.bunch_intensities)
        np.save(self.save_to + 'data/' + 'bunch_losses.npy', self.bunch_losses)
        np.save(self.save_to + 'data/' + 'bunch_phases.npy', self.bunch_phases)

    def line_density_measurement(self):
        self.beam_profile[self.ind_ld, :] = self.profile.n_macroparticles * self.tracker.beam.ratio

    def empty_measurement(self):
        r'''Dummy measurement for simulations not needing output.'''
        pass
