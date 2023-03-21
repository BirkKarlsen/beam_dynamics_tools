'''
Diagnostics function object to simulations in the SPS and LHC.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.constants import c

import analytical_functions.longitudinal_beam_dynamics as lbd
import beam_profiles.bunch_profile_tools as bpt
import data_visualisation.plot_profiles as ppr
import data_visualisation.plot_cavity_signals as pcs

class LHCDiagnostics(object):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the LHC.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, save_to, get_from,
                 n_bunches, setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000):

        self.turn = 0
        self.turns_after_injection = 0
        self.injection_number = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = LHCCavityLoop

        self.save_to = save_to
        self.get_from = get_from
        self.n_bunches = n_bunches

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.ind_cont = 0
        self.n_cont = int(self.tracker.rf_params.n_turns/self.dt_cont)
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        elif setting == 1:
            self.perform_measurements = getattr(self, 'lhc_2022_power_md')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')


    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.perform_measurements()
        self.turn += 1

    def injection(self, beam_ID, bucket):
        r'''Injection of beam from the SPS into the LHC.'''

        # import beam
        injected_beam = np.load(self.get_from + 'generated_beams/' + beam_ID + 'simulated_beam.npy')

        # place beam in the correct RF bucket
        sps_lhc_dt = (((2 * np.pi * lbd.R_SPS)/(lbd.h_SPS * c * lbd.beta)) -
                      self.tracker.rf_params.t_rf[0, self.tracker.counter[0]])/2
        lhc_bucket_dt = bucket * self.tracker.rf_params.t_rf[0, self.tracker.counter[0]]
        phase_offset_dt = self.tracker.rf_params.phi_rf[:, self.tracker.counter[0]] * \
                          self.tracker.rf_params.omega_rf[:, self.tracker.counter[0]]

        injected_beam[0, :] = injected_beam[0, :] - sps_lhc_dt + lhc_bucket_dt + phase_offset_dt

        # Add macroparticles to beam class
        self.tracker.beam += injected_beam

    def empty_measurement(self):
        pass

    def standard_measurement(self):
        r'''Default measurement rutine for LHC simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            self.max_power = np.zeros(self.n_cont)

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.beam_profile = np.zeros((self.n_cont, len(self.profile.n_macroparticles[::2])))

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            self.max_power[self.ind_cont] = np.max(self.cl.generator_power()[-self.cl.n_coarse:])

            self.beam_profile[self.ind_cont, :] = self.profile.n_macroparticles[::2] * self.tracker.beam.ratio

            bpos, blen = bpt.extract_bunch_position(self.profile.bin_centers, self.profile.n_macroparticles,
                                                    heighFactor=1000, wind_len=5)
            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

            # Save
            np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plot
            pcs.plot_generator_power(self.cl, self.turn, self.save_to + 'figures/')
            pcs.plot_cavity_voltage(self.cl, self.turn, self.save_to + 'figures/')
            pcs.plot_max_power(self.max_power, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

            # Save
            np.save(self.save_to + 'data/' + f'gen_power_{self.turn}.npy',
                    self.cl.generator_power()[-self.cl.n_coarse:])
            np.save(self.save_to + 'data/' + f'ant_volt_{self.turn}.npy',
                    self.cl.V_ANT[-self.cl.n_coarse:])

        plt.clf()
        plt.cla()
        plt.close()

    def lhc_2022_power_md(self):
        r'''Injection of beams and measurements done that are similar to the LHC RF MD on power performed
        during November 2022.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            self.max_power = np.zeros(self.n_cont)
            self.power_transient = np.zeros((500, self.cl.n_coarse))

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.beam_profile = np.zeros((self.n_cont, len(self.profile.n_macroparticles[::2])))

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            self.max_power[self.ind_cont] = np.max(self.cl.generator_power()[-self.cl.n_coarse:])

            self.beam_profile[self.ind_cont, :] = self.profile.n_macroparticles[::2] * self.tracker.beam.ratio

            bpos, blen, bint = bpt.extract_bunch_parameters(self.profile.bin_centers, self.profile.n_macroparticles *
                                                            self.tracker.beam.ratio,
                                                            heighFactor=1000, distance=500, wind_len=5)
            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

            # Save
            np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plot
            pcs.plot_generator_power(self.cl, self.turn, self.save_to + 'figures/')
            pcs.plot_cavity_voltage(self.cl, self.turn, self.save_to + 'figures/')
            pcs.plot_max_power(self.max_power, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

            # Save
            np.save(self.save_to + 'data/' + f'gen_power_{self.turn}.npy',
                    self.cl.generator_power()[-self.cl.n_coarse:])
            np.save(self.save_to + 'data/' + f'ant_volt_{self.turn}.npy',
                    self.cl.V_ANT[-self.cl.n_coarse:])

        if self.turns_after_injection >= 0 and self.turns_after_injection < 500:
            # Gather power transients during the first 500 turns after the three injections
            self.power_transient[self.turns_after_injection, :] = self.cl.generator_power()[-self.cl.n_coarse:]
            self.turns_after_injection += 1

            # Save power transients after 500 turns
            if self.turns_after_injection == 500:
                np.save(self.save_to + 'data/' + f'power_transient_injection_{self.injection_number}.npy',
                        self.power_transient)

        # Close all figures for this turn
        plt.clf()
        plt.cla()
        plt.close()

        # Injection of different beams into the LHC.
        if self.turn == 3000:
            # 36b injection
            beam_ID = 'LHC_power_MD_BCMS_36b/'
            self.injection(beam_ID, bucket=12 * 10 + 200)
            self.injection_number += 1
            self.turns_after_injection = 0
            self.power_transient = np.zeros((500, self.cl.n_coarse))
            print(f'Injected 36 bunches in bucket {12 * 10 + 200}!')

        if self.turn == 10000:
            # 144b injection
            beam_ID = 'LHC_power_MD_BCMS_144b/'
            self.injection(beam_ID, bucket=12 * 10 + 200 + 36 * 10 + 200)
            self.injection_number += 1
            self.turns_after_injection = 0
            self.power_transient = np.zeros((500, self.cl.n_coarse))
            print(f'Injected 144 bunches in bucket {12 * 10 + 36 * 10 + 2 * 200}!')


class SPSDiagnostics(object):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the SPS.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, SPSCavityFeedback, save_to, get_from,
                 n_bunches, setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = SPSCavityFeedback

        self.save_to = save_to
        self.get_from = get_from
        self.n_bunches = n_bunches

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.ind_cont = 0
        self.n_cont = int(self.tracker.rf_params.n_turns / self.dt_cont)
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
        r'''Standard measurements done in SPS simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.beam_profile = np.zeros((self.n_cont, len(self.profile.n_macroparticles[::2])))

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0:
            self.beam_profile[self.ind_cont, :] = self.profile.n_macroparticles[::2] * self.tracker.beam.ratio

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