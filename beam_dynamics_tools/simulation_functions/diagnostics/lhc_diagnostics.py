r'''
Diagnostics function object to simulations in the LHC.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.constants import c

import beam_dynamics_tools.analytical_functions.longitudinal_beam_dynamics as lbd
import beam_dynamics_tools.beam_profiles.bunch_profile_tools as bpt
import beam_dynamics_tools.data_visualisation.plot_profiles as ppr
import beam_dynamics_tools.data_visualisation.plot_cavity_signals as pcs
import beam_dynamics_tools.data_management.importing_data as ida

from beam_dynamics_tools.simulation_functions.diagnostics.diagnostics_base import Diagnostics


class LHCDiagnostics(Diagnostics):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the LHC.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, Ring, save_to, get_from,
                 n_bunches, injection_scheme, setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000, dt_prfl=500,
                 dt_ld=25):

        super().__init__(RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, Ring, save_to, get_from,
                 n_bunches, dt_cont=dt_cont, dt_beam=dt_beam, dt_cl=dt_cl, dt_prfl=dt_prfl, dt_ld=dt_ld)

        self.turns_after_injection = 0
        self.injection_number = 0
        self.injection_scheme = injection_scheme
        self.injection_keys = list(injection_scheme.keys())
        self.beam_structure = np.zeros(len(self.injection_keys) + 1, dtype=int)

        self.beam_structure[0] = self.n_bunches
        inj = 1
        for injection, inj_data in self.injection_scheme.items():
            self.n_bunches += int(inj_data[3])
            self.beam_structure[inj] = int(inj_data[3])
            inj += 1

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        elif setting == 1:
            self.perform_measurements = getattr(self, 'measurement_with_injection')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')

    def injection(self, beam_ID, bucket, simulated):
        r'''Injection of beam from the SPS into the LHC.'''

        # import beam
        if simulated:
            injected_beam = np.load(self.get_from + 'generated_beams/' + beam_ID + 'simulated_beam.npy')
        else:
            injected_beam = np.load(self.get_from + 'generated_beams/' + beam_ID + 'generated_beam.npy')

        # place beam in the correct RF bucket
        sps_lhc_dt = (((2 * np.pi * lbd.R_SPS)/(lbd.h_SPS * c * lbd.beta)) -
                      self.tracker.rf_params.t_rf[0, self.tracker.counter[0]])/2
        lhc_bucket_dt = bucket * self.tracker.rf_params.t_rf[0, self.tracker.counter[0]]
        phase_offset_dt = self.tracker.rf_params.phi_rf[:, self.tracker.counter[0]] * \
                          self.tracker.rf_params.omega_rf[:, self.tracker.counter[0]]

        injected_beam[0, :] = injected_beam[0, :] - sps_lhc_dt + lhc_bucket_dt + phase_offset_dt

        # Add macroparticles to beam class
        self.tracker.beam += injected_beam
        self.tracker.beam.intensity = self.tracker.beam.ratio * self.tracker.beam.n_macroparticles
        self.turns_after_injection = 0
        self.injection_number += 1

        # Modify cuts of the Beam Profile
        self.tracker.beam.statistics()
        self.profile.cut_options.track_cuts(self.tracker.beam)
        self.profile.set_slices_parameters()
        self.profile.track()

    def measure_uncaptured_losses(self):
        r'''Method to measure uncaptured losses.'''

        self.tracker.beam.losses_separatrix(self.ring, self.tracker.rf_params)
        uncaptured_beam = self.tracker.beam.n_macroparticles_lost * self.tracker.beam.ratio

        return uncaptured_beam

    def measure_slow_losses(self):
        r'''Method to measure slow losses throughout the simulation.'''
        bunch_losses = np.zeros(self.n_bunches)

        bunches = np.arange(self.beam_structure[0])
        bunch_losses[bunches] = self.bunch_intensities[0, bunches] \
                                - self.bunch_intensities[self.ind_cont, bunches]

        for inj in range(len(self.injection_keys)):
            bunches = bunches[-1] + 1 + np.arange(self.beam_structure[inj + 1])
            bunch_losses[bunches] = self.bunch_intensities[
                                        self.injection_scheme[self.injection_keys[inj]][1] // self.dt_cont, bunches]\
                                    - self.bunch_intensities[self.ind_cont, bunches]

        return bunch_losses

    def measure_ramp_losses(self):
        r'''Method to measure losses at the end of a short ramp.'''

        bucket_height = lbd.rf_bucket_height(self.tracker.voltage[0, self.tracker.counter[0]],
                                             phi_s=self.tracker.phi_s[self.tracker.counter[0]])
        self.tracker.beam.losses_below_energy(-bucket_height)
        losses_from_cut = self.tracker.beam.n_macroparticles_lost * self.tracker.beam.ratio

        return losses_from_cut

    def standard_measurement(self):
        r'''Default measurement rutine for LHC simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            self.max_power = np.zeros(self.n_cont)
            self.power_transient = np.zeros((500, self.cl.n_coarse))

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_intensities = np.zeros((self.n_cont, self.n_bunches))

            self.bunch_losses = np.zeros((self.n_cont, self.n_bunches))
            self.uncaptured_beam = self.measure_uncaptured_losses()

            loss_dict = {'Uncaptured losses': self.uncaptured_beam}
            ida.make_and_write_yaml('loss_summary.yaml', self.save_to, loss_dict)

            self.beam_profile = np.zeros((self.n_ld, len(self.profile.n_macroparticles)))

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Line density measurement
        if self.turn % self.dt_ld == 0 and self.ind_ld < self.n_ld:
            self.beam_profile[self.ind_ld, :] = self.profile.n_macroparticles * self.tracker.beam.ratio

            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0 and self.ind_cont < self.n_cont:
            self.max_power[self.ind_cont] = np.max(self.cl.generator_power()[-self.cl.n_coarse:])

            bpos, blen, bint = bpt.extract_bunch_parameters(self.profile.bin_centers,
                                                            self.profile.n_macroparticles * self.tracker.beam.ratio,
                                                            heighFactor=1000 * self.tracker.beam.ratio, wind_len=2.5,
                                                            n_bunches=self.n_bunches)
            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos
            self.bunch_intensities[self.ind_cont, :] = bint
            self.bunch_losses[self.ind_cont, :] = self.bunch_intensities[0, :] - \
                                                  self.bunch_intensities[self.ind_cont, :]

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_total_losses(self.bunch_losses, self.time_turns,
                                  self.ind_cont - 1, self.save_to + 'figures/',
                                  caploss=self.uncaptured_beam)

            # Save
            np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)
            np.save(self.save_to + 'data/' + 'bunch_intensities.npy', self.bunch_intensities)
            np.save(self.save_to + 'data/' + 'bunch_losses.npy', self.bunch_losses)

            if self.turn == self.tracker.rf_params.n_turns - 1:

                self.losses_from_cut = self.measure_ramp_losses()
                loss_dict = {'Losses after ramp': self.losses_from_cut}
                ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)


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
            np.save(self.save_to + 'data/' + 'max_power.npy', self.max_power)

        if self.turns_after_injection >= 0 and self.turns_after_injection < 500:
            # Gather power transients during the first 500 turns after the three injections
            self.power_transient[self.turns_after_injection, :] = self.cl.generator_power()[-self.cl.n_coarse:]
            self.turns_after_injection += 1

            # Save power transients after 500 turns
            if self.turns_after_injection == 500:
                np.save(self.save_to + 'data/' + f'power_transient_injection_{self.injection_number}.npy',
                        self.power_transient)
                self.power_transient = np.zeros((500, self.cl.n_coarse))

        plt.clf()
        plt.cla()
        plt.close()

    def measurement_with_injection(self):
        r'''Injection of beams and measurements.'''

        # Injection of different beams into the LHC.
        if self.injection_number < len(self.injection_keys) and \
                self.turn == self.injection_scheme[self.injection_keys[self.injection_number]][1]:
            beam_ID = self.injection_keys[self.injection_number] + '/'
            bucket = self.injection_scheme[self.injection_keys[self.injection_number]][0]
            simulated = bool(self.injection_scheme[self.injection_keys[self.injection_number]][2])
            self.injection(beam_ID, bucket=bucket, simulated=simulated)
            print(f'Injected {beam_ID} in bucket {bucket}!')

            self.uncaptured_beam = self.measure_uncaptured_losses()
            loss_dict = {f'Uncaptured losses {self.injection_number}': self.uncaptured_beam}
            ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.time_turns = np.linspace(0,
                                          (self.tracker.rf_params.n_turns - 1) * self.tracker.rf_params.t_rev[0],
                                          self.n_cont)

            self.max_power = np.zeros(self.n_cont)
            self.power_transient = np.zeros((500, self.cl.n_coarse))

            print(f'Found {self.n_bunches} to be injected in total')

            self.bunch_positions = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_lengths = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_intensities = np.zeros((self.n_cont, self.n_bunches))
            self.bunch_losses = np.zeros((self.n_cont, self.n_bunches))

            self.uncaptured_beam = self.measure_uncaptured_losses()

            loss_dict = {f'Uncaptured losses {self.injection_number}': self.uncaptured_beam}
            ida.make_and_write_yaml('loss_summary.yaml', self.save_to, loss_dict)

            self.beam_profile = np.zeros((self.n_ld, len(self.profile.n_macroparticles)))

            if not os.path.isdir(self.save_to + 'figures/'):
                os.mkdir(self.save_to + 'figures/')

            if not os.path.isdir(self.save_to + 'data/'):
                os.mkdir(self.save_to + 'data/')

        # Line density measurement
        if self.turn % self.dt_ld == 0 and self.ind_ld < self.n_ld:
            self.beam_profile[self.ind_ld, :] = self.profile.n_macroparticles * self.tracker.beam.ratio
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0 and self.ind_cont < self.n_cont:
            self.max_power[self.ind_cont] = np.max(self.cl.generator_power()[-self.cl.n_coarse:])

            bpos, blen, bint = bpt.extract_bunch_parameters(self.profile.bin_centers, self.profile.n_macroparticles *
                                                            self.tracker.beam.ratio,
                                                            heighFactor=1000 * self.tracker.beam.ratio,
                                                            distance=500, wind_len=2.5,
                                                            n_bunches=self.n_bunches)

            self.bunch_lengths[self.ind_cont, :] = blen
            self.bunch_positions[self.ind_cont, :] = bpos
            self.bunch_intensities[self.ind_cont, :] = bint
            self.bunch_losses[self.ind_cont, :] = self.measure_slow_losses()

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            # Plots
            ppr.plot_profile(self.profile, self.turn, self.save_to + 'figures/')
            ppr.plot_bunch_length(self.bunch_lengths, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_bunch_position(self.bunch_positions, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')
            ppr.plot_total_losses(self.bunch_losses, self.time_turns,
                                  self.ind_cont - 1, self.save_to + 'figures/',
                                  caploss=self.uncaptured_beam, beam_structure=self.beam_structure)

            # Save
            np.save(self.save_to + 'data/' + 'beam_profiles.npy', self.beam_profile)
            np.save(self.save_to + 'data/' + 'bunch_lengths.npy', self.bunch_lengths)
            np.save(self.save_to + 'data/' + 'bunch_positions.npy', self.bunch_positions)
            np.save(self.save_to + 'data/' + 'bunch_intensities.npy', self.bunch_intensities)
            np.save(self.save_to + 'data/' + 'bunch_losses.npy', self.bunch_losses)

            if self.turn == self.tracker.rf_params.n_turns - 1:
                self.losses_from_cut = self.measure_ramp_losses()
                loss_dict = {'Losses after ramp': self.losses_from_cut}
                ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)

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
            np.save(self.save_to + 'data/' + 'max_power.npy', self.max_power)

        if self.turns_after_injection >= 0 and self.turns_after_injection < 500:
            # Gather power transients during the first 500 turns after the three injections
            self.power_transient[self.turns_after_injection, :] = self.cl.generator_power()[-self.cl.n_coarse:]
            self.turns_after_injection += 1

            # Save power transients after 500 turns
            if self.turns_after_injection == 500:
                np.save(self.save_to + 'data/' + f'power_transient_injection_{self.injection_number}.npy',
                        self.power_transient)
                self.power_transient = np.zeros((500, self.cl.n_coarse))

        # Close all figures for this turn
        plt.clf()
        plt.cla()
        plt.close()
