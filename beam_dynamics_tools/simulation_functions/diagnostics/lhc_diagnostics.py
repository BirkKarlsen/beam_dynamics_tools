r'''
Diagnostics function object to simulations in the LHC.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.constants import c

import beam_dynamics_tools.data_visualisation.plot_profiles as ppr
import beam_dynamics_tools.data_visualisation.plot_cavity_signals as pcs
import beam_dynamics_tools.data_management.importing_data as ida

from beam_dynamics_tools.simulation_functions.diagnostics.diagnostics_base import Diagnostics

from blond.trackers.utilities import separatrix


class LHCDiagnostics(Diagnostics):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the LHC.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, Ring, save_to, get_from,
                 n_bunches, injection_scheme, setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000, dt_prfl=500,
                 dt_ld=25):

        super().__init__(RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, Ring, save_to, get_from,
                 n_bunches, dt_cont=dt_cont, dt_beam=dt_beam, dt_cl=dt_cl, dt_prfl=dt_prfl, dt_ld=dt_ld)

        # Beam
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
        self.init_beam_measurements()

        # LLRF
        if LHCCavityLoop is None:
            self.init_cl = getattr(self, 'empty_measurement')
            self.cl_infreq_meas = getattr(self, 'empty_measurement')
            self.cl_freq_meas = getattr(self, 'empty_measurement')
            self.inj_power = getattr(self, 'empty_measurement')
        else:
            self.init_cl = self.init_cavity_loop_measurements
            self.cl_infreq_meas = self.cavity_loop_infrequency_measurements
            self.cl_freq_meas = self.cavity_loop_frequent_measurements
            self.inj_power = self.injection_power_transient_measurement

        self.init_cl()

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        elif setting == 1:
            self.perform_measurements = getattr(self, 'measurement_with_injection')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')

    def injection(self, beam_ID, bucket, simulated):
        r'''Injection of beam from the SPS into the LHC.'''

        if beam_ID == "no injections":
            pass
        else:
            # import beam
            if simulated:
                injected_beam = np.load(self.get_from + 'generated_beams/' + beam_ID + 'simulated_beam.npy')
            else:
                injected_beam = np.load(self.get_from + 'generated_beams/' + beam_ID + 'generated_beam.npy')

            # place beam in the correct RF bucket
            sps_lhc_dt = (((2 * np.pi * 1100.009)/(4620 * c * self.ring.beta[0, self.tracker.counter[0]])) -
                          self.tracker.rf_params.t_rf[0, self.tracker.counter[0]])/2
            lhc_bucket_dt = bucket * self.tracker.rf_params.t_rf[0, self.tracker.counter[0]]
            phase_offset_dt = -self.tracker.rf_params.phi_rf[:, self.tracker.counter[0]] / \
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

        bucket_height = separatrix(self.ring, self.tracker.rf_params,
                                   np.array([self.tracker.rf_params.phi_s[self.tracker.counter[0]] / (2 * np.pi) *
                                   self.tracker.rf_params.t_rf[0, self.tracker.counter[0]]]))

        self.tracker.beam.losses_below_energy(-bucket_height[0])
        losses_from_cut = self.tracker.beam.n_macroparticles_lost * self.tracker.beam.ratio

        return losses_from_cut

    def init_cavity_loop_measurements(self):
        r'''Method to initiate necessary arrays for measurements of the cavity loop'''
        self.power_transient = np.zeros((500, self.cl.n_coarse))
        self.max_power = np.zeros(self.n_cont)

    def cavity_loop_frequent_measurements(self):
        r'''Frequent single-turn-single-value measurements'''
        self.max_power[self.ind_cont] = np.max(self.cl.generator_power()[-self.cl.n_coarse:])

    def cavity_loop_infrequency_measurements(self):
        r'''Infrequent single-turn-multi-value measurements'''
        # Plot
        pcs.plot_generator_power(self.cl, self.turn, self.save_to + 'figures/')
        pcs.plot_cavity_voltage(self.cl, self.turn, self.save_to + 'figures/')
        pcs.plot_max_power(self.max_power, self.time_turns, self.ind_cont - 1, self.save_to + 'figures/')

        # Save
        np.save(self.save_to + 'data/' + f'gen_power_{self.turn}.npy',
                self.cl.generator_power()[-self.cl.n_coarse:])
        np.save(self.save_to + 'data/' + f'ant_volt_{self.turn}.npy',
                self.cl.V_ANT_COARSE[-self.cl.n_coarse:])
        np.save(self.save_to + 'data/' + 'max_power.npy', self.max_power)

    def injection_power_transient_measurement(self):
        r'''Method to measure injection power transients'''
        # Gather power transients during the first 500 turns after the three injections
        self.power_transient[self.turns_after_injection, :] = self.cl.generator_power()[-self.cl.n_coarse:]

        # Save power transients after 500 turns
        if self.turns_after_injection == 499:
            np.save(self.save_to + 'data/' + f'power_transient_injection_{self.injection_number}.npy',
                    self.power_transient)
            self.power_transient = np.zeros((500, self.cl.n_coarse))

    def standard_measurement(self):
        r'''Default measurement rutine for LHC simulations.'''

        # Setting up arrays on initial track call
        if self.turn == 0:
            self.uncaptured_beam = self.measure_uncaptured_losses()

            loss_dict = {'Uncaptured losses': self.uncaptured_beam}
            ida.make_and_write_yaml('loss_summary.yaml', self.save_to, loss_dict)

        # Line density measurement
        if self.turn % self.dt_ld == 0 and self.ind_ld < self.n_ld:
            self.line_density_measurement()
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0 and self.ind_cont < self.n_cont:
            self.cl_freq_meas()
            self.beam_frequent_measurements()
            self.bunch_losses[self.ind_cont, :] = self.bunch_intensities[0, :] - \
                                                  self.bunch_intensities[self.ind_cont, :]

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.beam_infrequent_measurements()

            # Plots
            ppr.plot_total_losses(self.bunch_losses, self.time_turns,
                                  self.ind_cont - 1, self.save_to + 'figures/',
                                  caploss=self.uncaptured_beam)

            if self.turn == self.tracker.rf_params.n_turns - 1:
                loss_dict = {'Losses after ramp': self.measure_ramp_losses()}
                ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)

                loss_dict = {'End of simulation losses (separatrix)': self.measure_uncaptured_losses()}
                ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.cl_infreq_meas()

        if self.turns_after_injection >= 0 and self.turns_after_injection < 500:
            self.inj_power()
            self.turns_after_injection += 1

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
            print(f'Found {self.n_bunches} to be injected in total')

            self.uncaptured_beam = self.measure_uncaptured_losses()

            loss_dict = {f'Uncaptured losses {self.injection_number}': self.uncaptured_beam}
            ida.make_and_write_yaml('loss_summary.yaml', self.save_to, loss_dict)

        # Line density measurement
        if self.turn % self.dt_ld == 0 and self.ind_ld < self.n_ld:
            self.line_density_measurement()
            self.ind_ld += 1

        # Gather signals which are frequently sampled
        if self.turn % self.dt_cont == 0 and self.ind_cont < self.n_cont:
            self.cl_freq_meas()
            self.beam_frequent_measurements()
            self.bunch_losses[self.ind_cont, :] = self.measure_slow_losses()

            self.ind_cont += 1

        # Gather beam based measurements, save plots and save data
        if self.turn % self.dt_beam == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.beam_infrequent_measurements()
            ppr.plot_total_losses(self.bunch_losses, self.time_turns,
                                  self.ind_cont - 1, self.save_to + 'figures/',
                                  caploss=self.uncaptured_beam, beam_structure=self.beam_structure)

            if self.turn == self.tracker.rf_params.n_turns - 1:
                loss_dict = {'Losses after ramp': self.measure_ramp_losses()}
                ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)

                loss_dict = {'End of simulation losses (separatrix)': self.measure_uncaptured_losses()}
                ida.write_to_yaml('loss_summary.yaml', self.save_to, loss_dict)

        # Gather cavity based measurements, save plots and save data
        if self.turn % self.dt_cl == 0 or self.turn == self.tracker.rf_params.n_turns - 1:
            self.cl_infreq_meas()

        if self.turns_after_injection >= 0 and self.turns_after_injection < 500:
            self.inj_power()
            self.turns_after_injection += 1

        # Close all figures for this turn
        plt.clf()
        plt.cla()
        plt.close()
