r'''
Script to get beam phase from profile measurements with the high-resolution scope.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import h5py
from blond.trackers.tracker import RingAndRFTracker
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.constants import c, e, proton_mass

import blond.utils.bmath as bm

E_0 = proton_mass * c ** 2 / e
synchronous_energy = lambda p_s: np.sqrt(p_s ** 2 + E_0 ** 2)
relativistic_gamma = lambda E_s: E_s / E_0
relativistic_beta = lambda gamma: np.sqrt(1 - 1 / gamma ** 2)

from blond.beam.profile import Profile


# Beam-phase calculation from BLonD
def beam_phase(bin_centers: np.ndarray, profile: np.ndarray,
               alpha: float, omegarf: float,
               phirf: float, bin_size: float) -> float:
    scoeff = np.trapezoid(
        np.exp(alpha * (bin_centers))
        * np.sin(omegarf * bin_centers + phirf)
        * profile, dx=bin_size
    )
    ccoeff = np.trapezoid(
        np.exp(alpha * (bin_centers))
        * np.cos(omegarf * bin_centers + phirf)
        * profile, dx=bin_size
    )

    return scoeff / ccoeff


class MeasuredBeamPhase:

    def __init__(self, profiles, sampling: float = 40e9, beam_momentum: float = 450e9,
                 n_bunches: int = 1, time_shift: int = 0, jitter: np.ndarray = None,
                 alpha: float = 0, C: float = 26658.883, h: int = 35640, phi_rf: float = 0) -> None:

        # Parameters
        self.alpha = alpha
        self.C = C
        self.h = h
        self.phi_rf = phi_rf

        # Computing RF frequency
        beam_energy = synchronous_energy(beam_momentum)
        rel_gamma = relativistic_gamma(beam_energy)
        rel_beta = relativistic_beta(rel_gamma)
        f_rev = rel_beta * c / self.C

        self.omega_rf = 2 * np.pi * self.h * f_rev
        self.rf_period = 2 * np.pi / self.omega_rf

        # Fetch all relevant parameters for the analysis
        # Beam line-density
        self.profile = profiles
        # Sample rate of the scope
        self.sampling = sampling
        # The number of frames acquired
        self.n_records = self.profile.shape[1]
        # The length of the frames
        self.record_length = self.profile.shape[0] - time_shift

        self.profile = self.profile[time_shift:, :]

        # Jitter corrections
        if jitter is not None:
            self.jitter = jitter
        else:
            self.jitter = np.zeros(self.n_records)

        # Phases
        self.beam_phases = np.zeros((n_bunches, self.n_records))
        self.full_time = np.linspace(
            0,
            (self.record_length - 1) / self.sampling,
            self.record_length
        )

    def analyse_bunch(self, single_profile: np.ndarray, time_array: np.ndarray, bunch_number: int) -> None:

        for i in range(self.beam_phases.shape[1]):
            # Remove the offset
            frame = single_profile[:, i]

            # Estimate phase
            coeff = beam_phase(
                time_array + self.jitter[i],
                frame, self.alpha, self.omega_rf,
                self.phi_rf, 1 / self.sampling
            )
            self.beam_phases[bunch_number, i] = np.arctan(coeff)

    def analyse_measurement(self, no_beam_thres: float = 20) -> np.ndarray:

        bunch_ind = 0

        # Find number of rf buckets to iterate over
        n_buckets = int(self.full_time[-1] // self.rf_period)

        for i in tqdm(range(n_buckets)):

            mask_i = (self.full_time > i * self.rf_period) & (self.full_time < (i + 1) * self.rf_period)

            # Get time
            time_frame = self.full_time[mask_i]

            # Get profile measurement for bucket i
            frame_bucket = self.profile[mask_i, :]

            # Check if bucket is empty or not
            if not np.all(frame_bucket[:, 0] < no_beam_thres):
                self.analyse_bunch(frame_bucket, time_frame, bunch_ind)
                bunch_ind += 1

        return self.beam_phases * 180 / np.pi

    def _compute_phase_shift(self, sample_number):
        return sample_number / sample_number * self.omega_rf + self.phi_rf

    @staticmethod
    def moving_average(x: np.ndarray, w: int = 20):
        return np.convolve(x, np.ones(w), 'valid') / w


class PSExtractionBeamPhase(MeasuredBeamPhase):

    def __init__(self, profiles, sampling: float = 40e9, beam_momentum: float = 26e9, n_bunches: int = 1,
                 time_shift: int = 0, harmonic: int = 168, jitter=None):
        super().__init__(
            profiles, sampling, beam_momentum, n_bunches, time_shift, jitter,
            alpha=0, C=628.3185, h=harmonic, phi_rf=0
        )


class SPSBeamPhase(MeasuredBeamPhase):

    def __init__(self, profiles, sampling: float = 40e9, beam_momentum: float = 450e9, n_bunches: int = 1,
                 time_shift: int = 0, jitter=None):
        super().__init__(
            profiles, sampling, beam_momentum, n_bunches, time_shift, jitter,
            alpha=0, C=6911.560, h=4620, phi_rf=0
        )


class LHCBeamPhase(MeasuredBeamPhase):

    def __init__(self, profiles, sampling: float = 40e9, beam_momentum: float = 450e9, n_bunches: int = 1,
                 time_shift: int = 0, jitter=None):
        super().__init__(
            profiles, sampling, beam_momentum, n_bunches, time_shift, jitter,
            alpha=0, C=26658.883, h=35640, phi_rf=0
        )


class SimulatedBeamPhase:

    def __init__(self, profile: Profile, rf_tracker: RingAndRFTracker, n_bunches: int = 1, n_bunches_init: int = None):

        self.profile = profile
        self.rf_tracker = rf_tracker
        self.n_bunches = n_bunches

        if n_bunches_init is None:
            self.n_bunches_current = self.n_bunches

        self.bunch_phases = np.zeros(n_bunches)

    def set_number_of_bunches(self, new_number_of_bunches: float):
        self.n_bunches_current = new_number_of_bunches

    def extract_phases(self, no_beam_thres: float = 0.5):
        omega_rf = self.rf_tracker.rf_params.omega_rf[0, self.rf_tracker.rf_params.counter[0]]
        phi_rf = self.rf_tracker.rf_params.phi_rf[0, self.rf_tracker.rf_params.counter[0]]
        time_frame = self.profile.bin_centers
        normalized_profile = (np.copy(self.profile.n_macroparticles)
                              / np.sum(self.profile.n_macroparticles) * self.n_bunches_current)

        n_buckets = int((self.profile.bin_centers[-1] - self.profile.bin_centers[0])
                        * omega_rf / (2 * np.pi))
        rf_period = 2 * np.pi / omega_rf
        time_shift = (int(time_frame[0] * omega_rf / (2 * np.pi)) + 1)
        bunch_ind = 0

        for i in range(n_buckets):
            bucket_center = self.rf_tracker.rf_params.bucket_center(i + time_shift)
            left_side = bucket_center - rf_period/2
            right_side = bucket_center + rf_period/2
            mask_i = (time_frame > left_side) & (time_frame < right_side)

            # Get time
            time_bucket = time_frame[mask_i]

            # Get profile measurement for bucket i
            frame_bucket = normalized_profile[mask_i]

            if np.sum(frame_bucket) > no_beam_thres:
                self.bunch_phases[bunch_ind] = self.analyse_bunch(
                    frame_bucket, time_bucket, omega_rf, phi_rf,
                    self.profile.bin_size
                )
                bunch_ind += 1

        return self.bunch_phases * 180 / np.pi

    @staticmethod
    def analyse_bunch(single_profile, time_array, omega_rf, phi_rf, bin_size):

        coeff = bm.beam_phase(
            time_array, single_profile,
            0, omega_rf, phi_rf, bin_size
        )

        return np.arctan(coeff)


def main():
    fname = "power_md_2024/data/lld_scope_data/PROFILE_B1_b1781_20240820005259.h5"
    n_bunches = 96
    time_shift = 75  # 75 in Beam 1 and 0 in Beam 2

    # Import profile
    file = h5py.File(fname)

    # Loading the relevant profile data
    profiles = file['Profile/profile']
    sampling = file['Profile/samplerate'][0]
    jitter = file['Profile/trigger_offsets'][0, :]

    # Performing the phase analysis
    beam_phase_estimate = MeasuredBeamPhase(
        profiles, sampling=sampling, n_bunches=n_bunches, time_shift=time_shift, jitter=jitter
    )
    phases = beam_phase_estimate.analyse_measurement(no_beam_thres=50)

    # Filtering away high-frequency noise
    phases_filt = np.zeros((phases.shape[0], phases.shape[1] - 19))
    for i in range(n_bunches):
        phases_filt[i, :] = beam_phase_estimate.moving_average(phases[i, :])

    # Plotting the data
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(phases[0, :])
    ax.plot(phases_filt[0, :])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(phases[:, 7])
    ax.plot(phases_filt[:, 7])

    plt.show()


if __name__ == "__main__":
    main()
