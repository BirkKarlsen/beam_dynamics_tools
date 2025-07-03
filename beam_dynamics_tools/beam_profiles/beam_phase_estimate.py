r'''
Script to get beam phase from profile measurements with the high-resolution scope.

Author: Birk Emil Karlsen-Bæck
'''

from typing import TYPE_CHECKING
import numpy as np
import h5py
from blond.trackers.tracker import RingAndRFTracker
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.constants import c, e, proton_mass
from scipy.signals import find_peaks

import blond.utils.bmath as bm

E_0 = proton_mass * c ** 2 / e
synchronous_energy = lambda p_s: np.sqrt(p_s ** 2 + E_0 ** 2)
relativistic_gamma = lambda E_s: E_s / E_0
relativistic_beta = lambda gamma: np.sqrt(1 - 1 / gamma ** 2)

from blond.beam.profile import Profile

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Beam-phase calculation from BLonD
def beam_phase(
        bin_centers: NDArray, profile: NDArray,
        alpha: float, omegarf: float,
        phirf: float, bin_size: float
    ) -> float:
    """Function to estimate the beam phase for a given rf frequency based on a beam profile.

    Args:
        bin_centers (NDArray):
            The time series associated with the profile.
        profiles (NDArray):
            The profile measurement.
        alpha (float):
            The filter coefficient for the phase estimate.
        omegarf (float):
            The angular rf frequency at which to calculate the bunch phase.
        phirf (float):
            The instantaneous rf phase.
        bin_size (float):
            The time interval between each profile sample.
    """
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


def moving_average(x: NDArray, w: int = 20):
    """Convolution-based moving average.

    Args:
        x (NDArray):
            Signal to perform moving average on.
        w (int):
            Number of samples to average over.

    Returns:
        result (NDArray):
            Signal with moving average.
    """
    return np.convolve(x, np.ones(w), 'valid') / w


def detect_zero_phase(profile: NDArray, n_samples_per_bucket: int, amplitude_tol: float = 0.5):
    """Algorithm to auto-detect the zero phase of the profile measurement.

    Args:
        profile (NDArray):
            1D array with reference profile measurement.
        n_samples_per_bucket (int):
            Number of profile samples per bucket.
        amplitude_tol (float):
            The amplitude tolerance to discriminate bunches from noise.

    Returns:
        zero_phase (int):
            The index of the zero-phase of the profile measurement.
    """

    # Maximum of profile to get the
    height_threshold = np.max(profile) * amplitude_tol

    # Find first bunch using find_peaks
    peaks, _ = find_peaks(profile, height=height_threshold, distance=n_samples_per_bucket)

    return peaks[0] - round(n_samples_per_bucket / 2)


class MeasuredBeamPhase:
    """Class to estimate the beam phase for a given harmonic based on beam profiles.

    Args:
        profiles (NDArray):
            The profile to estimate the beam phase from. The first axis is along a single turn while
            the second axis turn-by-turn.
        sampling (float):
            The sampling [# samples / s] of the measurement.
        beam_momentum (float):
            The beam momentum [eV / c] the profile was measured at.
        n_bunches (int):
            The number of expected bunches.
        sample_correction (int):
            The shift along turn [# samples] to find the zero phase of the rf wave.
        jitter (NDArray):
            A 1D array of jitter correction [s] from the scope.
        alpha (float):
            The filter coefficient to estimate the beam phase.
        C (float):
            The circumference [m] of the accelerator.
        h (int):
            The harmonic to measure the beam phase at.
        phi_rf (float):
            The phase [rad] of the rf system relative to the start of the turn.
        auto_detect_rf_phase (bool):
            Flag to enable algorthm to automatically detect the correct zero phase of the rf system
            relative to the profile measurement. The ``sample_correction`` will be a correction
            on top of what the algorithm finds.

    Attributes:
        alpha (float):
            The filter coefficient to estimate the beam phase.
        C (float):
            The circumference [m] of the accelerator.
        h (int):
            The harmonic to measure the beam phase at.
        phi_rf (float):
            The phase [rad] of the rf system relative to the start of the turn.
        omega_rf (float):
            The angular rf frequency [rad / s] corresponding to the given harmonic.
        rf_period (float):
            The period of the rf system [s].
        profile (NDArray):
            The profile to estimate the beam phase from. The first axis is along a single turn while
            the second axis turn-by-turn.
        sampling (float):
            The sampling [# samples / s] of the measurement.
        n_records (int):
            The number of records, i.e. frames, to analyse.
        record_length (int):
            The length of each frame [# samples].
        jitter (NDArray):
            A 1D array of jitter correction [s] from the scope.
        beam_phases (NDArray):
            The calculated beam phases [deg.].
        full_time (NDArray):
            The time array along a single turn [s].

    """

    def __init__(
            self,
            profiles: NDArray,
            sampling: float = 40e9,
            beam_momentum: float = 450e9,
            n_bunches: int = 1,
            sample_correction: int = 0,
            jitter: NDArray = None,
            alpha: float = 0,
            C: float = 26658.883,
            h: int = 35640,
            phi_rf: float = 0,
            auto_detect_rf_phase: bool = False
        ):

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
        self.rf_period_samples = round(self.rf_period * sampling)

        if auto_detect_rf_phase:
            detected_zero_phase = self.detect_zero_phase(
                profiles[:, 0],
                self.rf_period_samples,
            )
        else:
            detected_zero_phase = 0

        sample_correction += detected_zero_phase

        # Fetch all relevant parameters for the analysis
        # Beam line-density
        self.profile = profiles
        # Sample rate of the scope
        self.sampling = sampling
        # The number of frames acquired
        self.n_records = self.profile.shape[1]
        # The length of the frames
        self.record_length = self.profile.shape[0] - sample_correction

        self.profile = self.profile[sample_correction:, :]

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

        for i in tqdm(range(n_buckets), desc="Analyzing buckets", unit="buckets", position=1, leave=False):

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
        return moving_average(x=x, w=w)

    @staticmethod
    def detect_zero_phase(profile: NDArray, n_samples_per_bucket: int, amplitude_tol: float = 0.5):
        return detect_zero_phase(
            profile=profile,
            n_samples_per_bucket=n_samples_per_bucket,
            amplitude_tol=amplitude_tol
        )


class PSExtractionBeamPhase(MeasuredBeamPhase):

    def __init__(
            self,
            profiles: NDArray,
            sampling: float = 40e9,
            beam_momentum: float = 26e9,
            n_bunches: int = 1,
            time_shift: int = 0,
            harmonic: int = 168,
            jitter: NDArray = None,
            auto_detect_rf_phase: bool = False
        ):
        super().__init__(
            profiles, sampling, beam_momentum, n_bunches, time_shift, jitter,
            alpha=0, C=628.3185, h=harmonic, phi_rf=0,
            auto_detect_rf_phase=auto_detect_rf_phase
        )


class SPSBeamPhase(MeasuredBeamPhase):

    def __init__(
            self,
            profiles: NDArray,
            sampling: float = 40e9,
            beam_momentum: float = 450e9,
            n_bunches: int = 1,
            time_shift: int = 0,
            jitter: NDArray = None,
            auto_detect_rf_phase: bool = False
        ):
        super().__init__(
            profiles, sampling, beam_momentum, n_bunches, time_shift, jitter,
            alpha=0, C=6911.560, h=4620, phi_rf=0,
            auto_detect_rf_phase=auto_detect_rf_phase
        )


class LHCBeamPhase(MeasuredBeamPhase):

    def __init__(
            self,
            profiles: NDArray,
            sampling: float = 40e9,
            beam_momentum: float = 450e9,
            n_bunches: int = 1,
            time_shift: int = 0, 
            jitter: NDArray = None,
            auto_detect_rf_phase: bool = False
        ):
        super().__init__(
            profiles, sampling, beam_momentum, n_bunches, time_shift, jitter,
            alpha=0, C=26658.883, h=35640, phi_rf=0,
            auto_detect_rf_phase=auto_detect_rf_phase
        )


class SimulatedBeamPhase:
    """Class to analyze the beam phase turn by turn in BLonD simulations.

    Args:
        profile (Profile):
            BLonD profile object with the beam information.
        rf_tracker (RingAndRFTracker):
            BLonD ring and rf tracker with the RF phase and frequency information.
        n_bunches (int):
            The total number of bunches in the ring and which are covered by the profile object.
            Default value is 1 bunch.
        n_bunches_init (int):
            In the case of a changing number of bunches during the simulation due to injections,
            you can specify the initial number of bunches in your simulation. The variable is None by the default
            and will therefore be the same value as n_bunches.
    """

    def __init__(self, profile: Profile, rf_tracker: RingAndRFTracker,
                 n_bunches: int = 1, n_bunches_init: int = None) -> None:

        self.profile = profile
        self.rf_tracker = rf_tracker
        self.n_bunches = n_bunches

        if n_bunches_init is None:
            self.n_bunches_current = self.n_bunches

        self.bunch_phases = np.zeros(n_bunches)

    def set_number_of_bunches(self, new_number_of_bunches: int) -> None:
        """Method to update the number of bunches in the simuation.

        Args:
            new_number_of_bunches (int):
                The new number of bunches in the ring.
        """
        self.n_bunches_current = new_number_of_bunches

    def extract_phases(self, no_beam_thres: float = 0.5) -> np.ndarray[float]:
        """Method to extract the beam phases from the BLonD profile object associated with the instance of
        this class.

        Args:
            no_beam_thres (float):
                The threshold for which to consider an rf bucket empty. The threshold is a fraction of the average
                bunch intensity in the simulation.

        Returns:
            bunch_phases
        """

        # Loading all the parameters from the BLonD simulation
        omega_rf = self.rf_tracker.rf_params.omega_rf[0, self.rf_tracker.rf_params.counter[0]]
        phi_rf = self.rf_tracker.rf_params.phi_rf[0, self.rf_tracker.rf_params.counter[0]]
        time_frame = self.profile.bin_centers

        # Normalize the beam line density to unity intensity
        normalized_profile = (np.copy(self.profile.n_macroparticles)
                              / np.sum(self.profile.n_macroparticles) * self.n_bunches_current)

        # Compute the total number of buckets to iterate over
        n_buckets = int((self.profile.bin_centers[-1] - self.profile.bin_centers[0])
                        * omega_rf / (2 * np.pi))

        # Compute the rf period and time shifts
        rf_period = 2 * np.pi / omega_rf
        time_shift = (int(time_frame[0] * omega_rf / (2 * np.pi)) + 1)
        bunch_ind = 0

        # Pre-compute bucket centers
        bucket_centers = self.rf_tracker.rf_params.bucket_center(np.arange(n_buckets) + time_shift)

            # Create masks for all buckets
        left_sides = bucket_centers - rf_period / 2
        right_sides = bucket_centers + rf_period / 2
        masks = [(time_frame > left) & (time_frame < right) for left, right in zip(left_sides, right_sides)]
        
        for mask in masks:
            # Get time coordinates of bucket
            time_bucket = time_frame[mask]

            # Get profile measurement for bucket i
            frame_bucket = normalized_profile[mask]

            # Check if there is beam in the bucket
            if np.sum(frame_bucket) > no_beam_thres:
                self.bunch_phases[bunch_ind] = self.analyse_bunch(
                    frame_bucket, time_bucket, omega_rf, phi_rf,
                    self.profile.bin_size
                )
                bunch_ind += 1

        return self.bunch_phases * 180 / np.pi

    @staticmethod
    def analyse_bunch(single_profile: np.ndarray[float], time_array: np.ndarray[float], omega_rf: float,
                      phi_rf: float, bin_size: float) -> float:
        """Method to compute the beam phase.

        Args:
            single_profile (NDArray[float]):
                Array for the line density coordinate.
            time_array (NDArray[float]):
                Array for the time coordinate for the line density.
            omega_rf (float):
                The RF angular frequency in radians per second.
            phi_rf (float):
                The RF phase in radians.
            bin_size (float):
                The size of the line density bins in seconds.

        Returns:
            phase (float):
                The beam phase in radians.
        """

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
