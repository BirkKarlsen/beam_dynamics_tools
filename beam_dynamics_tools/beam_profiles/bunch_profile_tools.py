'''
Functions to analyse bunch profile measurements.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import h5py
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.signal import find_peaks
from scipy.stats import linregress
import os

from blond_common.fitting.profile import binomial_amplitudeN_fit, FitOptions
from blond_common.interfaces.beam.analytic_distribution import binomialAmplitudeN

from beam_dynamics_tools.beam_profiles.cable_transfer_function import apply_lhc_cable_tf
from beam_dynamics_tools.signal_analysis.measured_signals import fit_sin


def getBeamPattern(timeScale, frames, heightFactor=0.015, distance=500, N_bunch_max=3564,
                     fit_option='fwhm', plot_fit=False, baseline_length=1, BASE=False,
                     wind_len=10, beam=1, apply_tf=False):
    dt = timeScale[1] - timeScale[0]
    fit_window = int(round(wind_len * 1e-9 / dt / 2))
    N_frames = frames.shape[1]
    N_bunches = np.zeros((N_frames,), dtype=int)
    Bunch_positions = np.zeros((N_frames, N_bunch_max))
    Bunch_lengths = np.zeros((N_frames, N_bunch_max))
    Bunch_peaks = np.zeros((N_frames, N_bunch_max))
    Bunch_intensities = np.zeros((N_frames, N_bunch_max))
    Bunch_positionsFit = np.zeros((N_frames, N_bunch_max))
    Bunch_peaksFit = np.zeros((N_frames, N_bunch_max))
    Bunch_Exponent = np.zeros((N_frames, N_bunch_max))
    Goodness_of_fit = np.zeros((N_frames, N_bunch_max))

    for i in np.arange(N_frames):
        frame = frames[:, i]

        pos, _ = find_peaks(frame, height=heightFactor, distance=distance)
        N_bunches[i] = len(pos)
        Bunch_positions[i, 0:N_bunches[i]] = timeScale[pos]
        Bunch_peaks[i, 0:N_bunches[i]] = frame[pos]

        for j, v in enumerate(pos):
            x = 1e9 * timeScale[v - fit_window:v + fit_window]
            y = frame[v - fit_window:v + fit_window]
            if BASE:
                baseline = np.mean(y[:baseline_length])
                y = y - baseline

            if apply_tf:
                y, x = apply_lhc_cable_tf(y, x * 1e-9, beam)
                x *= 1e9

            if fit_option == 'fwhm':
                (mu, sigma, amp) = fwhm(x, y, level=0.5)
            else:
                (amp, mu, sigma, exponent) = binomial_amplitudeN_fit(x, y,
                                                                     fitOpt=FitOptions(fittingRoutine='minimize'),
                                                                     plotOpt=None)
                y_fit = binomialAmplitudeN(x, *[amp, mu, sigma, exponent])


                if plot_fit:
                    if i % 1000 == 0:
                        print(f'Profile {i}')
                        print(amp, mu, sigma, exponent)

                sigma /= 4


            Bunch_lengths[i, j] = 4 * sigma
            Bunch_intensities[i, j] = np.sum(y)
            Bunch_positionsFit[i, j] = mu
            Bunch_peaksFit[i, j] = amp
            if fit_option != 'fwhm':
                Bunch_Exponent[i, j] = exponent
                Goodness_of_fit[i, j] = np.mean(np.abs(y - y_fit)/np.max(y)) * 100

    N_bunches_max = np.max(N_bunches)
    Bunch_positions = Bunch_positions[:, 0:N_bunches_max]
    Bunch_peaks = Bunch_peaks[:, 0:N_bunches_max]
    Bunch_lengths = Bunch_lengths[:, 0:N_bunches_max]
    Bunch_intensities = Bunch_intensities[:, 0:N_bunches_max]
    Bunch_positionsFit = Bunch_positionsFit[:, 0:N_bunches_max]
    Bunch_peaksFit = Bunch_peaksFit[:, 0:N_bunches_max]
    Bunch_Exponent = Bunch_Exponent[:, 0:N_bunches_max]
    Goodness_of_fit = Goodness_of_fit[:, 0:N_bunches_max]

    return N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
           Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit

def get_beam_pattern(profiles, t, height_factor=0.015, distance=500, n_bunch_max=3564,
                        wind_len=2.5e-9, single_turn=False):

    if single_turn:
        profiles = np.array([profiles]).T

    dt = t[1] - t[0]

    fit_window = int(round(wind_len / dt / 2))
    n_frames = profiles.shape[0]

    n_bunches = np.zeros(n_frames, dtype=int)
    bunch_positions = np.zeros((n_frames, n_bunch_max))
    bunch_lengths = np.zeros((n_frames, n_bunch_max))
    bunch_peaks = np.zeros((n_frames, n_bunch_max))
    bunch_peak_position = np.zeros((n_frames, n_bunch_max))

    for i in np.arange(n_frames):
        frame = profiles[i, :]

        pos, _ = find_peaks(frame, height=height_factor, distance=distance)
        n_bunches[i] = len(pos)

        for j, v in enumerate(pos):
            x = t[v - fit_window: v + fit_window]
            y = frame[v - fit_window: v + fit_window]

            (mu, sigma, amp) = fwhm(x, y, level=0.5)

            bunch_lengths[i, j] = 4 * sigma
            bunch_positions[i, j] = mu
            bunch_peaks[i, j] = amp
            bunch_peak_position[i, j] = peak_position(x, y, level=0.5)

    n_bunch_max = np.max(n_bunches)
    bunch_peaks = bunch_peaks[:, :n_bunch_max]
    bunch_lengths = bunch_lengths[:, :n_bunch_max]
    bunch_positions = bunch_positions[:, :n_bunch_max]
    bunch_peak_position = bunch_peak_position[:, :n_bunch_max]

    if single_turn:
        return bunch_positions[0, :], bunch_lengths[0, :], bunch_peaks[0, :], bunch_peak_position[0, :]

    return bunch_positions, bunch_lengths, bunch_peaks, bunch_peak_position


def fwhm(x, y, level=0.5):
    offset_level = np.mean(y[0:5])
    amp = np.max(y) - offset_level
    t1, t2 = interp_f(x, y, level)
    mu = (t1 + t2) / 2.0
    sigma = (t2 - t1) / 2.35482
    popt = (mu, sigma, amp)

    return popt


def interp_f(time, bunch, level):
    bunch_th = level * bunch.max()
    time_bet_points = time[1] - time[0]
    taux = np.where(bunch >= bunch_th)
    taux1, taux2 = taux[0][0], taux[0][-1]
    t1 = time[taux1] - (bunch[taux1] - bunch_th) / (bunch[taux1] - bunch[taux1 - 1]) * time_bet_points
    t2 = time[taux2] + (bunch[taux2] - bunch_th) / (bunch[taux2] - bunch[taux2 + 1]) * time_bet_points

    return t1, t2

def peak_position(x, y, level=0.5):
    r'''Find position of bunch peak from interpolation.'''
    y_lvl = level * np.max(y)
    inds = np.where(y >= y_lvl)

    y_fit = InterpolatedUnivariateSpline(x[inds], y[inds], k=4)
    roots = y_fit.derivative().roots()
    root_val = y_fit(roots)
    max_ind = np.argmax(root_val)

    return roots[max_ind]


def extract_bunch_position(time, profile, heighFactor=0.015, wind_len=10):
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(time, np.array([profile]).T,
                                                                     heightFactor=heighFactor, wind_len=wind_len)
    return Bunch_positionsFit[0, :], Bunch_lengths[0, :]


def extract_bunch_parameters(time, profile, heighFactor=0.015, wind_len=10, distance=500, n_bunches=None):
    r'''
    Extraces the bunch positions, bunch lengths and bunch intensities from a given profile with a given time array.
    If the expected number of bunches is passed then if the number of bunches found is less then the expected the
    function will return zero-valued elements at the end. If the number found is more than the expected the
    function will return the bunches with the highest intensities.

    :param time: time array for each measurement point
    :param profile: beam line density
    :param heighFactor: height factor to find bunch peaks
    :param wind_len: length of bunch window
    :param distance: minimum length between bunch peaks
    :return: bunch positions, bunch lenghts, bunch intensities
    '''
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(time, np.array([profile]).T,
                                                                     heightFactor=heighFactor, wind_len=wind_len,
                                                                     distance=distance)
    if n_bunches is None:
        return Bunch_positionsFit[0, :], Bunch_lengths[0, :], Bunch_intensities[0, :]
    else:
        if N_bunches[0] == n_bunches:
            return Bunch_positionsFit[0, :], Bunch_lengths[0, :], Bunch_intensities[0, :]
        elif N_bunches[0] < n_bunches:
            db = n_bunches - N_bunches[0]
            empty_bunches = np.zeros(db)

            Bunch_positionsFit = np.concatenate((Bunch_positionsFit[0, :], empty_bunches))
            Bunch_lengths = np.concatenate((Bunch_lengths[0, :], empty_bunches))
            Bunch_intensities = np.concatenate((Bunch_intensities[0, :], empty_bunches))

            return Bunch_positionsFit, Bunch_lengths, Bunch_intensities
        else:
            max_ind = np.argpartition(Bunch_intensities[0, :], -n_bunches)[-n_bunches:]
            max_ind = np.sort(max_ind)

            return Bunch_positionsFit[0, max_ind], Bunch_lengths[0, max_ind], Bunch_intensities[0, max_ind]


def find_offset(pos):
    r'''
    Takes in an array of bunch positions and does a linear regression of them.
    The bunch-by-bunch offset is then calculated by taking the difference between the two.
    :param pos: numpy-array - Ordered list of bunch positions in time
    :return: numpy-array of the bunch-by-bunch offset
    '''
    x = np.linspace(0, len(pos), len(pos))

    sl, inter, pval, rval, err = linregress(x, pos)
    fit_line = sl * x + inter

    offset_fit = pos - fit_line
    return offset_fit


def bunch_by_bunch_spacing(positions, batch_len):
    n_batch = len(positions) // batch_len
    positions = positions.reshape((n_batch, batch_len))
    spacings = np.zeros(positions.shape)

    for i in range(spacings.shape[0]):
        spacings[i, :] = find_offset(positions[i, :])

    return spacings


def find_batch_length(positions, bunch_spacing):
    r'''
    Finds the number of bunches in each batch and the number of batches.
    The function assumes that all batches have the same length.

    :param positions: The positions of all the bunches in the beam [ns]
    :param bunch_spacing: The bunch spacing within a batch [ns]
    :return: the number of bunches in each batch, number of batches
    '''
    n_batches = 1
    batch_len = len(positions)
    prev_pos = positions[0]

    for i, pos in enumerate(positions):
        if pos - prev_pos > 6 * bunch_spacing:
            if n_batches == 1:
                batch_len = i

            n_batches += 1

        prev_pos = pos

    return batch_len, n_batches

def bunch_position_from_COM(time, profile):
    M = np.trapz(profile, time)
    return np.trapz(profile * time, time) / M


def reshape_profile_data(data, t, T):
    N_turns = t[-1] // T
    n_samples = T // (t[1] - t[0])

    rdata = np.zeros((N_turns, n_samples))

    for i in range(N_turns):
        data_i = data[n_samples * i: n_samples * (i + 1)]

        rdata[i, :] = data_i

    return rdata


def get_profile_data(f, fdir):
    data = h5py.File(fdir + f, 'r')
    profile = data['Profile']['profile'][:]
    t = np.linspace(0, (data['Profile']['profile'][:].shape[0] - 1) / data['Profile']['samplerate'][0],
                    data['Profile']['profile'][:].shape[0])

    return profile, t


def find_synchrotron_frequency_from_profile(profile, t, T_rev, turn_constant, init_osc_length, final_osc_start):
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(t, profile, heightFactor=30,
                                                                         wind_len=4)

    bpos = Bunch_positionsFit[:, 0]
    t = np.linspace(0, (len(bpos) - 1), len(bpos)) * T_rev * turn_constant

    fit_dict_init = fit_sin(t[:init_osc_length], bpos[:init_osc_length])
    fit_dict_final = fit_sin(t[final_osc_start:], bpos[final_osc_start:])

    return fit_dict_init, fit_dict_final


def get_sorted_files(V, QL, cavity, beam, emittance, add=''):
    V = str(V).replace('.', '')
    return f'profile_V{V}MV_QL{QL}k_C{cavity}B{beam}{add}_{emittance}_emittance.h5'


def analyse_synchrotron_frequency_cavity_by_cavity(V, QL, cavitiesB1, cavitiesB2, emittance, fdir, T_rev, turn_constant,
                                                   init_osc_length, final_osc_start, add=''):
    B1_files = []
    B2_files = []
    for i in range(len(cavitiesB1)):
        B1_files.append(get_sorted_files(V=V, QL=QL, cavity=cavitiesB1[i],
                                             beam=1, emittance=emittance, add=add))
        B2_files.append(get_sorted_files(V=V, QL=QL, cavity=cavitiesB2[i],
                                             beam=2, emittance=emittance, add=add))

    freqs_init = np.zeros((len(cavitiesB1), 2))
    freqs_final = np.zeros((len(cavitiesB1), 2))

    for i in range(len(cavitiesB1)):
        # Beam 1
        B1_profiles, t = get_profile_data(B1_files[i], fdir)

        fit_dict_init, fit_dict_final = find_synchrotron_frequency_from_profile(B1_profiles, t, T_rev,
                                                                                turn_constant,
                                                                                init_osc_length, final_osc_start)
        freqs_init[i, 0] = fit_dict_init['freq']
        freqs_final[i, 0] = fit_dict_final['freq']

        # Beam 2
        B2_profiles, t = get_profile_data(B2_files[i], fdir)

        fit_dict_init, fit_dict_final = find_synchrotron_frequency_from_profile(B2_profiles, t, T_rev,
                                                                                turn_constant,
                                                                                init_osc_length, final_osc_start)
        freqs_init[i, 1] = fit_dict_init['freq']
        freqs_final[i, 1] = fit_dict_final['freq']

    return freqs_init, freqs_final


def analyse_synchrotron_frequency_with_all_cavities(V, QL, emittance, fdir, T_rev, turn_constant,
                                                    init_osc_length, final_osc_start, add=''):

    B1_file = get_sorted_files(V=V, QL=QL, cavity='all', beam=1, emittance=emittance, add=add)
    B2_file = get_sorted_files(V=V, QL=QL, cavity='all', beam=2, emittance=emittance, add=add)

    freqs_init = np.zeros(2)
    freqs_final = np.zeros(2)


    # Beam 1
    B1_profiles, t = get_profile_data(B1_file, fdir)

    fit_dict_init, fit_dict_final = find_synchrotron_frequency_from_profile(B1_profiles, t, T_rev,
                                                                            turn_constant,
                                                                            init_osc_length, final_osc_start)
    freqs_init[0] = fit_dict_init['freq']
    freqs_final[0] = fit_dict_final['freq']

    # Beam 2
    B2_profiles, t = get_profile_data(B2_file, fdir)

    fit_dict_init, fit_dict_final = find_synchrotron_frequency_from_profile(B2_profiles, t, T_rev,
                                                                            turn_constant,
                                                                            init_osc_length, final_osc_start)
    freqs_init[1] = fit_dict_init['freq']
    freqs_final[1] = fit_dict_final['freq']

    return freqs_init, freqs_final


def analyze_profile(profile, t, T_rev, turn_constant, init_osc_length, final_osc_start, mode='fwhm', wind_len=4,
                    beam=1, apply_tf=False):

    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(t, profile, heightFactor=30,
                                                                     wind_len=wind_len, fit_option=mode,
                                                                     plot_fit=True, beam=beam, apply_tf=apply_tf)

    bpos = Bunch_positionsFit[:, 0]
    t = np.linspace(0, len(bpos) - 1, len(bpos)) * T_rev * turn_constant

    fit_dict_init = fit_sin(t[:init_osc_length], bpos[:init_osc_length])
    fit_dict_final = fit_sin(t[final_osc_start:], bpos[final_osc_start:])

    return fit_dict_init, fit_dict_final, bpos, Bunch_lengths[:, 0], t


def analyze_profile_htf(profile, t, T_rev, T_s, N_T_s, turn_constant, mode='fwhm', wind_len=4, beam=1):
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(t, profile, heightFactor=30,
                                                                     wind_len=wind_len, fit_option=mode,
                                                                     plot_fit=True, beam=beam)

    bpos = Bunch_positionsFit[:, 0]
    t = np.linspace(0, len(bpos) - 1, len(bpos)) * T_rev * turn_constant
    N_turns = N_T_s * int(round(T_s / (T_rev * turn_constant)))
    bpos_extracted = bpos[:N_turns] - np.mean(bpos[:N_turns])
    error = np.max(np.abs(bpos_extracted))
    fit_dict = fit_sin(t[:N_turns], bpos[:N_turns] - np.mean(bpos[:N_turns]))

    return fit_dict, bpos, Bunch_lengths[:, 0], t, error


def analyze_profiles_cavity_by_cavity(V, QL, cavitiesB1, cavitiesB2, emittance, fdir, T_rev, turn_constant,
                                      init_osc_length, final_osc_start, add='', plt_cav1=None, plt_cav2=None,
                                      fbl_mean=1000, apply_tf=False, print_progress=False):
    B1_files = []
    B2_files = []
    for i in range(len(cavitiesB1)):
        B1_files.append(get_sorted_files(V=V, QL=QL, cavity=cavitiesB1[i],
                                         beam=1, emittance=emittance, add=add))
        B2_files.append(get_sorted_files(V=V, QL=QL, cavity=cavitiesB2[i],
                                         beam=2, emittance=emittance, add=add))

    freqs_init = np.zeros((len(cavitiesB1), 2))
    freqs_final = np.zeros((len(cavitiesB1), 2))
    init_bl = np.zeros((len(cavitiesB1), 2))
    final_bl = np.zeros((len(cavitiesB1), 2))

    for i in range(len(cavitiesB1)):
        if print_progress:
            print(f'Analyzing Cavity {cavitiesB1[i]}B1')
        # Beam 1
        B1_profiles, t = get_profile_data(B1_files[i], fdir)

        fit_dict_init, fit_dict_final, bpos_i, blen_i, ti = analyze_profile(B1_profiles, t, T_rev, turn_constant,
                                                                            init_osc_length, final_osc_start, beam=1,
                                                                            apply_tf=apply_tf)
        freqs_init[i, 0] = fit_dict_init['freq']
        freqs_final[i, 0] = fit_dict_final['freq']
        init_bl[i, 0] = blen_i[0]
        final_bl[i, 0] = np.mean(blen_i[-fbl_mean:])

        # Beam 2
        if print_progress:
            print(f'Analyzing Cavity {cavitiesB1[i]}B2')
        B2_profiles, t = get_profile_data(B2_files[i], fdir)

        fit_dict_init, fit_dict_final, bpos_i, blen_i, ti = analyze_profile(B2_profiles, t, T_rev, turn_constant,
                                                                            init_osc_length, final_osc_start, beam=2,
                                                                            apply_tf=apply_tf)

        freqs_init[i, 1] = fit_dict_init['freq']
        freqs_final[i, 1] = fit_dict_final['freq']
        init_bl[i, 1] = blen_i[0]
        final_bl[i, 1] = np.mean(blen_i[-fbl_mean:])

    return freqs_init, freqs_final, init_bl, final_bl


def analyse_profiles_all_cavities(V, QL, emittance, fdir, T_rev, turn_constant, init_osc_length,
                                  final_osc_start, add='', plt1=False, plt2=False, fbl_mean=1000):

    B1_file = get_sorted_files(V=V, QL=QL, cavity='all', beam=1, emittance=emittance, add=add)
    B2_file = get_sorted_files(V=V, QL=QL, cavity='all', beam=2, emittance=emittance, add=add)

    freqs_init = np.zeros(2)
    freqs_final = np.zeros(2)
    init_bl = np.zeros(2)
    final_bl = np.zeros(2)

    # Beam 1
    B1_profiles, t = get_profile_data(B1_file, fdir)

    fit_dict_init, fit_dict_final, bpos_i, blen_i, ti = analyze_profile(B1_profiles, t, T_rev, turn_constant,
                                                                        init_osc_length, final_osc_start)

    freqs_init[0] = fit_dict_init['freq']
    freqs_final[0] = fit_dict_final['freq']
    init_bl[0] = np.max(blen_i[:1000])
    final_bl[0] = np.mean(blen_i[-fbl_mean:])

    # Beam 2
    B2_profiles, t = get_profile_data(B2_file, fdir)

    fit_dict_init, fit_dict_final, bpos_i, blen_i, ti = analyze_profile(B2_profiles, t, T_rev, turn_constant,
                                                                        init_osc_length, final_osc_start)

    freqs_init[1] = fit_dict_init['freq']
    freqs_final[1] = fit_dict_final['freq']
    init_bl[1] = np.max(blen_i[:1000])
    final_bl[1] = np.mean(blen_i[-fbl_mean:])

    return freqs_init, freqs_final, init_bl, final_bl


def get_first_profiles(fdir, rev_str, profile_length):
    '''
    File to get first profiles from a folder fdir and containing rev_string in the filename.

    :param fdir:
    :param rev_str:
    :param profile_length:
    :return:
    '''
    file_names = []
    for file in os.listdir(fdir):
        if rev_str in file:
            file_names.append(file)

    data = np.zeros((profile_length, len(file_names)))
    ts = np.zeros((profile_length, len(file_names)))

    for i in range(len(file_names)):
        data_i, ti = get_profile_data(file_names[i], fdir)

        data[:, i] = data_i[:, 0]
        ts[:, i] = ti

    return data, ts, file_names


def find_bunch_length(fdir, emittance, n_samples=250):
    r'''
    File to get the average and standard deviation of the first turn bunch in the LHC with a small or
    nominal emittance.

    :param fdir:
    :param emittance:
    :param n_samples:
    :return:
    '''
    profiles, ts, ids = get_first_profiles(fdir, emittance, n_samples)

    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(ts[:, 0], profiles, heightFactor=30,
                                                                         wind_len=5, fit_option='fwhm')

    return np.mean(Bunch_lengths), np.std(Bunch_lengths)


def save_profile(Profile, turn, save_to):
    profile_data = np.zeros((2, len(Profile.n_macroparticles)))
    profile_data[0, :] = Profile.n_macroparticles
    profile_data[1, :] = Profile.bin_centers

    np.save(save_to + f'profile_data_{turn}', profile_data)


def renormalize_profiles(profiles, ts, N=1):
    renorm_profiles = np.zeros(profiles.shape)

    if profiles.ndim > 1:
        for i in range(profiles.shape[1]):
            N_i = np.trapz(profiles[:, i], ts[:, i])

            renorm_profiles[:, i] = (N/N_i) * profiles[:, i]
    else:
        N_i = np.trapz(profiles, ts)

        renorm_profiles = (N / N_i) * profiles

    return renorm_profiles


def center_profiles(profiles, ts, pos=2.5e-9/2):

    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(ts[:, 0], profiles, heightFactor=30,
                                                                     wind_len=5, fit_option='fwhm')

    for i in range(profiles.shape[1]):
        profile_i = profiles[:, i]
        t_i = ts[:, i]
        bs = Bunch_positionsFit[i][0] * 1e-9
        f = interp1d(t_i - bs + pos, profile_i, fill_value=0, bounds_error=False)
        profiles[:, i] = f(t_i)

    return profiles


def retrieve_profile_measurements_based_on_file_names(fns, fdir, profile_length=250):
    profiles = np.zeros((profile_length, len(fns)))
    ts = np.zeros((profile_length, len(fns)))
    ids = []

    for i in range(len(fns)):
        profile_i, ts_i, ids_i = get_first_profiles(fdir, fns[i], profile_length)
        if len(profile_i[0, :]) == 1 and fns[i] in ids_i[0]:
            profiles[:, i] = profile_i[:, 0]
            ts[:, i] = ts_i[:, 0]
            ids.append(ids_i[0])
        else:
            print(f'Error - something went wrong when retrieving {fns[i]}')

    return profiles, ts, ids


def get_sim_name(emittance, intensity, voltage, injection_error, turns):
    r'''
    File to get the right simulation name format for a given setting.

    :param emittance: Either 'small' or 'nominal'
    :param intensity: intensity in units of 10^8
    :param voltage: RF voltage in units of kV
    :param injection_error: Injection energy error in units of MeV
    :param turns: Number of turns that was simulated
    :return: Simulation file name
    '''
    return f'{emittance}_emittance_int{intensity:.0f}e8_v{voltage:.0f}kV_dE{injection_error:.0f}MeV_{turns:.0f}turns'


def get_sim_fit(emittance, intensity, V, dE, turns, ddir, mode='fwhm'):
    sim_str_i = get_sim_name(emittance, intensity, V, dE, turns)

    if mode == 'fwhm':
        bp_str = 'bunch_position_' + sim_str_i + '.npy'
    else:
        bp_str = 'bunch_position_com_' + sim_str_i + '.npy'

    time_str = 'time_since_injection_' + sim_str_i + '.npy'
    bp = np.load(ddir + bp_str)
    time = np.load(ddir + time_str)
    time = np.concatenate((np.array([0]), time[time != 0]))
    return fit_sin(time, bp[bp != 0])

def get_sim_init_and_final_bunch_lengths(emittance, intensity, V, dE, turns, ddir, fin_points=1000):
    sim_str_i = get_sim_name(emittance, intensity, V, dE, turns)

    bp_str = 'bunch_length_' + sim_str_i + '.npy'

    bp = np.load(ddir + bp_str)
    bp = bp[bp != 0]

    return bp[0], np.mean(bp[-fin_points:])


def find_beamline_from_shot(file):
    if 'short' in file:
        n_sl = len('_short_emittance.h5')
        if 'corr' in file:
            n_sl += len('_corr')
        elif 'Acq' in file:
            n_sl += len('_Acq00')
    elif 'nominal' in file:
        n_sl = len('_nominal_emittance.h5')
        if 'corr' in file:
            n_sl += len('_corr')
        elif 'Acq' in file:
            n_sl += len('_Acq00')
    else:
        n_sl = 0

    return int(file[-(1 + n_sl)])
