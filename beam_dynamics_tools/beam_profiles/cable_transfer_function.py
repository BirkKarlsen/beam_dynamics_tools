'''
Function to apply the cable transfer function to beam profile measurements.

Author: Birk Emil Karlsen-Baeck, Danilo Quartullo
'''

import numpy as np
import h5py
import os
import pandas as pd


def raised_cosine_filter(cutoff_left, cutoff_right, freq_tf):
    H_RC = np.zeros(len(freq_tf))
    index_inbetween = np.where((freq_tf <= cutoff_right) & (freq_tf >= cutoff_left))[0]
    index_before = np.where(freq_tf < cutoff_left)[0]
    index_after = np.where(freq_tf > cutoff_right)[0]

    H_RC[index_before] = 1
    H_RC[index_after] = 0
    H_RC[index_inbetween] = (1 + np.cos(
        np.pi / (cutoff_right - cutoff_left) * (freq_tf[index_inbetween] - cutoff_left))) / 2

    return H_RC


def set_profile_reference(profiles, new_reference=0, sample=25):
    if profiles.ndim == 1:
        profiles = profiles - np.mean(profiles[:sample]) + new_reference
    else:
        for i in range(profiles.shape[1]):
            profiles[:, i] = profiles[:, i] - np.mean(profiles[:sample, i]) + new_reference

    return profiles


def apply_lhc_cable_tf(profile, t, beam, extend=100e-9):
    dir_fil = os.path.dirname(os.path.abspath(__file__)) + '/cable_transfer_functions/'
    profile = set_profile_reference(profile, new_reference=0, sample=25)
    dt = t[1] - t[0]
    init_length = len(t)

    # Extending the time array to improve the deconvolution
    noints = t.shape[0]
    t = np.arange(t.min(), t.max() + extend, dt)
    profile = np.concatenate((profile, np.zeros(t.shape[0] - noints)))

    # Recalculate the number of points and the frequency array
    noints = t.shape[0]
    freq = np.fft.fftfreq(noints, d=dt)

    h5file = h5py.File(dir_fil + 'TF_B' + str(beam) + '.h5', 'r')
    freq_array = np.array(h5file["/TransferFunction/freq"])
    TF_array = np.array(h5file["/TransferFunction/TF"])
    h5file.close()
    TF = np.interp(freq, np.fft.fftshift(freq_array), np.fft.fftshift(TF_array.real)) + \
         1j * np.interp(freq, np.fft.fftshift(freq_array), np.fft.fftshift(TF_array.imag))

    # Remove zeros in high-frequencies
    TF[TF == 0] = 1.0 + 0j

    # Deconvolution
    filtered_f = np.fft.fft(profile) / TF
    filtered = np.fft.ifft(filtered_f).real
    filtered -= filtered[:10].mean()
    return filtered[:init_length], t[:init_length]


def sps_cable_tranfer_function(profile_time, profile_current, year: int = 2021):
    r'''
    Takes in time-array and profile-array and applies the cable tranfer function of the SPS.

    code written by Danilo Quartullo
    :param profile_time: numpy-array with time sample points
    :param profile_current: numpy-array with profile measurement
    :param year: int the year the TF was measured
    :return: profile measurement with cable transfer function applied
    '''
    # Import the CTF unfiltered
    dir_fil = os.path.dirname(os.path.abspath(__file__)) + '/cable_transfer_functions/'
    data = np.load(dir_fil + f'cableTF_{year}.npz')
    tf = data['transfer']
    freq_tf = data['freqArray']
    Delta_f = freq_tf[1] - freq_tf[0]  # 40 MHz
    t_max = 1 / Delta_f
    f_max = freq_tf[-1]
    Delta_t = 1 / (2 * f_max)  # 50 ps
    n_fft = 2 * (len(tf) - 1)

    # Apply raised cosine filter
    H_RC = np.zeros(len(freq_tf))
    cutoff_left = 2.5e9
    cutoff_right = 3.0e9
    index_inbetween = np.where((freq_tf <= cutoff_right) & (freq_tf >= cutoff_left))[0]
    index_before = np.where(freq_tf < cutoff_left)[0]
    index_after = np.where(freq_tf > cutoff_right)[0]
    H_RC[index_before] = 1
    H_RC[index_after] = 0
    H_RC[index_inbetween] = (1 + np.cos(
        np.pi / (cutoff_right - cutoff_left) * (freq_tf[index_inbetween] - cutoff_left))) / 2
    tf_RC = tf * H_RC

    # Interpolate if dt != 50 ps
    profile_dt = profile_time[1] - profile_time[0]
    if profile_dt != Delta_t:
        n_fft_new = max(len(profile_time), n_fft)
        if n_fft_new % 2 != 0:
            n_fft_new += 1
        freq_tf_new = np.fft.rfftfreq(n_fft_new, profile_dt)
        tf_real_new = np.interp(freq_tf_new, freq_tf, np.real(tf_RC))
        tf_imag_new = np.interp(freq_tf_new, freq_tf, np.imag(tf_RC))
        tf_new = tf_real_new + 1j * tf_imag_new
    else:
        tf_new = tf_RC

    # Apply the filtered CTF
    profile_spectrum = np.fft.rfft(profile_current, n=n_fft_new)
    profSpectrum_CTF = profile_spectrum * tf_new
    CTF_profile = np.fft.irfft(profSpectrum_CTF)[:len(profile_time)]

    return CTF_profile


def apply_sps_cable_tf(
        profile,
        t,
        extend: float = 100e-9,
        year: int = 2021,
        raised_cos_filter: bool = False,
        apply_pickup: bool = False,
    ):
    tf_dir = f'../transfer_functions/'
    profile = set_profile_reference(profile, new_reference=0, sample=25)
    dt = t[1] - t[0]
    init_length = len(t)

    # Extending the time array to improve the deconvolution
    noints = t.shape[0]
    t = np.arange(t.min(), t.max() + extend, dt)
    profile = np.concatenate((profile, np.zeros(t.shape[0] - noints)))

    # Recalculate the number of points and the frequency array
    noints = t.shape[0]
    freq = np.fft.rfftfreq(noints, d=dt)

    dir_fil = os.path.dirname(os.path.abspath(__file__)) + '/cable_transfer_functions/'
    data = np.load(dir_fil + f'cableTF_{year}.npz')
    
    tf_array = data['transfer']
    freq_array = data['freqArray']

    # Apply pick up transfer function
    if apply_pickup:
        # Extend the pick up from 6 GHz to 10 GHz
        TF_pickup_data = pd.read_csv(
            dir_fil + 'tf-apwl10-sig8v4.dat', sep="\s+",
            skiprows=4, names=['Freq. [Hz]', 'Re', 'Im']
        )

        TF_pickup_freq = TF_pickup_data['Freq. [Hz]'].to_numpy()
        TF_pickup_array = (TF_pickup_data['Re'].to_numpy() + TF_pickup_data['Im'].to_numpy() * 1j)

        freq_step = TF_pickup_freq[1] - TF_pickup_freq[0]
        additional_points = int(np.ceil((10e9 - TF_pickup_freq[-1]) / freq_step))
        new_freq = np.linspace(TF_pickup_freq[-1] + freq_step, 10e9, additional_points, endpoint=True)

        new_TF_pickup_array = np.full_like(new_freq, TF_pickup_array[-1].real) \
                              + 1j * np.full_like(new_freq, TF_pickup_array[-1].imag)

        extended_TF_pickup_freq = np.concatenate((TF_pickup_freq, new_freq))
        extended_TF_pickup_array = np.concatenate((TF_pickup_array, new_TF_pickup_array))

        tf_pickup_fine = np.interp(
            freq_array,
            extended_TF_pickup_freq,
            extended_TF_pickup_array.real
        ) + 1j * np.interp(
            freq_array,
            extended_TF_pickup_freq,
            extended_TF_pickup_array.imag
        )

        tf_array = tf_array * abs(tf_pickup_fine)
    
    tf = np.interp(freq, freq_array, tf_array.real) + \
         1j * np.interp(freq, freq_array, tf_array.imag)

    if raised_cos_filter:
        cutoff_left = 2.40e9  # CTF reliable up to 2.5 GHz
        cutoff_right = 2.50e9
        H_RC = raised_cosine_filter(cutoff_left, cutoff_right, freq)

    # Remove zeros in high-frequencies
    #tf[tf == 0] = 1.0 + 0j

    # Deconvolution
    filtered_f = np.fft.rfft(profile, n=noints) / tf

    if raised_cos_filter:
        filtered_f = filtered_f * H_RC

    filtered = np.fft.irfft(filtered_f).real
    filtered -= filtered[:10].mean()
    return filtered[:init_length], t[:init_length]
