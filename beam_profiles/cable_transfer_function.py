'''
Function to apply the cable transfer function to beam profile measurements.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import h5py

def apply_cable_tf(profile, t, beam, extend=100e-9):
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
    freq = np.fft.fftfreq(noints, d=dt)

    h5file = h5py.File(tf_dir + 'TF_B' + str(beam) + '.h5', 'r')
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
