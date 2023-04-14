'''
Functions to manipulate IQ-signals and signals related to feedbacks.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np


def from_iq_to_waveform(signal, t, omega_c):
    r'''
    Converts an IQ-signal to the demodulated waveform.

    :param signal: complex IQ-signal
    :param t: time-array related to the IQ-signal
    :param omega_c: the IQ carrier angular frequency
    :return: demodulated signal
    '''
    amp = np.abs(signal)
    phase = np.angle(signal)

    return amp * np.sin(omega_c * t + phase + np.pi / 2)

