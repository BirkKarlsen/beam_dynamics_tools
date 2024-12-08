'''
Functions to analyse signals from measurements.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def reshape_buffer_data(data, n_points):
    turns = len(data)//n_points
    rdata = np.zeros((n_points, turns), dtype=complex)

    for i in range(turns):
        rdata[:, i] = data[n_points * i: n_points * (i + 1)]

    return rdata


def reshape_to_turn_by_turn(data, t, t_rev):
    n_complete_turns = int(t[-1] / t_rev)
    sample_period = t[1] - t[0]
    n_samples_per_turn = int(t_rev / sample_period)
    data_interped = interp1d(t, data)

    reshaped_data = np.zeros((n_complete_turns, n_samples_per_turn))
    t_turn = np.linspace(0, (n_samples_per_turn - 1) * sample_period, n_samples_per_turn)

    for i in range(n_complete_turns):
        reshaped_data[i, :] = data_interped(i * t_rev + t_turn)

    return reshaped_data


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset, 0])

    def sinfunc(t, A, w, p, c, alpha): return A * np.sin(w*t + p) * np.exp(-alpha * t) + c
    try:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    except:
        #plt.figure()
        #plt.plot(tt, yy)
        #plt.show()
        popt = (-1, -1, -1, -1, -1)
        pcov = np.zeros((len(popt), len(popt)))

    A, w, p, c, alpha = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) * np.exp(-alpha * t) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "alpha": alpha, "freq": f,
            "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}
