'''
Functions to analyse signals from measurements.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt

def reshape_buffer_data(data, n_points):
    turns = len(data)//n_points
    rdata = np.zeros((n_points, turns), dtype=complex)

    for i in range(turns):
        rdata[:, i] = data[n_points * i: n_points * (i + 1)]

    return rdata


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
        plt.figure()
        plt.plot(tt, yy)
        plt.show()

    A, w, p, c, alpha = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) * np.exp(-alpha * t) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "alpha": alpha, "freq": f,
            "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}
