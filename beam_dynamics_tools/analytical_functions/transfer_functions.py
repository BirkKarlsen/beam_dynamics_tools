'''
Transfer functions from the LHC Cavity Loop.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np


def H_a(f, g_a=6.79e-6, tau_a=170e-6):
    r'''Analog feedback transfer function.

    :param f: frequency to evaluate the function at [Hz]
    :param g_a: Analog feedback gain [A/V]
    :param tau_a: Analog feedback delay [s]
    :return: Complex value of the transfer at f
    '''

    s = 1j * 2 * np.pi * f
    return g_a * tau_a * s / (1 + tau_a * s)


def H_d(f, g_a=6.79e-6, g_d=10, tau_d=400e-6, dphi_ad=0):
    r'''Digital feedback transfer function.

    :param f: frequency to evaluate the function at [Hz]
    :param g_a: Analog feedback gain [A/V]
    :param g_d: Digital feedback gain [-]
    :param tau_d: Digital feedback delay [s]
    :param dphi_ad: Phase shift between the digital and analog feedback [degrees]
    :return: Complex value of the transfer function at f
    '''

    s = 1j * 2 * np.pi * f
    dphi_ad = dphi_ad * np.pi / 180
    return g_a * g_d * np.exp(1j * dphi_ad) / (1 + tau_d * s)


def H_otfb(f, g_otfb, alpha, tau_ac, tau_comp, t_rev):
    r'''The LHC one-turn delay feedback transfer function.
    The function includes the AC couplers at the input and output.

    :param f:
    :param g_otfb:
    :param alpha:
    :param tau_ac:
    :param tau_comp:
    :param t_rev:
    :return: Complex value of the transfer function at f
    '''

    # OTFB transfer function
    h_otfb = lambda s: g_otfb * (1 - alpha) * np.exp(-t_rev * s) / (1 - alpha * np.exp(-t_rev * s))

    # AC coupler transfer function
    h_ac = lambda s: tau_ac * s / (1 + tau_ac * s)

    # Complimentary delay
    h_comp = lambda s: np.exp(tau_comp * s)

    # Laplace space coordinate
    s_ = 1j * 2 * np.pi * f

    # Return the combined transfer function assuming they are in series
    return h_ac(s_) * h_otfb(s_) * h_ac(s_) * h_comp(s_)


def Z_cav(f, df=0, f_rf=400.789e6, R_over_Q=45, Q_L=20000):
    r'''Approximate expression of the LHC cavity transfer function.

    :param f: frequency to evaluate the function at [Hz]
    :param df: Detuning of the cavity in frequency [Hz]
    :param f_rf: RF frequency [Hz]
    :param R_over_Q: Cavity R/Q [-]
    :param Q_L: Cavity loaded Q factor [-]
    :return: Complex value of the transfer at f
    '''

    s = 1j * 2 * np.pi * f
    domega = 2 * np.pi * df
    omega_rf = 2 * np.pi * f_rf

    return R_over_Q * Q_L / (1 + 2 * Q_L * (s - 1j * domega)/omega_rf)


def H_cl(f, H_a, H_d, Z_cav, tau_loop=1.2e-6, H_otfb=None):
    r'''Closed loop response of the LHC Cavity Loop without OTFB.

    :param f: frequency to evaluate the function at [Hz]
    :param tau_loop: Overall loop delay [s]
    :param H_a: Analog feedback transfer function at f [-]
    :param H_d: Digital feedback transfer function at f [-]
    :param Z_cav: LHC Cavity transfer function at f [ohm]
    :return: Complex value of the transfer at f
    '''
    if H_otfb is None:
        H_ad = H_a + H_d
    else:
        H_ad = H_a * (H_otfb + 1) + H_d

    s = 1j * 2 * np.pi * f
    return 2 * np.exp(-tau_loop * s) * H_ad * Z_cav / (1 + 2 * np.exp(-tau_loop * s) * H_ad * Z_cav)


def Z_cl(f, H_a, H_d, Z_cav, tau_loop=1.2e-6):
    H_ad = lambda f: H_a(f) + H_d(f)
    s = 1j * 2 * np.pi * f
    return Z_cav / (1 + 2 * np.exp(-tau_loop * s) * H_ad(f) * Z_cav)


def H_open(f, tau_loop, H_a, H_d, Z_cav):
    r'''Open loop response of the LHC Cavity Loop without OTFB.

    :param f: frequency to evaluate the function at [Hz]
    :param tau_loop: Overall loop delay [s]
    :param H_a: Analog feedback transfer function at f [-]
    :param H_d: Digital feedback transfer function at f [-]
    :param Z_cav: LHC Cavity transfer function at f [-]
    :return: Complex value of the transfer at f
    '''
    H_ad = H_a + H_d
    s = 1j * 2 * np.pi * f
    return 2 * np.exp(-tau_loop * s) * H_ad * Z_cav
