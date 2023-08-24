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
    r''' Digital feedback transfer function.

    :param f: frequency to evaluate the function at [Hz]
    :param g_a: Analog feedback gain [A/V]
    :param g_d: Digital feedback gain [-]
    :param tau_d: Digital feedback delay [s]
    :param dphi_ad: Phase shift between the digital and analog feedback [degrees]
    :return: Complex value of the transfer at f
    '''

    s = 1j * 2 * np.pi * f
    dphi_ad = dphi_ad * np.pi / 180
    return g_a * g_d * np.exp(1j * dphi_ad) / (1 + tau_d * s)


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


def H_cl(f, H_a, H_d, Z_cav, tau_loop=1.2e-6):
    r'''Closed loop response of the LHC Cavity Loop without OTFB.

    :param f: frequency to evaluate the function at [Hz]
    :param tau_loop: Overall loop delay [s]
    :param H_a: Analog feedback transfer function at f [-]
    :param H_d: Digital feedback transfer function at f [-]
    :param Z_cav: LHC Cavity transfer function at f [ohm]
    :return: Complex value of the transfer at f
    '''
    H_ad = H_a + H_d
    s = 1j * 2 * np.pi * f
    return 2 * np.exp(-tau_loop * s) * H_ad * Z_cav / (1 + 2 * np.exp(-tau_loop * s) * H_ad * Z_cav)


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
