'''
Functions taken from analytical expressions from longitudinal beam dynamics.

Author: Birk Emil Karlsen-Baeck
'''

# Import
import numpy as np
import scipy.constants as spc
import scipy.integrate as spi
from scipy.interpolate import interp1d
from scipy.special import jv, ellipk
from scipy.optimize import fsolve

from blond.impedances.impedance_sources import Resonators

# Physical Parameters for the MD
q = 1                           # [e]
phi_s = 0                       # [rad]
p_s = 450e9                     # [eV]
h = 35640                       # [-]
gamma_t1 = 53.606713            # [-]
gamma_t2 = 53.674152            # [-]
T_rev = 8.892465516509656e-05   # [s]
m_p = spc.physical_constants['proton mass energy equivalent in MeV'][0] * 1e6
E_s = np.sqrt(p_s**2 + m_p**2)
gamma_s = E_s / m_p
eta1 = (1/gamma_t1**2) - (1/gamma_s**2)
eta2 = (1/gamma_t2**2) - (1/gamma_s**2)
beta = np.sqrt(1 - 1/gamma_s**2)

gamma_t_SPS = 18.0
eta_SPS = (1/gamma_t_SPS**2) - (1/gamma_s**2)
h_SPS = 4620
R_SPS = 1100.009
T_rev_SPS = 2 * np.pi * R_SPS / (beta * spc.c)

# Functions

def synchrotron_frequency(V, h=h, eta=eta1, q=q, phi_s=phi_s, beta=beta, T_rev=T_rev, E_s=E_s):
    r'''Relation for synchrotron frequency when assuming small amplitude oscillations in
    longitudinal phase-space.

    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :param V: RF voltage [V]
    :param phi_s: Synchronous phase [rad]
    :param beta: Relativistic beta factor [-]
    :param T_rev: Revolution period [s]
    :param E_s: Synchronous Energy [eV]
    :return: Synchrotron frequency at the bucket center
    '''
    return np.sqrt((2 * np.pi * h * eta * q * V * np.cos(phi_s))/(beta**2 * T_rev**2 * E_s)) / (2 * np.pi)


def RF_voltage_from_synchrotron_frequency(f_s, E_s=E_s, beta=beta, T_rev=T_rev, h=h, eta=eta1, q=q, phi_s=phi_s):
    r'''Relation between the RF voltage and synchrotron frequency when assuming small amplitude oscillations in
    longitudinal phase-space.

    :param f_s: Synchrontron frequency [Hz]
    :param E_s: Synchronous Energy [eV]
    :param beta: Relativistic beta factor [-]
    :param T_rev: Revolution period [s]
    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :param phi_s: Synchtronous phase [rad]
    :return: RF voltage
    '''
    omega_s = 2 * np.pi * f_s
    return (E_s * omega_s**2 * beta**2 * T_rev**2)/(2 * np.pi * h * eta * q * np.cos(phi_s))


def RF_voltage_from_synchrotron_frequency_second_order(f_s, dt, E_s=E_s, beta=beta, T_rev=T_rev, h=h, eta=eta1, q=q):
    r'''RF voltage calculated from synchrotron frequency with a second order correction.

    :param f_s: Synchrontron frequency [Hz]
    :param dt: Distance from bucket center [s]
    :param E_s: Synchronous Energy [eV]
    :param beta: Relativistic beta factor [-]
    :param T_rev: Revolution period [s]
    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :return: RF voltage at dt from bucket center
    '''
    omega_s = 2 * np.pi * f_s
    A = (E_s * omega_s**2 * beta**2 * T_rev**2)
    B = (2 * np.pi * h * eta * q * np.cos(phi_s))
    omega_rf = 2 * np.pi * h / T_rev
    return A/B * 1/(1 - omega_rf**2 * dt**2 / 6)


def synchrotron_frequency_off_center(V, phi_max, h=h, eta=eta1, q=q, beta=beta, T_rev=T_rev, E_s=E_s):
    r'''Synchrotron frequency away from the RF bucket center.

    :param V: RF voltage [V]
    :param phi_max: distance away from RF bucket center in phase [rad]
    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :param beta: Relativistic beta factor [-]
    :param T_rev: Revolution period [s]
    :param E_s: Synchronous Energy [eV]
    :return: Synchrotron frequency at phi_max away from RF bucket center
    '''
    nu_s = np.sqrt((h * q * V * eta)/(2 * np.pi * beta**2 * E_s))
    K = ellipk(np.sin(phi_max / 2)**2)

    return np.pi * nu_s / (2 * T_rev * K)


def rf_bucket_height(V, h=h, eta=eta1, q=q, beta=beta, E_s=E_s, phi_s=phi_s):
    A = np.sqrt((2 * q * V * beta**2 * E_s)/(np.pi * h * np.abs(eta)))
    B = np.sqrt(np.abs(-np.cos(phi_s) + 0.5 * (np.pi - 2 * phi_s) * np.sin(phi_s)))
    return A * B


def phase_offset_from_energy_offset(dE, V, h=h, eta=eta1, q=q, beta=beta, E_s=E_s):
    r'''Offset in phase corresponding to an offset in energy for a single RF system.

    :param dE: Energy offset [eV]
    :param V: RF voltage [V]
    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :param beta: Relativistic beta factor [-]
    :param E_s: Synchronous Energy [eV]
    :return: phase offset
    '''
    return np.pi * np.sqrt((h * eta * np.pi)/(2 * beta**2 * E_s * q * V)) * dE


def calc_Z_pot(Z, Z_omega, omega_rev, S, S_omega, tau_max, n_harm=20):
    ImZ = Z
    S = S

    p = np.linspace(-n_harm, n_harm, 2 * n_harm + 1, dtype=int)

    Z_func = interp1d(Z_omega, ImZ, fill_value=0, bounds_error=False)
    S_func = interp1d(S_omega, S, fill_value=0, bounds_error=False)

    Z_for_sum = Z_func(omega_rev * p)
    S_for_sum = S_func(omega_rev * p)
    J_for_sum = jv(1, p * omega_rev * tau_max)

    return 2 * np.sum(Z_for_sum * J_for_sum * S_for_sum).imag / (omega_rev * tau_max)


def synchrotron_frequency_with_intensity_effect(V, phi_max, I_b, Z_dict, S_dict, h=h, eta=eta1, q=q, beta=beta,
                                                T_rev=T_rev, E_s=E_s, n_harm=20, phi_s=phi_s):
    tau_max = T_rev * phi_max / (h * 2 * np.pi)

    nu_s0 = np.sqrt((h * eta * q * V * np.cos(phi_s))/(2 * np.pi * beta**2 * E_s))

    if tau_max.ndim > 0:
        Zpot = np.zeros(tau_max.shape)
        for i in range(tau_max.shape[0]):
            Zpot[i] = calc_Z_pot(Z=Z_dict['Z'], Z_omega=Z_dict['omega'], omega_rev=(2 * np.pi)/T_rev,
                                 S=S_dict['S'], S_omega=S_dict['omega'], tau_max=tau_max[i], n_harm=n_harm)
    else:
        Zpot = calc_Z_pot(Z=Z_dict['Z'], Z_omega=Z_dict['omega'], omega_rev=(2 * np.pi) / T_rev,
                          S=S_dict['S'], S_omega=S_dict['omega'], tau_max=tau_max, n_harm=n_harm)

    xi = 2 * np.pi * I_b / (h * V * (-1) * np.cos(phi_s))

    return (1/T_rev) * nu_s0 * np.sqrt(1 - (xi / (2 * np.pi)) * Zpot)


def synchrotron_frequency_shift_with_laclaire(V, N, tau_b, ImZ_over_n=0.07, T_rev=T_rev, h=h, phi_s=phi_s):
    r'''Synchrotron frequency shift with a first order shift from intensity effects as computed from Laclaire.

    :param V: RF voltage [V]
    :param N: Number of particles per bunch [-]
    :param tau_b: Bunch length [s]
    :param ImZ_over_n: Characteristic Impedance [ohm]
    :param T_rev: Revolution period [s]
    :param h: Harmonic number [-]
    :param phi_s: Synchronous phase [rad]
    :return: Synchrotron frequency shift
    '''
    return 12 * spc.e * N / (V * h * np.cos(phi_s) * (2 * np.pi/T_rev)**2 * tau_b**3) * ImZ_over_n


def synchrotron_correction_intensity(V, N, tau_b, ImZ_over_n=0.07, T_rev=T_rev, h=h, phi_s=phi_s):
    r'''Synchrotron frequency including the Laclaire intensity correction.

    :param V: RF voltage [V]
    :param N: Number of particles per bunch [-]
    :param tau_b: Bunch length [s]
    :param ImZ_over_n: Characteristic Impedance [ohm]
    :param T_rev: Revolution period [s]
    :param h: Harmonic number [-]
    :param phi_s: Synchronous phase [rad]
    :return: Synchrotron frequency with intensity shift
    '''
    D = synchrotron_frequency_shift_with_laclaire(V, N, tau_b, ImZ_over_n=ImZ_over_n, T_rev=T_rev, h=h, phi_s=phi_s)
    return np.sqrt(1 - D)


def synchrotron_correction_energy_error(dE, V, h=h, eta=eta1, q=q, beta=beta, T_rev=T_rev, E_s=E_s):
    r'''Synchrotron frequency when taking energy offsets into account.

    :param dE: Eneergy offset [eV]
    :param V: RF voltage [V]
    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :param beta: Relativistic beta factor [-]
    :param T_rev: Revolution period [s]
    :param E_s: Synchronous energy [eV]
    :return: Synchrotron frequency with a shift due to an energy offset.
    '''

    dphi = phase_offset_from_energy_offset(dE, V, h, eta, q, beta, E_s)
    freq_s = synchrotron_frequency_off_center(V, dphi, h=h, eta=eta1, q=q, beta=beta, T_rev=T_rev, E_s=E_s)
    freq_0 = synchrotron_frequency(V, h=h, eta=eta1, q=q, phi_s=phi_s, beta=beta, T_rev=T_rev, E_s=E_s)
    return freq_0 - freq_s


def find_theoretical_total_voltage_scan(dEs, Vs, h=h, eta=eta1, q=q, beta=beta, T_rev=T_rev, E_s=E_s):
    dfreqs_tot = np.zeros((len(Vs), len(dEs)))
    freq_0s = np.zeros(len(Vs))
    relative_freq = np.zeros((len(Vs), len(dEs)))
    for i in range(len(Vs)):
        freq_0 = synchrotron_frequency(Vs[i] * 1e6, h=h, eta=eta, q=q, phi_s=phi_s,
                                       beta=beta, T_rev=T_rev, E_s=E_s)
        freq_0s[i] = freq_0
        dfreqs = synchrotron_correction_energy_error(dEs * 1e6, Vs[i] * 1e6, h=h, eta=eta,
                                                     q=q, beta=beta, T_rev=T_rev, E_s=E_s)
        dfreqs_tot[i, :] = dfreqs

        relative_freq[i, :] = (freq_0s[i] - dfreqs_tot[i, :]) / freq_0s[i]

    return dfreqs_tot, freq_0s, relative_freq


def calc_I(phi_b):
    r'''Solving the integral that arises when calculating the longitudinal emittance for a single RF system.

    :param phi_b: Phase corresponding to bunch length [rad]
    :return: I
    '''
    f = lambda phi : np.sqrt(2 * (np.cos(phi) - np.cos(phi_b)))
    return spi.quad(f, 0, phi_b)[0]


def find_emittance_single_RF(tau_b, V, h=h, eta=eta1, q=q, phi_s=phi_s, beta=beta, T_rev=T_rev, E_s=E_s):
    r'''Computing the longitudinal emittance for a single RF system.

    :param tau_b: Bunch length [s]
    :param V: RF voltage [V]
    :param h: Harmonic number [-]
    :param eta: Momentum compaction factor [-]
    :param q: Electric unit charge [e]
    :param phi_s: Synchronous phase [rad]
    :param beta: Relativistic beta factor [-]
    :param T_rev: Revolution period [s]
    :param E_s: Synchronous energy [eV]
    :return: Longitudinal emittance [eVs]
    '''
    omega_s = 2 * np.pi * synchrotron_frequency(V, h, eta, q, phi_s, beta, T_rev, E_s)
    omega_rf = 2 * np.pi * h / T_rev
    phi_b1 = omega_rf * tau_b / 2
    return (4 * E_s * omega_s * beta**2)/(omega_rf**2 * np.abs(eta)) * calc_I(phi_b1)


def find_bunch_length(tau_b, tau_b_guess=[0, 2.5e-9], q=q, beta=beta, E_s=E_s, SPS_dict=None, LHC_dict=None,
                      rel_tol=1e-7, max_it=1000, print_progress=False):
    r'''Find the corresponding bunch length in the LHC based on the bunch length in the SPS using the bisection method.

    :param tau_b: Bunch length in the SPS [s]
    :param tau_b_guess: List of two LHC bunch length guess [s]
    :param q: Electric unit charge [e]
    :param beta: Relativistic beta factor [-]
    :param E_s: Synchronous energy [eV]
    :param SPS_dict: Dictionary with SPS machine parameters
    :param LHC_dict: Dictionary with LHC machine parameters
    :param rel_tol: Relative tolerance for the bisection method
    :param max_it: Maximum number of iteration of the bisection method
    :param print_progress: Option for the function to print the computation progress
    :return: Bunch length in the LHC
    '''

    if SPS_dict is None:
        SPS_dict = {'V': 4.9e6, 'h': h_SPS, 'eta': eta_SPS, 'phi_s': 0, 'T_rev' : T_rev_SPS}

    if LHC_dict is None:
        LHC_dict = {'V': 1.5e6, 'h': h, 'eta': eta1, 'phi_s': 0, 'T_rev' : T_rev}

    emittance = find_emittance_single_RF(tau_b, SPS_dict['V'], SPS_dict['h'], SPS_dict['eta'],
                                         q, SPS_dict['phi_s'], beta, SPS_dict['T_rev'], E_s)

    tau_b_a = tau_b_guess[0]
    emittance_a = find_emittance_single_RF(tau_b_a, LHC_dict['V'], LHC_dict['h'], LHC_dict['eta'],
                                           q, LHC_dict['phi_s'], beta, LHC_dict['T_rev'], E_s)

    tau_b_b = tau_b_guess[1]

    tau_b_c = (tau_b_a + tau_b_b)/2
    emittance_c = find_emittance_single_RF(tau_b_c, LHC_dict['V'], LHC_dict['h'], LHC_dict['eta'],
                                           q, LHC_dict['phi_s'], beta, LHC_dict['T_rev'], E_s)

    err = (emittance - emittance_c)/emittance
    i = 0
    while np.abs(err) > rel_tol:
        i += 1
        if i > max_it:
            print('Failed to find emittance')
            return tau_b_c, emittance_c

        if (emittance - emittance_c) * (emittance - emittance_a) < 0:
            tau_b_b = tau_b_c
        else:
            tau_b_a = tau_b_c
            emittance_a = emittance_c

        tau_b_c = (tau_b_a + tau_b_b)/2
        emittance_c = find_emittance_single_RF(tau_b_c, LHC_dict['V'], LHC_dict['h'], LHC_dict['eta'],
                                               q, LHC_dict['phi_s'], beta, LHC_dict['T_rev'], E_s)
        err = (emittance - emittance_c) / emittance
        if print_progress:
            print(f'iteration {i}, error is {err} and bunch length is {tau_b_c}')

    return tau_b_c, emittance_c


def final_bunch_length_voltage_scan(Vs, bls, beamline=1, rel_tol=1e-7, max_it=1000):
    r'''Calculating the corresponding LHC bunch length based on RF voltages in the LHC as well as bunch lengths in the
    SPS.

    :param Vs: Array of LHC capture voltages [V]
    :param bls: Array of bunch lengths in the SPS [s]
    :param beamline: Beamline the bunches are injected to
    :param rel_tol: Relative tolerance for the bisection method
    :param max_it: Maximum number of iteration for the bisection method
    :return: Array of LHC bunch lenghts [s]
    '''
    tau_bs = np.zeros(Vs.shape)

    for i in range(len(Vs)):
        SPS_dict = {'V': 4.9e6, 'h': h_SPS, 'eta': eta_SPS, 'phi_s': 0, 'T_rev': T_rev_SPS}
        if beamline == 1:
            LHC_dict = {'V': Vs[i] * 1e6, 'h': h, 'eta': eta1, 'phi_s': 0, 'T_rev': T_rev}
        else:
            LHC_dict = {'V': Vs[i] * 1e6, 'h': h, 'eta': eta2, 'phi_s': 0, 'T_rev': T_rev}

        tau_b, emittance = find_bunch_length(bls[i] * 1e-9, SPS_dict=SPS_dict, LHC_dict=LHC_dict,
                                             max_it=max_it, rel_tol=rel_tol)
        tau_bs[i] = tau_b

    return tau_bs


def LHC_analytic_generator_current(V_a, Q_L=20000, R_over_Q=45, df=0, f_c=400.789e6, I_b=0.6, return_power=False):
    r'''Calculates the generator current assuming constant antenna
    voltage and beam current.'''

    I_g = V_a / (2 * R_over_Q) * (1/Q_L - 2 * 1j * df/f_c) + 1/2 * I_b

    if return_power:
        power = 1/2 * R_over_Q * Q_L * np.abs(I_g)**2
        return I_g, power
    else:
        return I_g


def LHC_analytic_power_half_detuning(V_a, I_b):
    return V_a * I_b / 8


def generator_power(Ig, R_over_Q, Q_L):
    return 0.5 * R_over_Q * Q_L * np.absolute(Ig)**2


def convert_detuning_to_phase(df, LHCCavityLoop=None, R_over_Q=45, f_r=400.789e6, Q_L=20000, deg=False):
    r'''
    Function to convert a given detuning value in Hz to phase in degrees or rad
    :param df: Detuning in frequency [Hz]
    :param LHCCavityLoop: BLonD LHC cavity loop object
    :param R_over_Q: Cavity R over Q
    :param f_r: Resonant frequency of the cavity [Hz]
    :param Q_L: Loaded quality factor of cavity
    :param deg: Option to get phase in degrees or radians
    :return: Corresponding phase
    '''
    if LHCCavityLoop is not None:
        R_over_Q = LHCCavityLoop.R_over_Q
        Q_L = LHCCavityLoop.Q_L
        f_r = LHCCavityLoop.omega_c / (2 * np.pi)

    R_S = R_over_Q * Q_L

    resonator = lambda f: R_S / (1 + 1j * Q_L * (f / f_r - f_r / f))

    return np.angle(resonator(f_r + df), deg=deg)

def convert_phase_to_detuning(phase, guess, LHCCavityLoop=None, R_over_Q=45, f_r=400.789e6, Q_L=20000):
    r'''
    Function to convert a given detuning value in Hz to phase in degrees or rad
    :param phase: Detuning phase
    :param LHCCavityLoop: BLonD LHC cavity loop object
    :param R_over_Q: Cavity R over Q
    :param f_r: Resonant frequency of the cavity [Hz]
    :param Q_L: Loaded quality factor of cavity
    :return: Corresponding frequency
    '''
    if LHCCavityLoop is not None:
        R_over_Q = LHCCavityLoop.R_over_Q
        Q_L = LHCCavityLoop.Q_L
        f_r = LHCCavityLoop.omega_c / (2 * np.pi)

    R_S = R_over_Q * Q_L

    Z = lambda f: R_S / (1 + 1j * Q_L * (f/f_r - f_r/f))
    Z_phase = lambda f: np.angle(Z(f), deg=True) - phase

    return fsolve(Z_phase, f_r + guess)[0] - f_r



