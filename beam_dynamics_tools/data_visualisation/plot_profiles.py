'''
Functions to plot profile data.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from beam_dynamics_tools.beam_profiles.bunch_profile_tools import getBeamPattern
import beam_dynamics_tools.analytical_functions.longitudinal_beam_dynamics as lbd


def plot_cavity(bpos, t, blen, i_fit, f_fit, cavID):

    plt.figure()
    plt.title(f'Bunch Position, {cavID}')
    plt.plot(t, bpos, label='M')
    plt.plot(t, i_fit['fitfunc'](t), label='FIT I')
    plt.plot(t, f_fit['fitfunc'](t), label='FIT F')
    plt.legend()

    plt.figure()
    plt.title(f'Bunch Length, {cavID}')
    plt.plot(t, blen)


def plot_cavity_by_cavity_voltage(cavities, freqs_init, freqs_final, V, QL, add_str=''):

    V_init1 = lbd.RF_voltage_from_synchrotron_frequency(freqs_init[:, 0])
    V_init2 = lbd.RF_voltage_from_synchrotron_frequency(freqs_init[:, 1], eta=lbd.eta2)
    V_final1 = lbd.RF_voltage_from_synchrotron_frequency(freqs_final[:, 0])
    V_final2 = lbd.RF_voltage_from_synchrotron_frequency(freqs_final[:, 1], eta=mre.eta2)

    fig, ax1 = plt.subplots()

    ax1.set_title(f'$V$ = {V} MV, $Q_L$ = {QL}k{add_str}')
    ax1.plot(cavities, freqs_init[:, 0], 'x', color='b')
    ax1.plot(cavities, freqs_final[:, 0], 'D', color='b')

    ax1.plot(cavities, freqs_init[:, 1], 'x', color='r')
    ax1.plot(cavities, freqs_final[:, 1], 'D', color='r')

    ax1.set_xlabel('Cavity Number [-]')
    ax1.set_ylabel('Synchrotron Frequency [Hz]')
    ax1.grid()
    ax1.set_xticks(cavities)

    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    mn = lbd.RF_voltage_from_synchrotron_frequency(mn)
    mx = lbd.RF_voltage_from_synchrotron_frequency(mx)
    V_s = 1e-6
    ax2.set_ylim(mn * V_s, mx * V_s)
    ax2.set_ylabel('RF Voltage [MV]')

    V_s = 1e-6
    fig, ax1 = plt.subplots()
    ax1.set_title(f'Measured Voltage, $V$ = {V} MV, $Q_L$ = {QL}k{add_str}')
    ax1.plot(cavities, V_init1 * V_s, 'x', color='b')
    ax1.plot(cavities, V_final1 * V_s, 'D', color='b')

    ax1.plot(cavities, V_init2 * V_s, 'x', color='r')
    ax1.plot(cavities, V_final2 * V_s, 'D', color='r')

    ax1.set_xlabel('Cavity Number [-]')
    ax1.set_ylabel('RF Voltage [MV]')
    ax1.set_xticks(cavities)
    ax1.grid()

    dummy_lines = []
    linestyles = ['x', 'D']
    for i in range(2):
        dummy_lines.append(ax1.plot([], [], linestyles[i], c="black")[0])
    lines = ax1.get_lines()
    legend2 = ax1.legend([dummy_lines[i] for i in [0, 1]], ["Initial", "Final"])



def plot_cavity_by_cavity(cavities, title, ylabel, *args):
    fig, ax = plt.subplots()

    ax.set_title(title)
    markers = ['x', 'D', '.']
    i = 0
    for arr in args:
        ax.plot(cavities, arr[:, 0], markers[i], color='b')

        ax.plot(cavities, arr[:, 1], markers[i], color='r')
        i += 1

    ax.set_xlabel('Cavity Number [-]')
    ax.set_ylabel(ylabel)
    ax.set_xticks(cavities)
    ax.grid()

    dummy_lines = []
    linestyles = ['x', 'D']
    for i in range(2):
        dummy_lines.append(ax.plot([], [], linestyles[i], c="black")[0])
    lines = ax.get_lines()
    legend2 = ax.legend([dummy_lines[i] for i in [0, 1]], ["Initial", "Final"])


def plot_profile(Profile, turn, save_to):
    fig, ax = plt.subplots()

    ax.set_title(f'Profile at turn {turn}')
    ax.plot(Profile.bin_centers * 1e9, Profile.n_macroparticles)
    ax.set_xlabel(r'$\Delta t$ [ns]')
    ax.set_ylabel(r'$N_m$ [-]')

    fig.savefig(save_to + f'profile_{turn}.png')


def plot_bunch_position(bp, time, j, save_to, COM=False):
    fig, ax = plt.subplots()

    if COM:
        ax.set_title('Bunch Position COM')
    else:
        ax.set_title('Bunch Position')

    ax.plot(time[:j], bp[:j])
    ax.set_xlabel(r'Time since injection [s]')
    ax.set_ylabel(r'Bunch position')

    if COM:
        fig.savefig(save_to + 'bunch_position_com.png')
    else:
        fig.savefig(save_to + 'bunch_position.png')


def plot_bunch_phase_offsets(phase_offset, turn, save_to):
    fig, ax = plt.subplots()

    ax.set_title(f'Phase offset, Turn {turn}')
    ax.plot(phase_offset * 1e3)
    ax.set_xlabel(r'Bunch number [-]')
    ax.set_ylabel(r'Bunch-by-bunch phase offset [ps]')

    plt.savefig(save_to + f'phase_offset_{turn}.png')


def plot_total_losses(bloss, time, j, save_to, caploss=None, beam_structure=None):
    fig, ax = plt.subplots()

    if beam_structure is None:
        ax.plot(time[:j], np.sum(bloss, axis=1)[:j], c='black')
    else:
        bunches = np.zeros(1, dtype=int) - 1
        cmap = plt.get_cmap('plasma', len(beam_structure))
        for inj in range(len(beam_structure)):
            bunches = bunches[-1] + 1 + np.arange(beam_structure[inj], dtype=int)
            ax.plot(time[:j], np.sum(bloss[:, bunches], axis=1)[:j], c=cmap(inj))


    ax.set_ylabel(r'Losses [Num. Protons]')
    ax.set_xlabel(r'Time since injection [s]')
    if caploss is not None:
        ax.hlines(caploss, 0, time[j], colors='r')

    fig.savefig(save_to + 'total_losses.png')


def plot_bunch_length(bl, time, j, save_to):
    fig, ax = plt.subplots()

    ax.set_title('Bunch Length')

    ax.plot(time[:j], bl[:j])
    ax.set_xlabel(r'Time since injection [s]')
    ax.set_ylabel(r'Bunch length')

    fig.savefig(save_to + 'bunch_length.png')


def plot_phase_space(Beam, des, dts):
    data = {r'$\Delta E$': Beam.dE * 1e-6, r'$\Delta t$': Beam.dt * 1e9}
    cp = sns.color_palette('coolwarm', as_cmap=True)
    sns.displot(data, x=r'$\Delta t$', y=r'$\Delta E$', cbar=True, cmap=cp, vmin=0, vmax=150)
    plt.xlabel(r'$\Delta t$ [ns]')
    plt.ylabel(r'$\Delta E$ [MeV]')
    plt.xlim((-1.25, 2.5 + 1.25))
    plt.ylim((-600, 600))

    plt.plot(dts * 1e9, des * 1e-6, color='black')
    plt.plot(dts * 1e9, -des * 1e-6, color='black')


def find_weird_bunches(profiles, ts, minimum_bl=1.0, PLOT=False):
    N_bunches, Bunch_positions, Bunch_peaks, Bunch_lengths, Bunch_intensities, Bunch_positionsFit, \
    Bunch_peaksFit, Bunch_Exponent, Goodness_of_fit = getBeamPattern(ts[:, 0], profiles, heightFactor=30,
                                                                     wind_len=5, fit_option='fwhm')
    if PLOT:
        plt.figure()
        plt.plot(Bunch_lengths, '.')

    ids = []
    for i in range(len(Bunch_lengths[:])):
        if Bunch_lengths[i][0] < minimum_bl:
            ids.append(i)

    return np.array(ids)