'''
Functions used to generate beams for simulation.

Author: Birk Emil Karlsen-Baeck
'''

import numpy as np

def generate_bunch_spacing(N_bunches, bunch_spacing, ps_batch_length, ps_batch_spacing, beam_type):
    r'''
    Generates bunch positions in the SPS for a given PS batch length, bunch spacing
    and beam type, e.g. BCMS or 8b4e.

    :return: numpy array with bunch positions in units of SPS RF buckets.
    '''
    bunch_positions = np.zeros(N_bunches)
    Delta_t = 0
    counter_8b4e = 0
    if N_bunches / ps_batch_length <= 1:
        for j in range(N_bunches):
            bunch_positions[j] = Delta_t
            counter_8b4e += 1
            if counter_8b4e == 8 and '8b4e' in beam_type:
                Delta_t += 5 * bunch_spacing
                counter_8b4e = 0
            else:
                Delta_t += bunch_spacing
    else:
        if N_bunches % ps_batch_length == 0:
            N_ps_batches = N_bunches // ps_batch_length
        else:
            N_ps_batches = N_bunches // ps_batch_length + 1
        l = 0
        for k in range(N_ps_batches):
            if k == 0:
                N_bunches_in_ps_batch = N_bunches - ps_batch_length * (N_ps_batches - 1)
            else:
                N_bunches_in_ps_batch = ps_batch_length

            for j in range(N_bunches_in_ps_batch):
                bunch_positions[l] = Delta_t
                l += 1
                counter_8b4e += 1
                if j != N_bunches_in_ps_batch - 1:
                    if counter_8b4e == 8 and '8b4e' in beam_type:
                        Delta_t += 5 * bunch_spacing
                        counter_8b4e = 0
                    else:
                        Delta_t += bunch_spacing

            if k != N_ps_batches - 1:
                Delta_t += ps_batch_spacing

    return bunch_positions


def generate_beam_ID(beam_type, number_bunches, ps_batch_length, intensity, bunchlength, voltage_200, voltage_800):
    r'''
    Generates a standard format for the ID of a generated beam in the SPS.

    :param beam_type: type of beam, e.g. BCMS or 8b4e
    :param number_bunches: total number of bunches in the batch
    :param ps_batch_length: number of bunches in the batches injected from the PS
    :param intensity: average bunch intensity
    :param bunchlength: average bunch length
    :param voltage_200: total RF voltage of the 200 MHz system
    :param voltage_800: total RF voltage of the 800 MHz system as a fraction of the 200 MHz system
    :return: the beam ID in form of a string
    '''
    return f'{beam_type}_{number_bunches}b_{ps_batch_length}pslen_{intensity * 1e3:.0f}e8_' \
    f'{bunchlength * 1e3:.0f}ps_{voltage_200 * 1e3:.0f}kV_{voltage_800 * 100:.0f}percent'