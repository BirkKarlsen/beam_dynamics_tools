'''
Functions used to generate beams for simulation.

Author: Birk Emil Karlsen-Bæck
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
                Delta_t += 4 * bunch_spacing
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
                if counter_8b4e == 8 and '8b4e' in beam_type:
                    Delta_t += 4 * bunch_spacing
                    counter_8b4e = 0
                else:
                    Delta_t += bunch_spacing

            if k != N_ps_batches - 1:
                Delta_t += ps_batch_spacing

    return bunch_positions