'''
File with funtions to handle data from TIMBER.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import csv
import os


def fetch_data_from_file(fdir, fname, pick_lines, line_len, skip_rows=3):
    data = np.zeros((len(pick_lines), line_len))

    with open(fdir + fname, 'r') as file:
        csvreader = csv.reader(file)
        i = 0
        j = 0
        l = 0
        for row in csvreader:
            if i >= skip_rows:
                if j in pick_lines:
                    data[l] = row[1:]
                    l += 1
                j += 1
            i += 1

    return data


def fetch_multiple_variables_from_file(fdir, fname, pick_lines, line_len, n_var, skip_rows):

    data = np.zeros((n_var, len(pick_lines), line_len))

    with open(fdir + fname, 'r') as file:
        csvreader = csv.reader(file)
        i = 0
        j = 0
        k = -1
        l = 0
        for row in csvreader:
            if len(row) > 0 and 'VARIABLE' in row[0]:
                i = 0
                j = 0
                l = 0
                k += 1

            if i >= skip_rows:
                if j in pick_lines and len(row[1:]) == line_len:
                    data[k, l, :] = row[1:]
                    l += 1
                j += 1
            i += 1

    return data