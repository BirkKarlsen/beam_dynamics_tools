'''
Functions to get data from directories.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import os

def get_power_measurements_from_folder(folder_name, setting=None, profile=False):
    r'''Function to get all power measurements with given properties.

    :param folder_name: Name of the data folder
    :param setting: Dictionary with how to discriminate the files
    :param profile: Special setting to enable if profiles are being fetched
    :return: list of the filenames fulfilling the criteria
    '''
    file_names = []
    indexes = []
    sorted_by = []
    signal = None
    date = None
    time = None
    beam = None
    cavity = None
    channel = None

    if setting is not None:
        if 'date' in setting:
            date = setting['date']

        if 'time' in setting:
            time = setting['time']

        if 'beam' in setting:
            beam = setting['beam']

        if 'cavity' in setting:
            cavity = setting['cavity']

        if 'signal' in setting:
            signal = setting['signal']

        if 'channel' in setting:
            channel = setting['channel']

    if date is not None:
        indexes.append([-14, -6])
        sorted_by.append(date)

    if time is not None:
        indexes.append([-6, -2])
        sorted_by.append(time)

    if beam is not None:
        if not profile:
            indexes.append([-21, -19])
        else:
            indexes.append([8, 9])
        sorted_by.append(beam)

    if signal is not None:
        indexes.append([0, -22])
        sorted_by.append(signal)

    if cavity is not None:
        indexes.append([-22, -21])
        sorted_by.append(cavity)

    if channel is not None:
        indexes.append([-18, -15])
        sorted_by.append(channel)

    for file in os.listdir(folder_name):
        valid = True
        file_i = file[:file.index('.')]

        for i in range(len(indexes)):
            part_i = file_i[indexes[i][0]:indexes[i][1]]
            if part_i != sorted_by[i]:
                valid = False

        if valid:
            file_names.append(file)

    return file_names


def check_file_exist_in_both_directories(filename, dir1, dir2):
    r'''Checks if the file exists in directories dir1 and dir2.

    :param filename: Name of the file
    :param dir1: Name of dir1
    :param dir2: Name of dir2
    :return: Either True or False
    '''

    if os.path.isfile(dir1 + filename) and os.path.isfile(dir2 + filename):
        return True
    else:
        return False


def find_power_measurement_properties(filename, profile=False):
    r'''Gets the properties of the power measurement file from its filename.

    :param filename: Name of the power measurement data file.
    :param profile: Wether or not if this a power measurement or a profile measurement.
    :return: Dictionary with the properties of the power data file.
    '''
    properties = []

    filename = filename[:filename.index('.')]

    # Date
    properties.append(filename[-14:-6])

    # Time
    properties.append(filename[-6:-2])

    # Beam
    if not profile:
        properties.append(filename[-21: -19])
    else:
        properties.append(filename[8: 9])

    # Signal
    properties.append(filename[0: -22])

    # Cavity
    properties.append(filename[-22: -21])

    return properties


def sort_acquisitons(filenames):
    r'''Sorts a list of power measurement files based on the cavity number.

    :param filenames: List of power measurement data filenames
    :return: Same list but sorted
    '''
    cav_numbers = np.zeros(len(filenames))
    for i in range(len(filenames)):
        prop_i = find_power_measurement_properties(filenames[i])
        cav_numbers[i] = int(prop_i[-1])

    i = 0
    while i < len(filenames) - 1:
        min_ind = i
        j = i + 1

        while j < len(filenames):
            if cav_numbers[j] < cav_numbers[min_ind]:
                min_ind = j

            j += 1

        cav_numbers[min_ind], cav_numbers[i] = cav_numbers[i], cav_numbers[min_ind]
        filenames[min_ind], filenames[i] = filenames[i], filenames[min_ind]
        i += 1

    return filenames


