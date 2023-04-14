'''
Functions to get data from directories.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import os
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm


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


def find_file_in_folder(f, fdir):
    file_name = None
    for file in os.listdir(fdir):
        if file.startswith(f):
            file_name = file

    return file_name


def find_files_in_folder_starting_and_ending_with(fdir, prefix=None, suffix=None):
    files = []
    for file in os.listdir(fdir):
        if suffix is None:
            if file.startswith(prefix):
                files.append(file)
        elif prefix is None:
            if file.endswith(suffix):
                files.append(file)
        else:
            if file.startswith(prefix) and file.endswith(suffix):
                files.append(file)

    return files


def remove_filetype_from_name(files, filetype=None):
    new_files = []
    if filetype is not None:
        for i in range(len(files)):
            new_files.append(files[i][:-len(filetype)])
    else:
        for i in range(len(files)):
            ind = files[i].index('.', -4, -1)
            new_files.append(files[i][:ind])

    return new_files



def make_and_write_yaml(fname, fdir, content_dict):
    r'''
    Makes a yaml-file called fname if there does not exist one. and then writes content_dict to that file.

    :param fname: name of yaml-file
    :param fdir: directory the yaml-file is in
    :param content_dict: dictonary with contents for the yaml-file
    '''

    if not os.path.isfile(fdir + fname):
        os.system(f'touch {fdir + fname}')

    with open(fdir + fname, 'w') as file:
        document = yaml.dump(content_dict, file)


def write_to_yaml(fname, fdir, content_dict):
    r'''
    Adds the contents of content_dict to the yaml-file called fname in directory fdir.

    :param fname: name of yaml-file
    :param fdir: directory the yaml-file is in
    :param content_dict: dictonary with contents for the yaml-file
    '''

    if not os.path.isfile(fdir + fname):
        print('Error: File not found')
    else:
        with open(fdir + fname) as file:
            existing_dict = yaml.full_load(file)

        with open(fdir + fname, 'w') as file:
            existing_dict.update(content_dict)
            document = yaml.dump(existing_dict, file)


def sort_sps_profiles(files):
    acq_num = np.zeros(len(files))
    for i in range(len(files)):
        acq_num[i] = int(files[i][-3:])

    i = 0
    while i < len(files) - 1:
        min_ind = i
        j = i + 1

        while j < len(files):
            if acq_num[j] < acq_num[min_ind]:
                min_ind = j

            j += 1

        acq_num[min_ind], acq_num[i] = acq_num[i], acq_num[min_ind]
        files[min_ind], files[i] = files[i], files[min_ind]
        i += 1

    return files


def import_sps_profiles(fdir, files, N_samples_per_file=9999900, prt=False):
    r'''
    Imports profile measurements in the SPS and corrects for gitter.

    :param fdir: directory the acquisition files are in
    :param files: the name of the acquisition files
    :param N_samples_per_file: number of samples per file
    :return: profile data, profile data with gitter correction
    '''
    profile_datas = np.zeros((len(files), N_samples_per_file))
    profile_datas_corr = np.zeros((len(files), N_samples_per_file))
    n = 0
    if prt:
        print(f'Fetching profiles...')

    for f in tqdm(files, disable=not prt):
        profile_datas[n,:] = np.load(fdir + f + '.npy')

        conf_f = open(fdir + f + '.asc', 'r')
        acq_params = conf_f.readlines()
        conf_f.close()

        delta_t = float(acq_params[6][39:-1])
        frame_length = [int(s) for s in acq_params[7].split() if s.isdigit()][0]
        N_frames = [int(s) for s in acq_params[8].split() if s.isdigit()][0]
        trigger_offsets = np.zeros(N_frames, )
        for line in np.arange(19, N_frames + 19):
            trigger_offsets[line - 20] = float(acq_params[line][35:-2])

        timeScale = np.arange(frame_length) * delta_t

        # data = np.load(fullpath)
        data = np.reshape(np.load(fdir + f + '.npy'), (N_frames, frame_length))
        data_corr = np.zeros((N_frames, frame_length))

        for i in range(N_frames):
            x = timeScale + trigger_offsets[i]
            A = interp1d(x, data[i, :], fill_value='extrapolate')
            data_corr[i,:] = A(timeScale)

        profile_datas_corr[n, :] = data_corr.flatten()
        n += 1

    return profile_datas, profile_datas_corr
