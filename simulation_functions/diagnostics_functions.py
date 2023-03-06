'''
Diagnostics function object to simulations in the SPS and LHC.

Author: Birk Emil Karlsen-Bæck
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LHCDiagnostics(object):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the LHC.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, LHCCavityLoop, save_to,
                 setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = LHCCavityLoop

        self.save_to = save_to

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')


    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.perform_measurements()
        self.turn += 1


    def empty_measurement(self):
        pass


    def standard_measurement(self):
        pass


class SPSDiagnostics(object):
    r'''
    Object for diagnostics of both the beam and the RF system in simulations of the SPS.
    '''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, SPSCavityFeedback, save_to,
                 setting=0, dt_cont=1, dt_beam=1000, dt_cl=1000):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = SPSCavityFeedback

        self.save_to = save_to

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl

        if setting == 0:
            self.perform_measurements = getattr(self, 'standard_measurement')
        else:
            self.perform_measurements = getattr(self, 'empty_measurement')


    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.perform_measurements()
        self.turn += 1


    def empty_measurement(self):
        pass


    def standard_measurement(self):
        pass
