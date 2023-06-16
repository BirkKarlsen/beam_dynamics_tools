r'''
Diagnostics function object to get data from simulations.

Author: Birk Emil Karlsen-Baeck
'''


class Diagnostics(object):
    r'''Object for diagnostics of both the beam and the RF system in simulations.'''

    def __init__(self, RingAndRFTracker, Profile, TotalInducedVoltage, CavityLoop, Ring, save_to, get_from,
                 n_bunches, dt_cont=1, dt_beam=1000, dt_cl=1000, dt_prfl=500, dt_ld=25):

        self.turn = 0

        self.tracker = RingAndRFTracker
        self.profile = Profile
        self.induced_voltage = TotalInducedVoltage
        self.cl = CavityLoop
        self.ring = Ring

        self.save_to = save_to
        self.get_from = get_from
        self.n_bunches = n_bunches

        # time interval between difference simulation measurements
        self.dt_cont = dt_cont
        self.ind_cont = 0
        self.n_cont = int(self.tracker.rf_params.n_turns / self.dt_cont)
        self.dt_beam = dt_beam
        self.dt_cl = dt_cl
        self.dt_prfl = dt_prfl

        self.dt_ld = dt_ld
        self.ind_ld = 0
        self.n_ld = int(self.tracker.rf_params.n_turns / self.dt_ld)

        self.perform_measurements = getattr(self, 'empty_measurement')

    def track(self):
        r'''Track attribute to perform measurement setting.'''
        self.reposition_profile_edges()
        self.perform_measurements()
        self.turn += 1

    def reposition_profile_edges(self):
        r'''Function to reposition profile cuts'''
        if self.turn % self.dt_prfl == 0:
            # Modify cuts of the Beam Profile
            self.tracker.beam.statistics()
            self.profile.cut_options.track_cuts(self.tracker.beam)
            self.profile.set_slices_parameters()

    def empty_measurement(self):
        r'''Dummy measurement for simulations not needing output.'''
        pass
