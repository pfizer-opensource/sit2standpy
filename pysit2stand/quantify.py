"""
Objects containing methods for quantifying Sit to stand transitions based on accelerometer data

Lukas Adamowicz
July 2019
Pfizer
"""
from pysit2stand.utility import Transition


class TransitionQuantifier:
    """
    Quantification of a sit-to-stand transition.
    """
    def __init__(self):
        pass

    def quantify(self, times, mag_acc_f=None, mag_acc_r=None, v_vel=None, v_pos=None):
        """
        Compute quantitative values from the provided signals

        Parameters
        ----------
        times : tuple
            Tuple of the start and end timestamps for the transition
        mag_acc_f : {None, numpy.ndarray}, optional
            Filtered acceleration magnitude during the transition.
        mag_acc_r : {None, numpy.ndarray}, optional
            Reconstructed acceleration magnitude during the transition.
        v_vel : {None, numpy.ndarray}, optional
            Vertical velocity during the transition.
        v_pos : {None, numpy.ndarray}, optional
            Vertical position during the transition.

        Returns
        -------
        transition : Transition
            Transition object containing metrics quantifying the transition.
        """
        if mag_acc_f is not None:
            max_acc = mag_acc_f.max()
            min_acc = mag_acc_f.min()
        else:
            max_acc = None
            min_acc = None
        if v_vel is not None:
            max_v_vel = v_vel.max()
            min_v_vel = v_vel.min()
        else:
            max_v_vel = None
            min_v_vel = None
        if v_pos is not None:
            v_disp = v_pos[-1] - v_pos[0]
        else:
            v_disp = None

        self.transition_ = Transition(times=times, v_displacement=v_disp, max_v_velocity=max_v_vel,
                                      min_v_velocity=min_v_vel, max_acceleration=max_acc, min_acceleration=min_acc)

        return self.transition_