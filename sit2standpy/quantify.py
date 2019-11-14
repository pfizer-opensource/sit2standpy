"""
Objects containing methods for quantifying Sit to stand transitions based on accelerometer data

Lukas Adamowicz
July 2019
Pfizer
"""
from numpy import ceil, log2, arange, abs, sum, sqrt, diff
from numpy.linalg import norm
from scipy.fftpack import fft
from sit2standpy.utility import Transition


class TransitionQuantifier:
    """
    Quantification of a sit-to-stand transition.
    """
    def __init__(self):
        pass

    def quantify(self, times, fs, raw_acc=None, mag_acc_f=None, mag_acc_r=None, v_vel=None, v_pos=None):
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
        if raw_acc is not None:
            acc = norm(raw_acc, axis=1)
            sparc, _, _ = TransitionQuantifier.sparc(acc, fs)
        else:
            sparc = None
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
                                      min_v_velocity=min_v_vel, max_acceleration=max_acc, min_acceleration=min_acc,
                                      sparc=sparc)

        return self.transition_

    @staticmethod
    def sparc(x, fs, padlevel=4, fc=10.0, amp_th=0.05):
        """
        SPectral ARC length metric for quantifying smoothness

        Parameters
        ----------
        x : numpy.ndarray
            Array containing the data to be analyzed for smoothness
        fs : float
            Sampling frequency
        padlevel : int, optional
            Indicates the amount of zero-padding to be done to the movement data for estimating the spectral arc length.
            Default is 4.
        fc : float, optional
            The max cutoff frequency for calculating the spectral arc length metric. Default is 10.0
        amp_th : float, optional
            The amplitude threshold to be used for determining the cut off frequency up to which the spectral arc length
            is to be estimated. Default is 0.05

        Returns
        -------
        sal : float
            The spectral arc length estimate of the given data's smoothness
        (f, Mf) : (numpy.ndarray, numpy.ndarray)
            The frequency and the magnitude spectrum of the input data. This spectral is from 0 to the Nyquist frequency
        (f_sel, Mf_sel) : (numpy.ndarray, numpy.ndarray)
            The portion of the spectrum that is selected for calculating the spectral arc length

        References
        ----------
        S. Balasubramanian, A. Melendez-Calderon, A. Roby-Brami, E. Burdet. "On the analysis of movement smoothness."
        Journal of NeuroEngineering and Rehabilitation. 2015.
        """
        # number of zeros to be padded
        nfft = int(pow(2, ceil(log2(len(x))) + padlevel))

        # frequency
        f = arange(0, fs, fs / nfft)
        # normalized magnitude spectrum
        Mf = abs(fft(x, nfft))
        Mf = Mf / Mf.max()

        # indices to choose only the spectrum withing the given cutoff frequency Fc
        # NOTE: this is a low pass filtering operation to get rid of high frequency noise from affecting the next step
        # (amplitude threshold based cutoff for arc length calculation
        fc_inx = ((f <= fc) * 1).nonzero()
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]

        # choose the amplitude threshold based cutoff frequency. Index of the last point on the magnitude spectrum is
        # greater than or equal to the amplitude threshold
        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        fc_inx = range(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        # calculate the arc length
        sal = -sum(sqrt(pow(diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) + pow(diff(Mf_sel), 2)))

        return sal, (f, Mf), (f_sel, Mf_sel)
