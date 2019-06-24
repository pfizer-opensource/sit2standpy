"""
Common methods for both acceleration only and imu-based postural transition detection

Lukas Adamowicz
June 2019
"""
from numpy import around
from numpy.linalg import norm
from scipy.signal import butter, filtfilt
import pywt

from pysit2stand import utility as u_


class Transition:
    def __init__(self, times=None, v_displacement=None, duration=None, max_v_velocity=None, min_v_velocity=None,
                 max_acceleration=None, min_acceleration=None):
        """
        Object for storing information about a postural transition
        """
        self.times = times
        if times is not None:
            self.start_time = times[0]
            self.end_time = times[1]
        else:
            self.start_time = None
            self.end_time = None
        self.v_displacement = v_displacement
        self.duration = duration
        self.max_v_velocity = max_v_velocity
        self.min_v_velocity = min_v_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration


class AccFilter:
    def __init__(self, reconstruction_method='moving average', lowpass_order=4, lowpass_cutoff=5,
                 window=0.25, discrete_wavelet='dmey', extension_mode='constant', reconstruction_level=1):
        """
        Object for filtering and reconstructing raw acceleration data

        Parameters
        ----------
        reconstruction_method : {'moving average', 'dwt'}, optional
            Method for computing the reconstructed acceleration. Default is 'moving average', which takes the moving
            average over the specified window. Other option is 'dwt', which uses the discrete wavelet transform to
            deconstruct and reconstruct the signal while filtering noise out.
        lowpass_order : int, optional
            Initial low-pass filtering order. Default is 4.
        lowpass_cutoff : float, optional
            Initial low-pass filtering cuttoff, in Hz. Default is 5Hz.
        window : float, optional
            Window to use for moving average, in seconds. Default is 0.25s. Ignored if reconstruction_method is 'dwt'
        discrete_wavelet : str, optional
            Discrete wavelet to use if reconstruction_method is 'dwt'. Default is 'dmey'. See
            pywt.wavelist(kind='discrete') for a complete list of options. Ignored if reconstruction_method is
            'moving average'.
        extension_mode : str, optional
            Signal extension mode to use in the DWT de- and re-construction of the signal. Default is 'constant', see
            pywt.Modes.modes for a list of options. Ignored if reconstruction_method is 'moving average'.
        reconstruction_level : int, optional
            Reconstruction level of the DWT processed signal. Default is 1. Ignored if reconstruction_method is
            'moving average'
        """
        self.method = reconstruction_method

        self.lp_ord = lowpass_order
        self.lp_cut = lowpass_cutoff

        self.window = window

        self.dwave = discrete_wavelet
        self.ext_mode = extension_mode
        self.recon_level = reconstruction_level

    def apply(self, accel, fs):
        """
        Apply the desired filtering to the provided signal.

        Parameters
        ----------
        accel : numpy.ndarray
            (N, 3) array of raw acceleration values.
        fs : float, optional
            Sampling frequency for the acceleration data.

        Returns
        -------
        mag_acc_f : numpy.ndarray
            (N, ) array of the filtered (low-pass only) acceleration magnitude.
        mag_acc_r : numpy.ndarray
            (N, ) array of the reconstructed acceleration magnitude. This is either filtered and then moving averaged,
            or filtered, and then passed through the DWT and inverse DWT with more filtering, depending on the
            reconstruction_method specified.
        """
        # compute the acceleration magnitude
        macc = norm(accel, axis=1)

        # setup the filter, and filter the acceleration magnitude
        fc = butter(self.lp_ord, 2 * self.lp_cut / fs, btype='low')
        macc_f = filtfilt(fc[0], fc[1], macc)

        if self.method == 'dwt':
            # deconstruct the filtered acceleration magnitude
            coefs = pywt.wavedec(macc_f, self.dwave, mode=self.ext_mode)

            # set all but the desired level of coefficients to be 0s
            if (len(coefs) - self.recon_level) < 1:
                print(f'Chosen reconstruction level is too high, setting reconstruction level to {len(coefs) - 1}')
                ind = 1
            else:
                ind = len(coefs) - self.recon_level

            for i in range(1, len(coefs)):
                if i != ind:
                    coefs[i][:] = 0

            macc_r = pywt.waverec(coefs, self.dwave, mode=self.ext_mode)
        elif self.method == 'moving average':
            n_window = int(around(fs * self.window))  # compute the length in samples of the moving average
            macc_r, _, _ = u_.mov_stats(macc_f, n_window)  # compute the moving average

        return macc_f, macc_r[:macc_f.size]
