"""
Wavelet based methods of detecting postural transitions

Lukas Adamowicz
June 2019
"""
from numpy import mean, diff, arange, logical_and, sum as npsum, std, timedelta64, where, insert, append, array_split
from scipy.signal import find_peaks
from pandas import to_datetime
import pywt
from multiprocessing import cpu_count, Pool

from sit2standpy.processing import process_timestamps, AccelerationFilter
from sit2standpy import detectors
from sit2standpy.quantify import TransitionQuantifier

'''
class __AutoSit2Stand:
    """
    Automatically run the sit-2-stand analysis on a sample of data. Data windowing will be done automatically if
    necessary based on the provided parameters

    Parameters
    ----------
    acceleration : numpy.ndarray
        (N, 3) array of accelerations measured by a lumbar mounted accelerometer. Units of m/s^2.
    timestamps : numpy.ndarray
        (N, ) array of timestamps.
    time_units : str, optional
        Units of the timestamps. Options are those for converting to pandas.datetimes, ('ns', 'us', 'ms', etc), or
        'datetime' if the timestamps are already pandas.datetime64. Default is 'us' (microseconds).
    window : bool, optional
        Window the provided data into parts of days. Default is True
    hours : tuple, optional
        Tuple of the hours to use to window the data. The indices define the start and stop time of the window during
        the day, ex ('00:00', '24:00') is the whole day. Default is ('08:00', '20:00').
    parallel : bool, optional
        Use parallel processing. Ignored if `window` is False. Default is False.
    parallel_cpu : {'max', int}, optional
        Number of CPUs to use for parallel processing. Ignored if parallel is False. 'max' uses the maximum number
        of CPUs available on the machine, or provide a number less than the maximum to use. Default is 'max'.

    Sit to Stand Detection Parameters
    ---------------------------------
    continuous_wavelet : str, optional
        Continuous wavelet to use for signal deconstruction. Default is 'gaus1'. CWT coefficients will be summed
        in frequency bands that will be used for detecting approximate STS locations.
    peak_pwr_band : {array_like, int, float}, optional
        Frequency band in which to sum the CWT coefficients. Either an array_like of length 2, with the lower and
        upper limits, or a number, which will be taken as the upper limit, and the lower limit will be set to 0.
        Default is [0, 0.5].
    peak_pwr_par : {None, dict}, optional
        Extra parameters (key-word arguments) to pass to scipy.signal.find_peaks when finding peaks in the
        summed CWT coefficient power band data. Default is None, which will use the default parameters, unless
        std_height is True.
    std_height : bool, optional
        Use the standard deviation of the power for peak finding. Default is True.

    Attributes
    ----------
    s2s : Sit2Stand
        The Sit2Stand object framework for detecting sit-to-stand transitions
    days : list
        List of tuples of the indices corresponding to the different days as determined by hours and if data has been
        windowed
    abs_time : pandas.DatetimeIndex
        DatetimeIndex, converted from the provided timestamps
    """
    def __init__(self, acceleration, timestamps, time_units='us', window=True, hours=('08:00', '20:00'), parallel=False,
                 parallel_cpu='max', continuous_wavelet='gaus1', peak_pwr_band=[0, 0.5], peak_pwr_par=None,
                 std_height=True, verbose=True):
        self.verbose = verbose

        if parallel_cpu == 'max':
            self.n_cpu = cpu_count()
        elif parallel_cpu > cpu_count():
            self.n_cpu = cpu_count()
        else:
            self.n_cpu = parallel_cpu

        if not window:
            self.parallel = False
        else:
            self.parallel = parallel

        if time_units is not 'datetime':
            if self.verbose:
                print('Converting timestamps to datetimes...\n')

            if self.parallel:
                pool = Pool(self.n_cpu)
                times = array_split(timestamps, self.n_cpu)

                other_args = ('raise', False, False, None, True, None, True, time_units)
                result = pool.starmap(to_datetime, [(t, ) + other_args for t in times])
                self.abs_time = result[0].append(result[1:])
                pool.close()
            else:
                self.abs_time = to_datetime(timestamps, unit=time_units)
        else:
            self.abs_time = timestamps

        if window:
            if self.verbose:
                print('Setting up windows...\n')
            days_inds = self.abs_time.indexer_between_time(hours[0], hours[1])

            day_ends = days_inds[where(diff(days_inds) > 1)[0]]
            day_starts = days_inds[where(diff(days_inds) > 1)[0] + 1]

            self.days = []
            if day_ends.size == 0 and day_starts == 0:
                self.days.append(days_inds)
            else:
                if day_ends[0] < day_starts[0]:
                    day_starts = insert(day_starts, 0, 0)
                if day_starts[-1] > day_ends[-1]:
                    day_ends = append(day_ends, self.abs_time.size - 1)

                for start, end in zip(day_starts, day_ends):
                    self.days.append(range(start, end))
        else:
            self.days = [range(0, self.abs_time.size)]

        self.accel = acceleration

        # initialize the sit2stand detection object
        if parallel:
            self.s2s = [Sit2Stand(continuous_wavelet=continuous_wavelet, peak_pwr_band=peak_pwr_band,
                             peak_pwr_par=peak_pwr_par, std_height=std_height) for i in range(len(self.days))]
        else:
            self.s2s = Sit2Stand(continuous_wavelet=continuous_wavelet, peak_pwr_band=peak_pwr_band,
                                 peak_pwr_par=peak_pwr_par, std_height=std_height)
        if self.verbose:
            print('Initialization Done!\n')

    def run(self, acc_filter_kwargs=None, detector='stillness', detector_kwargs=None):
        """
        Run the sit to stand detection

        Parameters
        ----------
        acc_filter_kwargs : {None, dict}, optional
            AccFilter key-word arguments. See Notes for default values. See `sit2standpy.AccFilter` for description
            of the parameters
        detector : {'stillness', 'displacement'}
            Detector method to use. Default is 'stillness'
        detector_kwargs : {None, dict}, optional
            Detector method key-word arguments. See Notes for the default values, and `sit2standpy.detectors` for the
            parameters of the chosen detector.

        Returns
        -------
        sts : dict
            Dictionary of sit2standpy.Transition objects containing information about a individual sit-to-stand
            transition. Keys for the dictionary are string timestamps of the start of the transition.

        Attributes
        ----------
        acc_filter : AccFilter
            The AccFilter object
        self.detector : {detectors.Stillness, detectors.Displacement}
            The detector object as determined by the choice in `detector`

        Notes
        -----
        AccFilter default parameters
            - reconstruction_method='moving average'
            - lowpass_order=4
            - lowpass_cutoff=5
            - window=0.25,
            - discrete_wavelet='dmey'
            - extension_mode='constant'
            - reconstruction_level=1
        Detector methods default parameters
            - gravity=9.81
            - thresholds=None
            - gravity_pass_ord=4
            - gravity_pass_cut=0.8
            - long_still=0.5,
            - moving_window=0.3
            - duration_factor=10
            - displacement_factor=0.75
            - lmax_kwargs=None
            - lmin_kwargs=None
            - trans_quant=TransitionQuantifier()
        """
        if self.verbose:
            print('Setting up filters and detector...\n')

        if acc_filter_kwargs is None:
            acc_filter_kwargs = {}
        if detector_kwargs is None:
            detector_kwargs = {}

        self.acc_filter = AccFilter(**acc_filter_kwargs)

        if detector == 'stillness':
            self.detector = detectors.Stillness(**detector_kwargs)
        elif detector == 'displacement':
            self.detector = detectors.Displacement(**detector_kwargs)
        else:
            raise ValueError(f"detector '{detector}' not recognized.")

        if self.parallel:
            if self.verbose:
                print('Processing in parallel...\n')
            pool = Pool(min(self.n_cpu, len(self.days)))

            tmp = [pool.apply_async(self.s2s[i].fit, args=(self.accel[day], self.abs_time[day], self.detector,
                                                           self.acc_filter)) for i, day in enumerate(self.days)]
            results = [p.get() for p in tmp]

            pool.close()

        else:
            if self.verbose:
                print('Processing...\n')
            results = self.s2s.fit(self.accel, self.abs_time, self.detector, self.acc_filter)

        if self.verbose:
            print('Done!\n')
        return results
'''


class Sit2Stand:
    def __init__(self, method='stillness', gravity=9.81, thresholds=None, gravity_order=4, gravity_cut=0.8,
                 long_still=0.5, still_window=0.3, duration_factor=10, displacement_factor=0.75, lmax_kwargs=None,
                 lmin_kwargs=None, transition_quantifier=TransitionQuantifier(), window=False,
                 hours=('08:00', '20:00'), continuous_wavelet='gaus1', power_band=[0, 0.5], power_peak_kwargs=None,
                 power_stdev_height=True,  reconstruction_method='moving average', lowpass_order=4, lowpass_cutoff=5,
                 filter_window=0.25, discrete_wavelet='dmey', extension_mode='constant', reconstruction_level=1):
        """
        Class for storing information and parameters for the detection of sit-to-stand transitions, and extracting
        features to assess performance.

        Parameters
        ----------
        method : {'stillness', 'displacement'}, optional
            Method to use for detection, based on how strict the requirement for stillness before a transition is.
            `stillness` requires that stillness preceeds a transition, whereas `displacement` only uses stillness
            if possible.
        gravity : float, optional
            Gravitational acceleration. Default is 9.81 m/s^2.
        thresholds : {None, dict}, optional
            Thresholds for sit-to-stand detection. Default is None, which uses the default values. See Notes.
        gravity_order : int, optional
            Lowpass filter order for estimation of the direction of gravity. Default is 4.
        gravity_cut : int, optional
            Lowpass filter cutoff frequency for estimation of the direction of gravity. Default is 0.8Hz.
        long_still : float, optional
            Length of stillness required to determine a period of long stillness. Default is 0.5s
        still_window : float, optional
            Window, in seconds, for the determination of stillness. Default is 0.3.
        duration_factor : float, optional
            Factor for the duration of transitions. Larger values discard less transitions.
        displacement_factor : float, optional
            Factor multiplied by the median vertical displacement to determin the minimum vertical displacement
            for a valid transition. Default is 0.75
        lmax_kwargs : {None, dict}, optional
            scipy.signal.find_peaks key-word arguments for detection of local maxima. Default is None.
        lmin_kwargs : {None, dict}, optional
            scipy.signal.find_peaks key-word arguments for detection of local minima. Default is None.
        transition_quantifier : pysit2stand.TransitionQuantifier
            Class for quantifing the transition.
        window : bool, optional
            Window the data based days, with each day a separate window. Default is False.
        hours : array_like, optional
            Hours to use from each day. Default is from 08:00 to 20:00 (tuple('08:00', '20:00')). Ignored if
            `window` is False.
        continuous_wavelet : str, optional
            Continuous wavelet to use for the CWT used in deconstructing the acceleration signal to look for
            STS locations. Default is 'gaus1'.
        power_band : array_like, optional
            Power band to sum, which gives the peaks and locations of possible sit-to-stand transitions. Default is
            [0, 0.5]
        power_peak_kwargs : {None, dict}, optional
            scipy.signal.find_peaks additional key-word arguments to use when finding peaks in the CWT power band.
            Default is None, though using a distance equal to the number of samples in 1 second is recommended.
        power_stdev_height : bool, optional
            Whether or not to use the standard deviation of the power signal as the minimum peak height. Default is
            True.
        reconstruction_method : {'moving average', 'dwt'}, optional
            Reconstruction method to use for the reconstructed acceleration. Default is `moving average`.
        lowpass_order : int, optional
            Initial low-pass filtering order. Default is 4.
        lowpass_cutoff : float, optional
            Initial low-pass filtering cuttoff, in Hz. Default is 5Hz.
        filter_window : float, optional
            Window to use for moving average, in seconds. Default is 0.25s. Ignored if reconstruction_method is 'dwt'.
        discrete_wavelet : str, optional
            Discrete wavelet to use if reconstruction_method is 'dwt'. Default is 'dmey'. See
            pywt.wavelist(kind='discrete') for a complete list of options. Ignored if reconstruction_method is
            'moving average'.
        extension_mode : str, optional
            Signal extension mode to use in the DWT de- and re-construction of the signal. Default is 'constant', see
            pywt.Modes.modes for a list of options. Ignored if reconstruction_method is 'moving average'.
        reconstruction_level : int, optional
            Reconstruction level of the DWT processed signal. Default is 1. Ignored if reconstruction_method is
            'moving average'.

        Attributes
        ----------

        References
        ----------
        L. Adamowicz et al. "Sit-to-Stand Detection Using Only Lumbar Acceleration: Clinical and Home Application."
        Journal of Biomedical and Health Informatics. 2020.
        """
        self._method = method
        self._grav = gravity
        self._ths = thresholds
        self._grav_ord = gravity_order
        self._grav_cut = gravity_cut
        self._long_still = long_still
        self._still_window = still_window
        self._duration_factor = duration_factor
        self._disp_factor = displacement_factor
        self._lmax_kw = lmax_kwargs
        self._lmin_kw = lmin_kwargs
        self._tq = transition_quantifier
        self._window = window
        self._hours = hours
        self._cwave = continuous_wavelet
        self._pwr_band = power_band
        self._pwr_pk_kw = power_peak_kwargs
        self._pwr_std_h = power_stdev_height
        self._recon_method = reconstruction_method
        self._lp_order = lowpass_order
        self._lp_cut = lowpass_cutoff
        self._filt_window = filter_window
        self._dwave = discrete_wavelet
        self._ext_mode = extension_mode
        self._recon_level = reconstruction_level

    def apply(self, accel, time, time_units='us', time_conv_kw=None):
        """
        Apply the sit-to-stand detection using the given parameters.

        Parameters
        ----------
        accel : numpy.ndarray
            (N, 3) array of accelerations in m/s^2.
        time : numpy.ndarray
            (N, ) array of time values. If in unix timestamps, will be converted using the set `time_units` and
            `time_conv_kw`.
        time_units : str, optional
            Units of the time provided. Default is 'us' for microseconds unix time.
        time_conv_kw : {None, dict}, optional
            pd.to_datetime additional optional key-word arguments. Default is None.

        Returns
        -------
        sist : dict
            Dictionary of sit-to-stand objects with attributes for the duration, and performance features.
        """
        # convert timestamps and window if chosen
        if not self._window:
            timestamps, dt = process_timestamps(time, accel, time_units=time_units, conv_kw=time_conv_kw,
                                                window=self._window, hours=self._hours)
        else:
            timestamps, dt, acc_win = process_timestamps(time, accel, time_units=time_units, conv_kw=time_conv_kw,
                                                         window=self._window, hours=self._hours)

        # acceleration filter object
        acc_filt = AccelerationFilter(continuous_wavelet=self._cwave, power_band=self._pwr_band,
                                      power_peak_kw=self._pwr_pk_kw, power_std_height=self._pwr_std_h,
                                      reconstruction_method=self._recon_method, lowpass_order=self._lp_order,
                                      lowpass_cutoff=self._lp_cut, window=self._filt_window,
                                      discrete_wavelet=self._dwave, extension_mode=self._ext_mode,
                                      reconstruction_level=self._recon_level)

        if not self._window:
            filt_accel, rec_accel, power, power_peaks = acc_filt.apply(accel, 1 / dt)  # run the filtering
        else:
            filt_accel, rec_accel, power, power_peaks = {}, {}, {}, {}
            for day in acc_win.keys():
                filt_accel[day], rec_accel[day], power[day], power_peaks[day] = acc_filt.apply(acc_win[day], 1 / dt)

        # setup the STS detection
        if self._method == 'stillness':
            detect = detectors.Stillness(gravity=self._grav, thresholds=self._ths, gravity_pass_ord=self._grav_ord,
                                         gravity_pass_cut=self._grav_cut, long_still=self._long_still,
                                         moving_window=self._still_window, duration_factor=self._duration_factor,
                                         displacement_factor=self._disp_factor, lmax_kwargs=self._lmax_kw,
                                         lmin_kwargs=self._lmin_kw, trans_quant=self._tq)
        elif self._method == 'displacement':
            detect = detectors.Displacement(gravity=self._grav, thresholds=self._ths, gravity_pass_ord=self._grav_ord,
                                            gravity_pass_cut=self._grav_cut, long_still=self._long_still,
                                            moving_window=self._still_window, duration_factor=self._duration_factor,
                                            displacement_factor=self._disp_factor, lmax_kwargs=self._lmax_kw,
                                            lmin_kwargs=self._lmin_kw, trans_quant=self._tq)
        else:
            raise ValueError('Method must be set as `stillness` or `displacement`.')

        if not self._window:
            sist = detect.apply(accel, filt_accel, rec_accel, timestamps, dt, power_peaks)
        else:
            sist = {}
            for day in filt_accel.keys():
                day_sist = detect.apply(acc_win[day], filt_accel[day], rec_accel[day], timestamps[day], dt,
                                        power_peaks[day])
                sist.update(day_sist)

        return sist


