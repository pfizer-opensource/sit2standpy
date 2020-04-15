"""
Methods for sit-to-stand trnasfer detection, using previously processed data

Lukas Adamowicz
2019-2020
Pfizer
"""

from numpy import ceil, log2, abs, where, diff, sum, insert, append, arange, sign, median, sqrt, array, mean
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.integrate import cumtrapz
from scipy.fftpack import fft

from sit2standpy.v2.utility import get_stillness
from sit2standpy.v2.base import _BaseProcess, DATA, PROC


__all__ = ['Detector']


class Detector(_BaseProcess):
    def __init__(self, stillness_constraint=True, gravity=9.81, thresholds=None, gravity_pass_order=4,
                 gravity_pass_cutoff=0.8, long_still=0.5, moving_window=0.3, duration_factor=10,
                 displacement_factor=0.75, **kwargs):
        """
        Method for detecting sit-to-stand transitions based on a series of heuristic signal processing rules.

        Parameters
        ----------
        stillness_constraint : bool, optional
            Whether or not to impose the stillness constraint on the detected transitions. Default is True.
        gravity : float, optional
            Value of gravitational acceleration measured by the accelerometer when still. Default is 9.81 m/s^2.
        thresholds : dict, optional
            A dictionary of thresholds to change for stillness detection and transition verification. See *Notes* for
            default values. Only values present will be used over the defaults.
        gravity_pass_order : int, optional
            Low-pass filter order for estimating the direction of gravity by low-pass filtering the raw acceleration.
            Default is 4.
        gravity_pass_cutoff : float, optional
            Low-pass filter frequency cutoff for estimating the direction of gravity. Default is 0.8Hz.
        long_still : float, optional
            Length of time of stillness for it to be considered a long period of stillness. Used to determine the
            integration window limits when available. Default is 0.5s
        moving_window : float, optional
            Length of the moving window for calculating the moving statistics for determining stillness.
            Default is 0.3s.
        duration_factor : float, optional
            The factor for the maximum different between the duration before and after the generalized location of the
            sit to stand as located by the `power_peaks` in the `AccelerationFilter`. Lower factors results in more
            equal time before and after the detected time point. Default is 10, which effectively removes this
            constraint.
        displacement_factor : float, optional
            Factor multiplied by the median of the vertical displacements for all transitions to determine the threshold
            for checking if a transition is a partial or full transition. Default is 0.75

        Notes
        -----
        `stillness_constraint` determines whether or not a sit-to-stand transition is required to start and the
        end of a still period in the data. This constraint is suggested for at-home data. For processing clinic data,
        it is suggested to set this to `False`, especially if processing a task where sit-to-stands are repeated in
        rapid succession.

        Default thresholds:
            - stand displacement: 0.125
            - transition velocity: 0.2
            - accel moving avg: 0.2
            - accel moving std: 0.1
            - jerk moving avg: 2.5
            - jerk moving std: 3

        """
        super().__init__(**kwargs)
        # set the default thresholds
        self._default_thresholds = {
            'stand displacement': 0.125,
            'transition velocity': 0.2,
            'accel moving avg': 0.2,
            'accel moving std': 0.1,
            'jerk moving avg': 2.5,
            'jerk moving std': 3
        }
        # assign attributes
        self.stillness_constraint = stillness_constraint
        self.grav = gravity

        self.thresh = {i: self._default_thresholds[i] for i in self._default_thresholds}
        if thresholds is not None:
            self.thresh.update(thresholds)

        self.grav_ord = gravity_pass_order
        self.grav_cut = gravity_pass_cutoff
        self.long_still = long_still
        self.mov_window = moving_window
        self.dur_factor = duration_factor
        self.disp_factor = displacement_factor

    def _call(self):
        days = [i for i in self.data['Processed']['Sit2Stand'] if 'Day' in i]
        # compute the sampling frequency if necessary
        if 'dt' in self.data['Sensors']['Lumbar']:
            dt = self.data['Sensors']['Lumbar']['dt'][()]
        else:
            dt = mean(diff(self.data['Sensors']['Lumbar']['Unix Time'][:100]))
            self.data = ('Sensors/Lumbar/dt', dt)  # save for future use

        feats = ['STS Times', 'Duration', 'Vertical Displacement', 'Max. Accel.', 'Min. Accel.', 'SPARC']

        for day in days:
            # allocate lists for transition parameters
            sts = {i: [] for i in feats}

            # find stillness
            try:
                start, stop = self.data['Processed']['Sit2Stand'][day]['Indices']
            except KeyError:
                start, stop = 0, self.data['Sensors']['Lumbar']['Accelerometer'].shape[0]

            # shorthand names
            time = self.data['Sensors']['Lumbar']['Unix Time'][start:stop]
            raw_acc = self.data['Sensors']['Lumbar']['Accelerometer'][start:stop]
            filt_acc = self.data['Processed']['Sit2Stand'][day]['Filtered Acceleration'][()]

            # process
            still, starts, stops = get_stillness(filt_acc, dt, self.mov_window, self.grav, self.thresh)
            still_dt = (stops - starts) * dt  # duration in seconds of still periods
            lstill_starts = starts[still_dt > self.long_still]
            lstill_stops = stops[still_dt > self.long_still]

            # compute an estimate of the direction of gravity, assumed to be the vertical direction
            sos = butter(self.grav_ord, 2 * self.grav_cut * dt, btype='low', output='sos')
            vert = sosfiltfilt(sos, raw_acc, axis=0, padlen=None)
            vert /= norm(vert, axis=1, keepdims=True)  # make into unit vectors

            # get an estimate of the vertical acceleration
            v_acc = sum(vert * raw_acc, axis=1)

            # iterate over the power peaks (potential sts time points)
            prev_int_start = -1
            prev_int_end = -1

            for ppk in self.data['Processed']['Sit2Stand'][day]['Power Peaks']:
                try:  # look for the preceding end of stillness
                    end_still = self._get_end_still(time, stops, lstill_stops, ppk)
                except IndexError:
                    continue
                try:  # look for the next start of stillness
                    start_still, still_at_end = self._get_start_still(time, starts, lstill_starts, ppk)
                except IndexError:
                    start_still = int(ppk + (5 / dt))  # try to use a set time after the peak
                    still_at_end = False

                # INTEGRATE between the determined indices
                if end_still < prev_int_start or start_still > prev_int_end:
                    v_vel, v_pos = self._integrate(v_acc[end_still:start_still], dt, still_at_end)

                    # save integration region limits - avoid extra processing if possible
                    prev_int_start = end_still
                    prev_int_end = start_still

                    # get zero crossings
                    pos_zc = insert(where(diff(sign(v_vel)) > 0)[0], 0, 0) + end_still
                    neg_zc = append(where(diff(sign(v_vel)) < 0)[0], v_vel.size - 1) + end_still

                # make sure the velocity is high enough to indicate a peak
                if v_vel[ppk - prev_int_start] < self.thresh['transition velocity']:  # index gets velocity STS
                    continue

                if self.stillness_constraint:
                    sts_start = end_still
                else:
                    try:  # look for the previous postive zero crossing as the start of the transition
                        sts_start = pos_zc[pos_zc < ppk][-1]
                        p_still = stops[stops < ppk][-1]
                        # possibly use the end of stillness if it is close enough to the ZC
                        if -0.5 < (dt * (p_still - sts_start)) < 0.7:
                            sts_start = p_still
                    except IndexError:
                        continue
                # transition end
                try:
                    sts_end = neg_zc[neg_zc > ppk][0]
                except IndexError:
                    continue

                # QUALITY CHECKS
                # --------------
                if (time[sts_end] - time[sts_start]) > 4.5:  # threshold from various lit
                    continue
                if (time[ppk] - time[sts_start]) > (self.dur_factor * (time[sts_end] - time[ppk])):
                    continue

                t_start_i = sts_start - prev_int_start  # integrated value start index
                t_end_i = sts_end - prev_int_start  # integrated value end index
                if t_start_i == t_end_i:
                    continue
                if (v_pos[t_end_i] - v_pos[t_start_i]) < self.thresh['stand displacement']:
                    continue

                # sit to stand assignment
                if len(sts['STS Times']) == 0:
                    # compute sit-to-stand parameters/features
                    dur_ = time[sts_end] - time[sts_start]
                    mx_ = filt_acc[sts_start:sts_end].max()
                    mn_ = filt_acc[sts_start:sts_end].min()
                    vdisp_ = v_pos[t_end_i] - v_pos[t_start_i]
                    sal_, *_ = self.sparc(norm(raw_acc[sts_start:sts_end], axis=1), 1 / dt)

                    sts['STS Times'].append([time[sts_start], time[sts_end]])
                    sts['Duration'].append(dur_)
                    sts['Vertical Displacement'].append(vdisp_)
                    sts['Max. Accel.'].append(mx_)
                    sts['Min. Accel.'].append(mn_)
                    sts['SPARC'].append(sal_)
                else:
                    if (time[sts_start] - sts['STS Times'][-1][1]) > 0.4:
                        # compute sit-to-stand parameters/features
                        dur_ = time[sts_end] - time[sts_start]
                        mx_ = filt_acc[sts_start:sts_end].max()
                        mn_ = filt_acc[sts_start:sts_end].min()
                        vdisp_ = v_pos[t_end_i] - v_pos[t_start_i]
                        sal_, *_ = self.sparc(norm(raw_acc[sts_start:sts_end], axis=1), 1 / dt)

                        sts['STS Times'].append([time[sts_start], time[sts_end]])
                        sts['Duration'].append(dur_)
                        sts['Vertical Displacement'].append(vdisp_)
                        sts['Max. Accel.'].append(mx_)
                        sts['Min. Accel.'].append(mn_)
                        sts['SPARC'].append(sal_)

            # check to ensure no partial transitions
            sts['Vertical Displacement'] = array(sts['Vertical Displacement'])
            partial = sts['Vertical Displacement'] < (self.disp_factor * median(sts['Vertical Displacement']))

            # sts['STS Times'] = array(sts['STS Times'])[~partial]
            # sts['Duration'] = array(sts['Duration'])[~partial]
            # sts['Vertical Displacement'] = sts['Vertical Displacement'][~partial]
            # sts['Max. Accel.'] = array(sts['Max. Accel.'])[~partial]
            # sts['Min. Accel.'] = array(sts['Min. Accel.'])[~partial]
            # sts['SPARC'] = array(sts['SPARC'])[~partial]

            mtd = 'Stillness' if self.stillness_constraint else 'Displacement'
            key = 'Processed/Sit2Stand/{day}/{method} Method/{param}'
            for feat in sts:
                self.data = (key.format(day=day, method=mtd, param=feat), array(sts[feat])[~partial])

    def _integrate(self, vert_accel, dt, still_at_end):
        """
        Double integrate the acceleration along 1 axis to get velocity and position

        Parameters
        ----------
        vert_accel : numpy.ndarray
            (N, ) array of acceleration values to integrate
        dt : float
            Sampling time in seconds
        still_at_end : bool
            Whether or not the acceleration provided ends with a still period. Determines drift mitigation strategy.

        Returns
        -------
        vert_vel : numpy.ndarray
            (N, ) array of vertical velocity
        vert_pos : numpy.ndarray
            (N, ) array of vertical positions
        """
        x = arange(vert_accel.size)

        # integrate and drift mitigate
        if not still_at_end:
            """
            Old/testing:
            fc = butter(1, [2 * 0.1 * dt, 2 * 5 * dt], btype='band')
            vel = cumtrapz(filtfilt(fc[0], fc[1], accel), dx=dt, initial=0)
            """
            vel = detrend(cumtrapz(vert_accel, dx=dt, initial=0))
            if abs(vel[0]) > 0.05:  # it too far away from 0
                vel -= vel[0]  # reset the beginning back to 0, the integration always starts with stillness
        else:
            vel_dr = cumtrapz(vert_accel, dx=dt, initial=0)
            vel = vel_dr - (((vel_dr[-1] - vel_dr[0]) / (x[-1] - x[0])) * x)  # no intercept

        # integrate the velocity to get position
        pos = cumtrapz(vel, dx=dt, initial=0)

        return vel, pos

    def _get_end_still(self, time, still_stops, lstill_stops, peak):
        if self.stillness_constraint:
            end_still = still_stops[still_stops < peak][-1]
            if (time[peak] - time[end_still]) > 2:
                raise IndexError
        else:
            end_still = lstill_stops[lstill_stops < peak][-1]
            if (time[peak] - time[end_still]) > 30:  # don't want to integrate too far out
                raise IndexError
        return end_still

    def _get_start_still(self, time, still_starts, lstill_starts, peak):
        still_at_end = False
        start_still = lstill_starts[lstill_starts > peak][0]
        if (time[start_still] - time[peak]) < 30:
            still_at_end = True
        else:
            raise IndexError
        return start_still, still_at_end

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
        nfft = int(2**(ceil(log2(len(x))) + padlevel))

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
        fc_inx = arange(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        # calculate the arc length
        sal = -sum(sqrt((diff(f_sel) / (f_sel[-1] - f_sel[0]))**2 + diff(Mf_sel)**2))

        return sal, (f, Mf), (f_sel, Mf_sel)



