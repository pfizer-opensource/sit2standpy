"""
Acceleration filtering and preprocessing
"""
import pywt
from numpy import mean, diff, around, arange, sum, std, ascontiguousarray
from numpy.linalg import norm
from scipy.signal import butter, sosfiltfilt, find_peaks
from warnings import warn

from sit2standpy.v2.base import _BaseProcess, PROC, DATA
from sit2standpy.v2.utility import mov_stats


__all__ = ['AccelerationFilter']


class AccelerationFilter(_BaseProcess):
    def __init__(self, continuous_wavelet='gaus1', power_band=None, power_peak_kw=None, power_std_height=True,
                 power_std_trim=0, reconstruction_method='moving average', lowpass_order=4, lowpass_cutoff=5,
                 window=0.25, discrete_wavelet='dmey', extension_mode='constant', reconstruction_level=1, **kwargs):
        """
        Filter acceleration and located potential sit-to-stand time points.

        Parameters
        ----------
        continuous_wavelet : str, optional
            Continuous wavelet to use for signal deconstruction. Default is 'gaus1'. CWT coefficients will be summed
            in the frequency range defined by `power_band`
        power_band : {array_like, int, float}, optional
            Frequency band in which to sum the CWT coefficients. Either an array_like of length 2, with the lower and
            upper limits, or a number, which will be taken as the upper limit, and the lower limit will be set to 0.
            Default is [0, 0.5].
        power_peak_kw : {None, dict}, optional
            Extra key-word arguments to pass to `scipy.signal.find_peaks` when finding peaks in the
            summed CWT coefficient power band data. Default is None, which will use the default parameters except
            setting minimum height to 90, unless `power_std_height` is True.
        power_std_height : bool, optional
            Use the standard deviation of the power for peak finding. Default is True. If True, the standard deviation
            height will overwrite the `height` setting in `power_peak_kw`.
        power_std_trim : float, int, optional
            Number of seconds to trim off the start and end of the power signal before computing the standard deviation
            for `power_std_height`. Default is 0s, which will not trim anything. Suggested value of trimming is 0.5s.
        reconstruction_method : {'moving average', 'dwt'}, optional
            Method for computing the reconstructed acceleration. Default is 'moving average', which takes the moving
            average over the specified window. Other option is 'dwt', which uses the discrete wavelet transform to
            deconstruct and reconstruct the signal while filtering noise out.
        lowpass_order : int, optional
            Initial low-pass filtering order. Default is 4.
        lowpass_cutoff : float, optional
            Initial low-pass filtering cuttoff, in Hz. Default is 5Hz.
        window : float, optional
            Window to use for moving average, in seconds. Default is 0.25s. Ignored if reconstruction_method is 'dwt'.
        discrete_wavelet : str, optional
            Discrete wavelet to use if reconstruction_method is 'dwt'. Default is 'dmey'. See
            `pywt.wavelist(kind='discrete')` for a complete list of options. Ignored if reconstruction_method is
            'moving average'.
        extension_mode : str, optional
            Signal extension mode to use in the DWT de- and re-construction of the signal. Default is 'constant', see
            pywt.Modes.modes for a list of options. Ignored if reconstruction_method is 'moving average'.
        reconstruction_level : int, optional
            Reconstruction level of the DWT processed signal. Default is 1. Ignored if reconstruction_method is
            'moving average'.

        Notes
        -----
        The default height threshold of 90 in `power_peak_kw` was determined on data sampled at 128Hz, and would likely
        need to be adjusted for different sampling frequencies. Especially if using a different sampling frequency,
        use of `power_std_height=True` is recommended.
        """
        super().__init__(**kwargs)

        self.cwave = continuous_wavelet

        if power_band is None:
            self.power_start_f = 0
            self.power_end_f = 0.5
        elif isinstance(power_band, (int, float)):
            self.power_start_f = 0
            self.power_end_f = power_band
        else:
            self.power_start_f, self.power_end_f = power_band

        self.std_height = power_std_height
        self.std_trim = power_std_trim

        if power_peak_kw is None:
            self.power_peak_kw = {'height': 90}
        else:
            self.power_peak_kw = power_peak_kw

        self.method = reconstruction_method
        self.lp_ord = lowpass_order
        self.lp_cut = lowpass_cutoff
        self.window = window
        self.dwave = discrete_wavelet
        self.ext_mode = extension_mode
        self.recon_level = reconstruction_level

    def _call(self):

        # compute the sampling frequency if necessary
        if 'dt' in self.data['Sensors']['Lumbar']:
            dt = self.data['Sensors']['Lumbar']['dt'][()]
        else:
            dt = mean(diff(self.data['Sensors']['Lumbar']['Unix Time'][:100]))
            self.data = ('Sensors/Lumbar/dt', dt)  # save for future use
        # set-up the filter that will be used
        sos = butter(self.lp_ord, 2 * self.lp_cut * dt, btype='low', output='sos')

        if 'Processed' in self.data:
            days = [i for i in self.data['Processed']['Sit2Stand'].keys() if 'Day' in i]
        else:
            days = ['Day 1']
        for iday, day in enumerate(days):
            try:
                start, stop = self.data['Processed']['Sit2Stand'][day]['Indices']
            except KeyError:
                start, stop = 0, self.data['Sensors']['Lumbar']['Accelerometer'].shape[0]
            # compute the magnitude of the acceleration
            m_acc = norm(self.data['Sensors']['Lumbar']['Accelerometer'][start:stop], axis=1)

            f_acc = ascontiguousarray(sosfiltfilt(sos, m_acc))

            # reconstructed acceleration
            if self.method == 'dwt':
                # deconstruct the filtered acceleration magnitude
                coefs = pywt.wavedec(f_acc, self.dwave, mode=self.ext_mode)

                # set all but the desired level of coefficients to be 0s
                if (len(coefs) - self.recon_level) < 1:
                    warn(f'Chosen reconstruction level is too high, setting to {len(coefs) - 1}', UserWarning)
                    ind = 1
                else:
                    ind = len(coefs) - self.recon_level

                for i in range(1, len(coefs)):
                    if i != ind:
                        coefs[i][:] = 0
                r_acc = pywt.waverec(coefs, self.dwave, mode=self.ext_mode)
            elif self.method == 'moving average':
                n_window = int(around(self.window / dt))
                r_acc, *_ = mov_stats(f_acc, n_window)

            # CWT power peak detection
            coefs, freqs = pywt.cwt(r_acc, arange(1, 65), self.cwave, sampling_period=dt)

            # sum the coefficients over the frequencies in the power band
            f_mask = (freqs <= self.power_end_f) & (freqs >= self.power_start_f)
            power = sum(coefs[f_mask, :], axis=0)

            # find the peaks in the power data
            if self.std_height:
                if self.std_trim != 0:
                    trim = int(self.std_trim / dt)
                    self.power_peak_kw['height'] = std(power[trim:-trim], ddof=1)
                else:
                    self.power_peak_kw['height'] = std(power, ddof=1)

            power_peaks, _ = find_peaks(power, **self.power_peak_kw)

            self.data = (PROC.format(day_n=iday+1, value='Filtered Acceleration'), f_acc)
            self.data = (PROC.format(day_n=iday+1, value='Reconstructed Acceleration'), r_acc[:m_acc.size])
            self.data = (PROC.format(day_n=iday+1, value='Power'), power)
            self.data = (PROC.format(day_n=iday+1, value='Power Peaks'), power_peaks)


