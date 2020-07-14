"""
Misc utility functions used in detection/quantification of STS transfers

Lukas Adamowicz
2019-2020
Pfizer
"""

from numpy import zeros, ceil, mean, std, around, gradient, where, diff, insert, append, array, savetxt
from numpy.lib import stride_tricks
import h5py
import udatetime as udt


__all__ = ['tabulate_results']


def tabulate_results(results, csv_path, method='stillness'):
    """
    Tabulate the results as calculated by the sequential pipeline.

    Parameters
    ----------
    results : {dict, str}
        Either a dictionary of the results, or the path to the h5 file where the results were stored.
    csv_path : str
        Path to save the tabular data at
    method : {'stillness', 'displacement'}, optional
        Which method to tabulate results for. Default is 'stillness'.
    """
    # get the results
    days, times, duration, vdisp, mxa, mna, sparc = [], [], [], [], [], [], []
    mtd = f'{method.capitalize()} Method'
    if isinstance(results, dict):
        day_list = [i for i in results['Processed']['Sit2Stand'] if 'Day' in i]

        for day in day_list:
            days.extend([int(day[4:])] * results['Processed']['Sit2Stand'][day][mtd]['STS Times'].shape[0])
            times.extend(results['Processed']['Sit2Stand'][day][mtd]['STS Times'])
            duration.extend(results['Processed']['Sit2Stand'][day][mtd]['Duration'])
            vdisp.extend(results['Processed']['Sit2Stand'][day][mtd]['Vertical Displacement'])
            mxa.extend(results['Processed']['Sit2Stand'][day][mtd]['Max. Accel.'])
            mna.extend(results['Processed']['Sit2Stand'][day][mtd]['Min. Accel.'])
            sparc.extend(results['Processed']['Sit2Stand'][day][mtd]['SPARC'])
    else:
        with h5py.File(results, 'r') as f:
            day_list = [i for i in f['Processed/Sit2Stand'] if 'Day' in i]

            for day in day_list:
                days.extend([int(day[4:])] * f[f'Processed/Sit2Stand/{day}/{mtd}/STS Times'].shape[0])
                times.extend(f[f'Processed/Sit2Stand/{day}/{mtd}/STS Times'])
                duration.extend(f[f'Processed/Sit2Stand/{day}/{mtd}/Duration'])
                vdisp.extend(f[f'Processed/Sit2Stand/{day}/{mtd}/Vertical Displacement'])
                mxa.extend(f[f'Processed/Sit2Stand/{day}/{mtd}/Max. Accel.'])
                mna.extend(f[f'Processed/Sit2Stand/{day}/{mtd}/Min. Accel.'])
                sparc.extend(f[f'Processed/Sit2Stand/{day}/{mtd}/SPARC'])

    table = zeros((len(days), 12), dtype='object')
    table[:, 0] = days
    table[:, 1:3] = array(times)
    table[:, 7] = duration
    # table[:, 8] = vdisp
    table[:, 9] = mxa
    table[:, 10] = mna
    table[:, 11] = sparc

    for i, ts in enumerate(table[:, 1]):
        dt = udt.utcfromtimestamp(ts)
        table[i, 3] = dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        table[i, 4] = dt.hour
        table[i, 5] = dt.minute
        table[i, 6] = dt.weekday() >= 5  # is the day a weekend. 0=Monday, 6=Sunday

    hdr = 'Day,Start Unix Time,End Unix Time,Start Time,Hour,Minute,Weekend,Duration,Vertical Displacement,' \
          'Max. Accel.,Min. Accel., SPARC'
    fmt = '%d, %f, %f, %s, %i, %i, %s, %f, %f, %f, %f, %f'
    savetxt(csv_path, table, header=hdr, fmt=fmt)


def mov_stats(seq, window):
    """
    Compute the centered moving average and standard deviation.

    Parameters
    ----------
    seq : numpy.ndarray
        Data to take the moving average and standard deviation on.
    window : int
        Window size for the moving average/standard deviation.

    Returns
    -------
    m_mn : numpy.ndarray
        Moving average
    m_st : numpy.ndarray
        Moving standard deviation
    pad : int
        Padding at beginning of the moving average and standard deviation
    """

    def rolling_window(x, wind):
        if not x.flags['C_CONTIGUOUS']:
            raise ValueError("Data must be C-contiguous to be able to window for moving statistics")
        shape = x.shape[:-1] + (x.shape[-1] - wind + 1, wind)
        strides = x.strides + (x.strides[-1],)
        return stride_tricks.as_strided(x, shape=shape, strides=strides)

    m_mn = zeros(seq.shape)
    m_st = zeros(seq.shape)

    if window < 2:
        window = 2

    pad = int(ceil(window / 2))

    rw_seq = rolling_window(seq, window)

    n = rw_seq.shape[0]

    m_mn[pad:pad + n] = mean(rw_seq, axis=-1)
    m_st[pad:pad + n] = std(rw_seq, axis=-1, ddof=1)

    m_mn[:pad], m_mn[pad + n:] = m_mn[pad], m_mn[-pad - 1]
    m_st[:pad], m_st[pad + n:] = m_st[pad], m_st[-pad - 1]
    return m_mn, m_st, pad


def get_stillness(filt_accel, dt, window, gravity, thresholds):
    """
    Stillness determination based on filtered acceleration magnitude and jerk magnitude

    Parameters
    ----------
    filt_accel : numpy.ndarray
        1D array of filtered magnitude of acceleration data, units of m/s^2
    dt : float
        Sampling time, in seconds
    window : float
        Moving statistics window length, in seconds
    gravity : float
        Gravitational acceleration, as measured by the sensor during static periods.
    thresholds : dict
        Dictionary of the 4 thresholds to be used - accel moving avg, accel moving std, 
        jerk moving avg, and jerk moving std.  
        Acceleration average thresholds should be for difference from gravitional acceleration.

    Returns
    -------
    still : numpy.ndarray
        (N, ) boolean array of stillness (True)
    starts : numpy.ndarray
        (Q, ) array of indices where stillness starts. Includes index 0 if still[0] is True. Q < (N/2)
    stops : numpy.ndarray
        (Q, ) array of indices where stillness ends. Includes index N-1 if still[-1] is True. Q < (N/2)
    """
    # compute the sample window length from the time value
    n_window = int(around(window / dt))
    # compute the acceleration moving stats
    acc_rm, acc_rsd, _ = mov_stats(filt_accel, n_window)
    # compute the jerk
    jerk = gradient(filt_accel, dt, edge_order=2)
    # compute the jerk moving stats
    jerk_rm, jerk_rsd, _ = mov_stats(jerk, n_window)

    # create the stillness masks
    arm_mask = abs(acc_rm - gravity) < thresholds['accel moving avg']
    arsd_mask = acc_rsd < thresholds['accel moving std']
    jrm_mask = abs(jerk_rm) < thresholds['jerk moving avg']
    jrsd_mask = jerk_rsd < thresholds['jerk moving std']

    still = arm_mask & arsd_mask & jrm_mask & jrsd_mask
    starts = where(diff(still.astype(int)) == 1)[0]
    stops = where(diff(still.astype(int)) == -1)[0]

    if still[0]:
        starts = insert(starts, 0, 0)
    if still[-1]:
        stops = append(stops, len(still) - 1)

    return still, starts, stops