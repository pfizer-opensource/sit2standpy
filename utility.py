"""
Misc utility functions for detecting postural transitions

Lukas Adamowicz
June 2019
"""
from numpy import zeros, ceil, mean, std, sqrt, array, cross, dot, arccos, cos, sin
from numpy.linalg import norm
from numpy.lib import stride_tricks


def mov_stats(seq, window):
    """
    Compute the centered moving average and standard deviation
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
        shape = x.shape[:-1] + (x.shape[-1] - wind + 1, wind)
        strides = x.strides + (x.strides[-1], )
        return stride_tricks.as_strided(x, shape=shape, strides=strides)

    m_mn = zeros(seq.shape)
    m_st = zeros(seq.shape)

    if window < 2:
        window = 2

    pad = int(ceil(window / 2))

    rw_seq = rolling_window(seq, window)

    m_mn[pad:-pad + 1] = mean(rw_seq, axis=-1)
    m_st[pad:-pad + 1] = std(rw_seq, axis=-1, ddof=1)
    """
    # OLD METHOD
    # compute first window stats
    m_mn[pad] = mean(seq[:window], axis=0)
    m_st[pad] = std(seq[:window], axis=0, ddof=1)**2  # ddof of 1 indicates sample standard deviation, no population

    # compute moving mean and standard deviation
    for i in range(1, seq.shape[0] - window):
        diff_fl = seq[window + i - 1] - seq[i - 1]  # difference in first and last elements of sliding window
        m_mn[pad + i] = m_mn[pad + i - 1] + diff_fl / window
        m_st[pad + i] = m_st[pad + i - 1] + (seq[window + i - 1] - m_mn[pad + i] + seq[i - 1] -
                                             m_mn[pad + i - 1]) * diff_fl / (window - 1)

    m_st = sqrt(abs(m_st))  # added absolute value to ensure that any round off error doesn't effect the results
    """
    m_mn[:pad], m_mn[-pad:] = m_mn[pad], m_mn[-pad-1]
    m_st[:pad], m_st[-pad:] = m_st[pad], m_st[-pad-1]
    return m_mn, m_st, pad


def vec2quat(v1, v2):
    """
    Find the rotation quaternion between two vectors. Rotate v1 onto v2
    Parameters
    ----------
    v1 : numpy.ndarray
        Vector 1
    v2 : numpy.ndarray
        Vector 2
    Returns
    -------
    q : numpy.ndarray
        Quaternion representing the rotation from v1 to v2
    """
    angle = arccos(dot(v1.flatten(), v2.flatten()) / (norm(v1) * norm(v2)))

    # Rotation axis is always normal to two vectors
    axis = cross(v1.flatten(), v2.flatten())
    axis = axis / norm(axis)  # normalize

    q = zeros(4)
    q[0] = cos(angle / 2)
    q[1:] = axis * sin(angle / 2)
    q /= norm(q)

    return q


def quat2matrix(q):
    """
    Transform quaternion to rotation matrix
    Parameters
    ----------
    q : numpy.ndarray
        Quaternion
    Returns
    -------
    R : numpy.ndarray
        Rotation matrix
    """
    if q.ndim == 1:
        s = norm(q)
        R = array([[1 - 2 * s * (q[2] ** 2 + q[3] ** 2), 2 * s * (q[1] * q[2] - q[3] * q[0]),
                    2 * s * (q[1] * q[3] + q[2] * q[0])],
                   [2 * s * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * s * (q[1] ** 2 + q[3] ** 2),
                    2 * s * (q[2] * q[3] - q[1] * q[0])],
                   [2 * s * (q[1] * q[3] - q[2] * q[0]), 2 * s * (q[2] * q[3] + q[1] * q[0]),
                    1 - 2 * s * (q[1] ** 2 + q[2] ** 2)]])
    elif q.ndim == 2:
        s = norm(q, axis=1)
        R = array([[1 - 2 * s * (q[:, 2]**2 + q[:, 3]**2), 2 * s * (q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0]),
                    2 * s * (q[:, 1] * q[:, 3] + q[:, 2] * q[:, 0])],
                   [2 * s * (q[:, 1] * q[:, 2] + q[:, 3] * q[:, 0]), 1 - 2 * s * (q[:, 1]**2 + q[:, 3]**2),
                    2 * s * (q[:, 2] * q[:, 3] - q[:, 1] * q[:, 0])],
                   [2 * s * (q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0]), 2 * s * (q[:, 2] * q[:, 3] + q[:, 1] * q[:, 0]),
                    1 - 2 * s * (q[:, 1]**2 + q[:, 2]**2)]])
        R = R.transpose([2, 0, 1])
    return R


# ------------------------------------------------
#     QUATERNION METHODS
# ------------------------------------------------
def quat_mult(q1, q2):
    """
    Multiply quaternions
    Parameters
    ----------
    q1 : numpy.ndarray
        1x4 array representing a quaternion
    q2 : numpy.ndarray
        1x4 array representing a quaternion
    Returns
    -------
    q : numpy.ndarray
        1x4 quaternion product of q1*q2
    """
    if q1.shape != (1, 4) and q1.shape != (4, 1) and q1.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q1 has more or less than 4 elements')
    if q2.shape != (1, 4) and q2.shape != (4, 1) and q2.shape != (4,):
        raise ValueError('Quaternions contain 4 dimensions, q2 has more or less than 4 elements')
    if q1.shape == (4, 1):
        q1 = q1.T

    Q = array([[q2[0], q2[1], q2[2], q2[3]],
               [-q2[1], q2[0], -q2[3], q2[2]],
               [-q2[2], q2[3], q2[0], -q2[1]],
               [-q2[3], -q2[2], q2[1], q2[0]]])

    return q1 @ Q


def quat_conj(q):
    """
    Compute the conjugate of a quaternion
    Parameters
    ----------
    q : numpy.ndarray
        Nx4 array of N quaternions to compute the conjugate of.
    Returns
    -------
    q_conj : numpy.ndarray
        Nx4 array of N quaternion conjugats of q.
    """
    return q * array([1, -1, -1, -1])
