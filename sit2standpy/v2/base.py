"""
Functionality for processes in the gait processing pipeline
"""
import h5py


__all__ = ['']

PROC = 'Processed/Sit2Stand/Day {day_n}/{value}'
DATA = 'Sensors/Lumbar/{data}'


class _BaseProcess:
    def __init__(self, sampling_frequency=None):
        """
        General class (hidden), intended to be overwritten by subclasses
        """
        # initialize variables
        self._t_series = None

        self._data = {}

        self.fs = sampling_frequency

        self._parnt = None

    @property
    def _parent(self):
        return self._parnt

    @_parent.setter
    def _parent(self, value):
        self._parnt = value
        if self.fs is None:
            self.fs = self._parnt.fs

    @staticmethod
    def __set_key(x, key, value):
        keys = key.split('/', 1)
        if len(keys) == 2:
            if keys[0] not in x:
                x[keys[0]] = {}
            elif not isinstance(x[keys[0]], dict):
                raise ValueError(f"Key ({keys[0]}) is not a dictionary.")

            _BaseProcess.__set_key(x[keys[0]], keys[1], value)
        else:
            x[keys[0]] = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, values):
        """
        Value is a tuple-like:
        (key, data)

        where key is a forward-slash delimited string of keys, ie 'Time Series 1/Gait Bout 2/Bout Start Stop'
        would go into ['Time Series 1']['Gait Bout 2']['Bout Start Stop']
        """
        key, value = values
        if isinstance(self._data, dict):
            _BaseProcess.__set_key(self._data, key, value)
        else:
            self._data[key] = value

    def predict(self, data):
        """
        Fit and transform the data with the given process.

        Parameters
        ----------
        data : {str, dict}
            Either a H5 file path (string), or a dictionary. Both the H5 format and the dictionary must follow the
            below format.

        Notes
        -----
        The layout for the H5 file or the dictionary must be as follows (keys that are generated from processing steps
        are in angle brackets <...>):

        * Data

          * Time Series 1

            * Timestamps
            * IMU Data
            * <Windowed IMU Data>
            * <Signal Features>
            * <Gait Classification>
            * <Gait Bout 1>

              * <Bout Start Stop>
              * <Initial Contacts>
              * <Final Contacts>
              * <Per Step Features>

            * <Gait Bout 2>

              * ...
          * Time Series 2

            * ...
        """
        if isinstance(data, dict):  # dictionary file passed
            self._data = data  # directly set
            self._t_series = [i for i in self.data.keys() if 'time series' in i.lower()]
            self._call()
        else:
            with h5py.File(data, 'r+') as self._data:
                # get a list of time series in the data
                self._t_series = [i for i in self.data.keys() if 'time series' in i.lower()]
                self._call()

    def _call(self):
        pass

    @staticmethod
    def _check_sign(val, name, pos=True, inc_zero=False):
        if pos:
            if inc_zero:
                if val < 0:
                    raise ValueError(f"{name} must be greater than or equal to 0.")
            else:
                if val <= 0:
                    raise ValueError(f"{name}  must be greater than 0.")
        else:
            if inc_zero:
                if val > 0:
                    raise ValueError(f"{name} must be less than or equal to 0.")
            else:
                if val >= 0:
                    raise ValueError(f"{name} must be less than 0.")

    def _requires(self, keys):
        """
        Check for required keys in the data being used.

        Parameters
        ----------
        keys : {str, list-like}
            A string or list-like of keys that are required to perform the processing

        Raises
        ------
        KeyError
            If the required keys are not found
        """
        pass

    @staticmethod
    def _old_requires(data, series_keys=None, bout_keys=None):
        """
        Check for required keys in the data being passed.

        Parameters
        ----------
        series_keys : list
            List of required keys for time series
        bout_keys : list
            List of required keys for gait bout information
        """
        assert (series_keys is not None) or (bout_keys is not None), 'Either series_keys or bout_keys must be provided.'

        if series_keys is None:
            series_keys = []
        if bout_keys is None:
            bout_keys = []

        time_series = [i for i in data.keys() if 'time series' in i.lower()]
        if not time_series:
            raise ValueError('No time series entries found')
        for ts in time_series:
            for key in series_keys:
                if key not in data[ts]:
                    raise ValueError(f"Key ({key}) not in data Time Series ({ts}) keys.")
            bouts = [i for i in data[ts].keys() if 'Bout' in i]

            for bout in bouts:
                for key in bout_keys:
                    if key not in data[ts][bout]:
                        raise ValueError(f'Key ({bout}.{key}) not in data Time Series ({ts}) keys.')
