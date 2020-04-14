"""
Windowing processes for windowing over days
"""
import udatetime
import datetime
from numpy import argmin, abs, array

from sit2standpy.v2.base import _BaseProcess, PROC, DATA


__all__ = ['WindowDays']


class WindowDays(_BaseProcess):
    def __init__(self, hours=[8, 20], **kwargs):
        """
        Window data into days, with the default behaviour to take the hours of most likely wakefulness

        Parameters
        ----------
        hours : list-like of int
            Hours to include in the windowed data. Default is 8 to 20, which excludes the night from
            the detection of sit-to-stand transfers.
        """
        super().__init__(**kwargs)

        self._hours = hours

    def _call(self):
        utime = self.data['Sensors']['Lumbar']['Unix Time']
        # get the first timepoint to know which day to start and end with
        time_sdt = udatetime.utcfromtimestamp(utime[0])
        time_edt = udatetime.utcfromtimestamp(utime[-1])

        n_days = (time_edt.date() - time_sdt.date()).days
        if time_edt.hour > self._hours[0]:
            n_days += 1

        # set the start and end hours for the first day
        day_start = time_sdt.replace(hour=self._hours[0], minute=0, second=0, microsecond=0)
        day_end = time_sdt.replace(hour=self._hours[1], minute=0, second=0, microsecond=0)

        iend = 10  # set so can reference in the i=0 loop
        for i in range(n_days):
            istart = argmin(abs(utime[iend-10:] - day_start.timestamp())) + iend - 10
            iend = argmin(abs(utime[istart:] - day_end.timestamp())) + istart + 1

            self.data = (PROC.format(day_n=i+1, value='Indices'), array([istart, iend]))

            day_start += datetime.timedelta(days=1)
            day_end += datetime.timedelta(days=1)
