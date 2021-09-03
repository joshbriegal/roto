from typing import Optional

import numpy as np
import numpy.fft as fft
from matplotlib.axes import Axes

from roto.methods.periodfinder import PeriodFinder, Periodogram, PeriodResult


class FFTPeriodFinder(PeriodFinder):
    """Fast Fourier Transform (FFT) method to find periods.
    Conforms to PeriodFinder interface.
    """

    def calculate_periodogram(self, **kwargs) -> Periodogram:
        """Calculates FFT Periodogram of data

        Args:
            n_times_data (int, default 32): how many times to pad data for additional precision.
            len_ft (int): input length of fourier series (if longer than data will provide additional precision).
            power_of_two (bool, default True): if True, will extend input data series to power of two length for additional speed up.
            pad_both_sides (bool, default True): if True, will add padding to start and end of data as opposed to just the end.

        Returns:
            Periodogram: frequency/power periodogram.
        """

        n_times_data = kwargs.get("n_times_data", 32)
        len_ft = kwargs.get("len_ft", len(self.flux) * n_times_data)
        power_of_two = kwargs.get("power_of_two", True)
        pad_both_sides = kwargs.get("pad_both_sides", True)

        if power_of_two:
            # add more zeros to make the length of the array a power of 2, for computational speed up
            len_ft = int(2.0 ** np.ceil(np.log2(len_ft)))
        if pad_both_sides:
            num_zeros = max(0, len_ft - len(self.flux))
            num_zeros = num_zeros / 2
            if int(num_zeros) != num_zeros:
                # rounding
                zeros1 = np.zeros(int(num_zeros))
                zeros2 = np.zeros(int(num_zeros) + 1)
            else:
                zeros1 = zeros2 = np.zeros(int(num_zeros))
            flux = np.append(zeros1, np.append(self.flux, zeros2))
        else:
            flux = self.flux

        complex_ft = fft.rfft(flux, n=len_ft)
        real_ft = np.absolute(complex_ft)

        freqs = fft.rfftfreq(len_ft, self.timeseries[1] - self.timeseries[0])

        return Periodogram(frequency_axis=freqs, power_axis=real_ft)

    def plot(
        self, ax: Axes, period: PeriodResult, colour: Optional[str] = "orange"
    ) -> Axes:
        """Given a figure and an axis plot the interesting output of the object.

        Args:
            ax ([type]): Matplotlib axis
            period (PeriodResult): Outputted period to plot around
        """
        ax = self.plot_periodogram(ax, period, colour=colour)
        ax.set_title("FFT Periodogram")
        return ax
