from typing import Optional
from typing import Tuple

import numpy as np
import numpy.fft as fft

from src.methods.periodfinder import PeriodFinder, Periodogram


class FFTPeriodFinder(PeriodFinder):

    def __init__(
        self,
        timeseries: np.ndarray,
        flux: np.ndarray,
        flux_errors: Optional[np.ndarray] = None,
    ):
        super().__init__(timeseries, flux, flux_errors)

    def calculate_periodogram(self, **kwargs) -> Periodogram:
        """
        Caluclate FFT of LCObject Data
        :return: tuple
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
