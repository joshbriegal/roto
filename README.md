[![Python tests](https://github.com/joshbriegal/roto/actions/workflows/pythontests.yaml/badge.svg?branch=main)](https://github.com/joshbriegal/roto/actions/workflows/pythontests.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# Rotation Toolkit (RoTo)

Hello RoTo

## What does this package do?

This package provides tools for extracting rotation periods from astrophysical time series data.

```
timeseries, flux, (flux_errors) -> rotation_period(s)
```

Output rotation period + uncertainty (master).
Output rotation period + uncertainty for each method used.

```python
roto = RoTo(timeseries, flux, flux_errors, **config_kwargs)  # instantiate object

roto()  # call to find periods

print(roto)  # will print an ascii table of the outputted periods

```

Inspiration from the astropy lomb scargle class:

```python
ls = LombScargle(timeseries, flux, flux_errors, **kwargs)

ls.autopower()

```

### Lomb Scargle

AstroPy Lomb Scargle
Bayesian Lomb Scargle:
    - LS on whole LC, rough estimate of period
    - Sliding window across time series (of e.g. 5*initial guess)
    - Build up series of periods, estimate uncertainty.

### G-ACF

The Generalised Autocorrelation Function (G-ACF)
    - Need an uncertainty
    - Damped harmonic oscillator model

### GP Regression

Gaussian Process Model(s)
    - Celerite2 - fast
    - George? - users note it may be better for modellling evolution

### Wavelet Analysis

Wavelet Analysis - TBC

