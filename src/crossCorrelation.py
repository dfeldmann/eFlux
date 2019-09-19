#!/usr/bin/env python3
# Purpose:  Compute cross-correlation coefficients for two data series a and b
#           of same shape; one- or two-dimensional data sets. Here, the data
#           series a represents the "test image" and the data series b
#           represents the "probe image", which is traveresd across the "test
#           image". The correlation is computed in Fourier space making use of
#           the convolution theorem, see e.g. page 602 in Numerical Recipes.
# Usage:    python crossCorrelation.py 
# Authors:  Daniel Feldmann, Mohammad Umair
# Date:     28th March 2019
# Modified: 16th September 2019



def corr1d(a, b):
    # input:  1d data series a and b of same shape
    # output: 1d cross-correlation of a and b
    
    import numpy as np
 
    na = np.linalg.norm(a)              # L2 norm (RMS) of first data series
    nb = np.linalg.norm(b)              # L2 norm (RMS) of second data series
    n  = na * nb                        # for normalisation
    n  = 1.0                            # for normalisation
    n  = len(a)                         # normalise with length of data series

    af = np.fft.fft(a)                  # 1d FFT of first data series
    bf = np.fft.fft(b)                  # 1d FFT of second data series
    c  = np.fft.ifft(af * np.conj(bf))  # compute cross-correlation in Fourier space

    c  = np.fft.fftshift(c) / n         # shift and normalise
    return c.real                       # output only the real part of correlation



def corr2d(a, b):
    # input:  2d data series a and b of same shape
    # output: 2d cross-correlation of a and b
    
    import numpy as np
 
    na = np.linalg.norm(a)              # L2 norm (RMS) of first data series
    nb = np.linalg.norm(b)              # L2 norm (RMS) of second data series
    n  = na * nb                        # for normalisation

    af = np.fft.fft2(a)                 # 2d FFT of first data series
    bf = np.fft.fft2(b)                 # 2d FFT of second data series
    c  = np.fft.ifft2(af * np.conj(bf)) # compute cross-correlation in Fourier space

    c  = np.fft.fftshift(c) / n         # shift and normalise
    return c.real                       # output only the real part of correlation
