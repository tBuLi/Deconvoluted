# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np

# Set the fourier convention as described in
# http://mathworld.wolfram.com/FourierTransform.html
Convention = namedtuple('Convention', ['a', 'b'])
signal = Convention(a=0, b=- 2 * np.pi)  # Signal processing

def fourier_transform(f, *vars, convention=signal):
    """
    Performs the multidimensional Fourier transform of
    :math:`f(x_1, \ldots, x_n)` with respect to any number of variables
    :math:`x_i`.

    Example::

         F, k = fourier_transform(f, x)

    :param f: array representing a function :math:`f(x_1, \ldots, x_n)`
    :param vars: list of axis w.r.t. which the Fourier transform has to be
        computed.
    :param convention: The Fourier convention to be used.
    :return: :math:`F(k_1, \ldots, k_n)`, the Fourier transform of
        :math:`f(x_1, \ldots, x_n)`.
    """
    F = np.fft.fftshift(np.fft.fft(f))
    ks = []
    for x in vars:
        d = x[1]-x[0]
        k = np.fft.fftfreq(len(x), d=d)
        k = np.fft.fftshift(k)  # Go from -inf to inf
        ks.append(k)
    return (F, *ks)


def inverse_fourier_transform(F, *vars, convention=signal):
    """
    Perform an inverse Fourier transform. See
    :func:`deconvoluted.transforms.fourier_transform` for more info.

    :param F: Fourier transform :math:`F(k_1, \ldots, k_n)`
        of :math:`f(x_1, \ldots, x_n)`.
    :param vars: Any number of :math:`k` variables.
    :param convention:
    :return: :math:`f(x_1, \ldots, x_n)`, the inverse fourier transform of
        :math:`F(k_1, \ldots, k_n)`
    """
    f = np.fft.ifft(np.fft.ifftshift(F))
    xs = []
    for k in vars:
        d = k[1] - k[0]
        x = np.fft.fftfreq(len(k), d=d)
        x = np.fft.fftshift(x)  # Go from -inf to inf
        xs.append(x)
    return (f, *xs)

def laplace_tranform(f, *vars, s):
    raise NotImplementedError('Laplace transforms are not yet supported.')
