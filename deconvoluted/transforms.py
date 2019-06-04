# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np

# Set the fourier convention as described in
# http://mathworld.wolfram.com/FourierTransform.html
Convention = namedtuple('Convention', ['a', 'b'])
signal = Convention(a=0, b=- 2 * np.pi)  # Signal processing

def determine_axes(f, *vars):
    if len(vars) != len(f.shape):
        raise TypeError('The number of variables has to match the dimension of '
                        '`f`. Use `None` for axis with respect to which no '
                        'transform should be performed.')

    return [i for i, var in enumerate(vars) if var is not None]

def fourier_transform(f, *vars, convention=signal):
    """
    Performs the multidimensional Fourier transform of
    :math:`f(x_1, \ldots, x_n)` with respect to any number of variables
    :math:`x_i`.

    Examples::

        # 1D transform
        F, k = fourier_transform(f, x)

        # 2D transform
        F_pq, p, q = fourier_transform(f_xy, x, y)

        # 2D function, transform only 1 axis
        F_py, p = fourier_transform(f_xy, x, None)

    :param f: array representing a function :math:`f(x_1, \ldots, x_n)`
    :param vars: list of :math:`x_i` w.r.t. which the Fourier transform has to
        be computed. In case of multi-dimensional functions :math:`f` the
        number of ``vars`` has to match the dimension of ``f``. Any axis that
        should be ignored should be provided as ``None``::

            F_py, p = fourier_transform(f_xy, x, None)

    :param convention: The Fourier convention to be used. :math:`a=0` and
        :math:`b=- 2 \pi` by default, which is the signal processing standard.
    :return: :math:`F(k_1, \ldots, k_n)`, the Fourier transform of
        :math:`f(x_1, \ldots, x_n)`.
    """
    axes = determine_axes(f, *vars)

    F = np.fft.fftn(f, axes=axes)
    F = np.fft.fftshift(F, axes=axes)
    ks = []
    for x in vars:
        if x is None:
            continue
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
    :param vars: Any number of :math:`k` variables or ``None``.
    :param convention: The Fourier convention to be used. :math:`a=0` and
        :math:`b=- 2 \pi` by default, which is the signal processing standard.
    :return: :math:`f(x_1, \ldots, x_n)`, the inverse fourier transform of
        :math:`F(k_1, \ldots, k_n)`
    """
    axes = determine_axes(F, *vars)

    f = np.fft.ifftshift(F, axes=axes)
    f = np.fft.ifftn(f, axes=axes)
    xs = []
    for k in vars:
        if k is None:
            continue
        d = k[1] - k[0]
        x = np.fft.fftfreq(len(k), d=d)
        x = np.fft.fftshift(x)  # Go from -inf to inf
        xs.append(x)
    return (f, *xs)
