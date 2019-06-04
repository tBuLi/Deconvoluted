#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deconvoluted` package."""


import unittest

import numpy as np

from deconvoluted import fourier_transform, inverse_fourier_transform

class TestDeconvoluted(unittest.TestCase):
    """Tests for `deconvoluted` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_fft_1d(self):
        """
        Test the FFT capabilities.
        """
        # Number of sample points
        N = 60
        # sample spacing
        T = 1.0 / 800.0
        x = np.linspace(- N * T, N * T, 2 * N + 1)  # (- 0.75 , 0.75)
        # Place a peak at 50 Hz and 80 Hz.
        y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)

        F, k = fourier_transform(y, x)
        y_new, x_new = inverse_fourier_transform(F, k)
        np.testing.assert_almost_equal(x, x_new)
        np.testing.assert_almost_equal(y, y_new.real)
        np.testing.assert_almost_equal(np.zeros_like(y), y_new.imag)

    def test_fft_2d(self):
        x = np.linspace(-10, 10, 21)
        y = np.linspace(-10, 10, 21)
        X, Y = np.meshgrid(x, y)
        f_xy = np.sin(30 * 2 * np.pi * X + 15 * 2 * np.pi * Y)
        F_pq, p, q = fourier_transform(f_xy, x, y)
        f_xy_new, x_new, y_new = fourier_transform(F_pq, p, q)

        np.testing.assert_almost_equal(x, x_new)
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(f_xy, f_xy_new.real)
        np.testing.assert_almost_equal(np.zeros_like(f_xy), f_xy_new.imag)

        with self.assertRaises(TypeError):
            fourier_transform(f_xy, x)

        F_py, p_new = fourier_transform(f_xy, x, None)
        F_pq_new, q_new = fourier_transform(F_py, None, y)

        np.testing.assert_almost_equal(p, p_new)
        np.testing.assert_almost_equal(q, q_new)
        np.testing.assert_almost_equal(F_pq_new, F_pq)

        # TODO: Do the inverses!
        # Do the inverse in two steps
        F_py_new, y_new = inverse_fourier_transform(F_pq, None, q)
        # Compare the intermediate
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(F_py, F_py_new)
        # Compare the final state, which should be the initial state
        f_xy_new, x_new = inverse_fourier_transform(F_py_new, p, None)
        np.testing.assert_almost_equal(x, x_new)
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(f_xy, f_xy_new.real)
        np.testing.assert_almost_equal(np.zeros_like(f_xy), f_xy_new.imag)

        # Do the entire inverse in one step
        f_xy_new, x_new, y_new = inverse_fourier_transform(F_pq, p, q)
        np.testing.assert_almost_equal(x, x_new)
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(f_xy, f_xy_new.real)
        np.testing.assert_almost_equal(np.zeros_like(f_xy), f_xy_new.imag)

