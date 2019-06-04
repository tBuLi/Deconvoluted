#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deconvoluted` package."""


import unittest

import numpy as np

from deconvoluted import fourier_transform, inverse_fourier_transform

class TestDeconvoluted(unittest.TestCase):
    """Tests for `deconvoluted` package."""

    def setUp(self):
        x = np.linspace(-20, 20, 41)
        y = np.linspace(-10, 10, 21)
        X, Y = np.meshgrid(x, y)
        f_xy = np.sin(0.2 * 2 * np.pi * X + 0.1 * 2 * np.pi * Y)
        self.data2d = (f_xy, x, y)
        # Test if these are really inverses.
        F_pq, p, q = fourier_transform(f_xy, x, y)
        self.ft_data2d = (F_pq, p, q)

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

    def test_ifft_2d_simultanious(self):
        """
        Check the 2d ifft by performing the transform for both axes at the
        same time.
        """
        f_xy, x, y = self.data2d
        F_pq, p, q = self.ft_data2d
        # Test if these are really inverses.
        f_xy_new, x_new, y_new = inverse_fourier_transform(F_pq, p, q)

        np.testing.assert_almost_equal(x, x_new)
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(f_xy, f_xy_new.real)
        np.testing.assert_almost_equal(np.zeros_like(f_xy), f_xy_new.imag)

    def test_ifft_2d_single(self):
        """
        Test the API of a single axis transform on 2d data, by transforming
        only one axis and then transforming it back.
        """
        f_xy, x, y = self.data2d

        # Test single transforms
        F_py, p_new = fourier_transform(f_xy, x, None)
        f_xy_new, x_new = inverse_fourier_transform(F_py, p_new, None)
        np.testing.assert_almost_equal(x_new, x)
        np.testing.assert_almost_equal(f_xy_new.real, f_xy)

    def test_fft_2d_two_singles(self):
        """
        Test the API of a single axis transform on 2d data by performing the
        full 2d transform in two 1d steps.
        """
        f_xy, x, y = self.data2d
        F_pq, p, q = self.ft_data2d

        # Behavior for every axis needs to be indicated.
        with self.assertRaises(TypeError):
            fourier_transform(f_xy, x)

        # Test single transforms
        F_py, p_new = fourier_transform(f_xy, x, None)
        F_pq_new, q_new = fourier_transform(F_py, None, y)

        np.testing.assert_almost_equal(p, p_new)
        np.testing.assert_almost_equal(q, q_new)
        np.testing.assert_almost_equal(F_pq_new, F_pq)

    def test_ifft_2d_two_singles(self):
        """
        Compute the 2d ifft by doing two singles.
        :return:
        """
        f_xy, x, y = self.data2d
        F_pq, p, q = self.ft_data2d

        # Behavior for every axis needs to be indicated.
        with self.assertRaises(TypeError):
            inverse_fourier_transform(F_pq, p)

        # Do the inverse from two sides to see if the result matches
        F_py_f, p_new = fourier_transform(f_xy, x, None)  # Forward
        F_py_b, y_new = inverse_fourier_transform(F_pq, None, q)  # Backward

        # Compare the intermediate
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(p, p_new)
        np.testing.assert_almost_equal(F_py_f, F_py_b)

        # Compare the final state, which should be the initial state
        f_xy_new, x_new = inverse_fourier_transform(F_py_b, p, None)
        np.testing.assert_almost_equal(x, x_new)
        np.testing.assert_almost_equal(y, y_new)
        np.testing.assert_almost_equal(f_xy, f_xy_new.real)
        np.testing.assert_almost_equal(np.zeros_like(f_xy), f_xy_new.imag)

