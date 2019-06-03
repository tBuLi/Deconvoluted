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

    def test_fft(self):
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
