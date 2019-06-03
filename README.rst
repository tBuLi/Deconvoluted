============
Deconvoluted
============


.. image:: https://img.shields.io/pypi/v/deconvoluted.svg
        :target: https://pypi.python.org/pypi/deconvoluted

.. image:: https://img.shields.io/travis/tbuli/deconvoluted.svg
        :target: https://travis-ci.org/tbuli/deconvoluted

.. image:: https://readthedocs.org/projects/deconvoluted/badge/?version=latest
        :target: https://deconvoluted.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Deconvoluted makes performing numerical integral transforms simple and pythonic!


* Free software: MIT license
* Documentation: https://deconvoluted.readthedocs.io.


Features
--------

As a first example, let's perform a Fourier transform:

.. code-block:: python

    p, F = fourier_transform(x, f)

By default, Fourier transforms use Fourier coefficients :math:`a=0`, :math:`b=-2\pi`. Using another convention is simple:

.. code-block:: python

    k, F = fourier_transform(x, f, convention=(-1, 1))

As a physicist myself, I therefore switch the labelling of the output from :math:`k` for wavenumber, to :math:`p` for momentum.

To perform a Laplace transform, we simply do

.. code-block:: python

    s, F = laplace_transform(x, f)

See the documentation for more examples!
