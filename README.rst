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

Fourier Transforms
~~~~~~~~~~~~~~~~~~

As a first example, let's perform a Fourier transform:

.. code-block:: python

    t = np.linspace(0, 10, 201)
    f = np.sin(3 * 2 * np.pi * t)
    F, nu = fourier_transform(f, t)

By default, Fourier transforms use Fourier coefficients :math:`a=0`,
:math:`b=-2\pi`. Using another convention is simple:

.. code-block:: python

    F, omega = fourier_transform(f, t, convention=(-1, 1))

As a physicist myself, I therefore switch the labelling of the output from
:math:`\nu` for frequency, to :math:`\omega` for angular frequency.

Performing multidimensional transforms is just as easy. For example:

.. code-block:: python

    F_pq, p, q = fourier_transform(f_xy, x, y)

transforms both :math:`x` and :math:`y` at the same time.
Transforming only one of the two variables can be done simply by setting those
that shouldn't transform to ``None``:

.. code-block:: python

    F_py, p = fourier_transform(f_xy, x, None)
    F_xq, q = fourier_transform(f_xy, None, y)

See the documentation for more examples!
