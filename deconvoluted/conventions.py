from collections import namedtuple

import numpy as np

# Set the fourier convention as described in
# http://mathworld.wolfram.com/FourierTransform.html
Convention = namedtuple('Convention', ['a', 'b'])
Convention.__doc__ = """
Fourier convention, as described in [1](# http://mathworld.wolfram.com/FourierTransform.html).

Some standard settings are available under the following names:
- signal: :math:`(0, - 2 \pi)`. signal processing standard.
- phys_sym: :math:`(0, 1)`. Common in physics, symmetric Fourier pair measured 
    in angular frequency.
- math: :math:`(1, -1)`. Used in pure math
- prob: :math:`(1, 1)`. Used in probability theory when calculating 
    characteristic functions.
- phys_class: :math:`(-1, 1)`. Classical Physics.
"""

signal = Convention(a=0, b=- 2 * np.pi)  # Signal processing
phys_sym = Convention(a=0, b=1)  # Symmetric, physics -> angular frequency
math = Convention(a=1, b=-1)  # Pure maths and system engineering
prob = Convention(a=1, b=1)  # statistics
phys_class = Convention(a=-1, b=1)  # Classical Physics

conventions = [signal, phys_sym, math, prob, phys_class]
