#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A method wrapper that will normally wrap SciPy minimizers accessed
via so.basinhopping and so.minimize, but is designed to allow
overloading by other user-provided minimizers.

As of SciPy 0.13.3, you can use these SciPy methods with MinimizeMethod:

    Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, TNC, SLSQP

The remaining SciPy minimizers are not supported since they are not
black-box but require at least function derivations.  COBYLA is not
supported as it (yet?) does not implement a callback functionality.
"""

import string
import sys
import time
import warnings

import numpy as np
import scipy.optimize as so


class MinimizeMethod(object):
    """
    A specific minimization method initialized  with regard to a specific
    benchmarked function (that could influence some of the parameters, etc.).
    The object is a callable that, when called, should start the minimization
    method (see ``__call__`` description below).

    Attributes:
        ``name``: Provided name of the method.
        ``fi``: Provided FInstance object describing the benchmark function.
        ``outer_loop``: A method to be used as a minimizer callable.
            Typically, this would be a global minimization routine with
            a calling convention similar to `scipy.optimize.basinhopping`;
            if that doesn't suit you, provide a wrapper lambda or override
            the ``__call__`` method as well.
        ``minimizer_kwargs``: The minimizer parameters.

    Example:

    >> m = MinimizeMethod("BFGS", fi)
    >> m(lambda x: np.sum(x ** 2), x0, inner_cb = callback)

    You are expected to inherit this class if you want to allow other methods
    than those that are part of scipy.  The recommended way is to override the
    _set_method() method to parse the provided name and in case of a match,
    reconfigure the callable.  If you want to use the stock scipy basinhopping
    outer_loop, you will need scipy version 0.14 or newer to be able to pass
    a callable instead of a string as the `method` argument.

    Note that even though this interface is designed to perform global
    minimization, it may sometimes terminate early outside of global optimum
    and you should plan your experiments to restart it in that case.
    """

    def __init__(self, name, fi):
        """
        Initialize the method described by ``name`` for the particular
        FInstance ``fi``.  By default, we set the `outer_loop` to
        `scipy.optimize.basinhopping` to perform global minimization,
        and set up minimize parameters to perform a bounded search
        between -6.0 and +6.0.

        Method-specific setup is done in `_setup_method`, called at
        the end of the constructor.
        """
        self.name = name
        self.fi = fi

        self.outer_loop = so.basinhopping
        self.minimizer_kwargs = dict(
                # Bounded local optimizers
                bounds = [(-6., +6.) for d in range(fi.dim)],
                # COBYLA
                constraints = ({ "type": "ineq", "fun": lambda x: np.min(x+5) },
                               { "type": "ineq", "fun": lambda x: np.min(-(x-5)) }),
                # Specific options
                options = dict(
                    # COBYLA
                    rhoend = fi.f.precision,
                ),
            )

        self._setup_method(name)

    def _setup_method(self, name):
        """
        Set up the particular method.  By default, all method names are
        simply passed on as scipy (multivariate) minimization method strings
        (e.g. "L-BFGS-B", "Nelder-Mead", etc.).

        For custom methods support, inherit the class, override this method,
        try to parse the name for custom values and (recommended) call the
        super()._setup_method() as a callback.
        """
        if name.lower() in ['anneal', 'cobyla']:
            raise RuntimeError('MinimizationMethod does not support method %s (does not provide callback functionality).' % name)
        else:
            self.minimizer_kwargs['method'] = name

    def __call__(self, fun, x0, inner_cb = None, outer_cb = None):
        """
        A callable interface.  Call on ``fun`` objective function with
        initial solution ``x0``.  ``inner_cb`` has the semantics of
        `scipy.optimize.minimize` callback, ``outer_cb`` has the semantics
        of `scipy.optimize.basinhopping` callback.  ``outer_cb`` may not
        ever get called if the method is non-restarting (e.g. CMA global
        optimization); ``inner_cb`` is the stepping functionality stopping
        point.
        """
        return self.outer_loop(fun, x0, callback = outer_cb,
                minimizer_kwargs = dict(callback = inner_cb, **self.minimizer_kwargs))
