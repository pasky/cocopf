#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimization method (optimization algorithm) interface that allows
per-iteration stepped access to various optimizers, referred to by
their name as a string.

By default, this wraps around the SciPy optimizers and the reference
CMA-ES Python implementation by N. Hansen (if you installed it).  In the
future, we will probably add even more optimizers.  However, if you want
to add some optimizers locally, this class is designed to be subclassed
for precisely that purpose.

All in all, these minimization method names are recognized:

    * CMA, IPOP-CMA, BIPOP-CMA: The CMA algorithm.  IPOP and BIPOP are
      some specific restart strategies that enhance performance; especially
      BIPOP will make CMA quite universally great across the BBOB benchmark.

      `pip install cma` to get the module, otherwise you will get an
      exception when you try to use it.  For BIPOP, you will need either
      some very new cma version or on your own apply our patch from

            http://pasky.or.cz/dev/scipy/cma-1.1.02-bipop.patch

    * Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, TNC, SLSQP (as of SciPy 0.13.3;
      as other minimizers appear in SciPy, they will work automatically).
      These are local searches; the so.minimize call is wrapped in
      so.basinhopping which provides an intelligent restart strategy.

      Note that some SciPy minimizers (not listed above) are not supported
      since they are not black-box but require at least function derivations.
      COBYLA is not supported as it (yet?) does not implement a callback
      functionality.
"""

import os

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
        self.minimizer_kwargs = dict()

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
        if name.upper() in ['CMA', 'IPOP-CMA', 'BIPOP-CMA']:
            self._setup_cma(name)
        else:
            # General fallback, open ended method naming
            self._setup_scipy(name)

    def _setup_cma(self, name):
        import cma

        def cma_wrapper(fun, x0, callback, minimizer_kwargs):
            class InnerCMACallback:
                def __init__(self, realcb):
                    self.realcb = realcb

                def __call__(self, cma):
                    self.realcb(cma.best.x)

            cb = minimizer_kwargs.pop('callback')
            minimizer_kwargs['options']['termination_callback'] = InnerCMACallback(cb) if cb is not None else None

            if 'restarts' in minimizer_kwargs:
                # We ignore passed x0 in case a restart strategy is employed
                # as it is important to start at a different point in each restart
                # (esp. in the smallpop stages of BIPOP, obviously).
                x0 = '10. * np.random.rand(%d) - 5' % minimizer_kwargs.pop('dim')

            try:
                return cma.fmin(fun, x0, 10./4., **minimizer_kwargs)
            except cma._Error, e:
                print "CMA error: " + str(e)
                return None

        self.outer_loop = cma_wrapper

        self.minimizer_kwargs = dict(
                options={'ftarget': self.fi.f.ftarget,
                         'maxfevals': self.fi.maxfunevals - self.fi.f.evaluations,
                         'verb_disp': 100, 'verb_filenameprefix': '/tmp/outcmaes'}
            )

        # Possibly set up a restart strategy
        if name.upper() == 'IPOP-CMA':
            self.minimizer_kwargs['restarts'] = 9
        elif name.upper() == 'BIPOP-CMA':
            self.minimizer_kwargs['restarts'] = 9
            self.minimizer_kwargs['bipop'] = True

        if 'restarts' in self.minimizer_kwargs:
            self.minimizer_kwargs['dim'] = self.fi.dim

    def _setup_scipy(self, name):
        if name.lower() in ['anneal', 'cobyla']:
            raise RuntimeError('MinimizationMethod does not support SciPy method %s (does not provide callback functionality).' % name)

        self.minimizer_kwargs = dict(
                method=name,
                # Bounded local optimizers
                bounds=[(-6., +6.) for d in range(self.fi.dim)],
                # COBYLA
                constraints=({"type": "ineq", "fun": lambda x: np.min(x+5)},
                             {"type": "ineq", "fun": lambda x: np.min(-(x-5))}),
                # Specific options
                options=dict(
                    # COBYLA
                    rhoend=self.fi.f.precision,
                ),
            )

    def __call__(self, fun, x0, inner_cb=None, outer_cb=None):
        """
        A callable interface.  Call on ``fun`` objective function with
        initial solution ``x0``.  ``inner_cb`` has the semantics of
        `scipy.optimize.minimize` callback, ``outer_cb`` has the semantics
        of `scipy.optimize.basinhopping` callback.  ``outer_cb`` may not
        ever get called if the method is non-restarting (e.g. CMA global
        optimization); ``inner_cb`` is the stepping functionality stopping
        point.

        Note that some minimization methods (e.g. *IPOP-CMA) may currently
        ignore the passed ``x0`` value.
        """
        return self.outer_loop(fun, x0, callback=outer_cb,
                minimizer_kwargs=dict(callback=inner_cb, **self.minimizer_kwargs))


class SteppingData:
    """
    This class logs data on current progress of method stepping
    (typically after each method iteration) to an .mdat file.
    """
    def __init__(self, fi):
        self.f = fi.f
        self.total_iters = 0
        self.last_best = None

        # XXX: This is evil; copied from beginning of fgeneric.evalfun()
        if not self.f._is_setdim or self.f._dim != fi.dim:
            self.f._setdim(fi.dim)
        if not self.f._is_ready():
            self.f._readytostart()

        self.datafile = open(os.path.splitext(self.f.datafile)[0] + '.mdat', 'a')
        self.datafile.write("% function evaluation | portfolio iteration | instance index | instance method | instance invocations | instance best noise-free fitness - Fopt | best noise-free fitness - Fopt\n")  # | x1 | x2...

    def end_iter(self):
        self.total_iters += 1

    def record(self, i, name, iters, fitness, point):
        e = self.f.lasteval
        best = e.bestf - self.f.fopt
        res = ('%d %d %d %s %d %+10.9e'
               % (e.num, self.total_iters, i, name, iters, fitness))

        if self.last_best is None or best != self.last_best:
            res += (' %+10.9e' % best)
            self.last_best = best

        # This information is not really useful and taking it out reduces
        # the uncompressed .mdat file size to 1/5.
        # tmp = []
        # for x in point:
        #     tmp.append(' %+5.4e' % x)
        # res += ''.join(tmp)

        self.datafile.write(res + '\n')
        self.datafile.flush()
