#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Run a single given minimization method outside of a portfolio.
# (N.B. SciPy minimizers are still wrapped in basinhopping.)  If
# RESTARTS is specified (and bigger than 0), the function evaluation
# limits are scheduled to allow for #RESTARTS independent restarts
# (but the restart split is not precise, as methods are black boxes).
#
# At any rate, if the minimizer converges but does not reach optimum
# and some time is still left, it is restarted.
#
# Usage: single.py METHOD [RESTARTS] [MAXFEV]
#
# Example: single.py L-BFGS-B 0 1000

import sys # in case we want to control what to run via command line args
import time
import warnings

import numpy as np
import scipy.optimize as so

sys.path.append('.')
import fgeneric
import bbobbenchmarks
from cocopf.experiment import Experiment
from cocopf.methods import MinimizeMethod


class MMCancel(Exception):
    pass

def minimize_f(fi, method = None, wantrestarts = None):
    """
    Minimize the ``fi`` function instance.  Returns the number of minimization
    iterations performed.
    """
    f = fi.f
    n_restarts = -1

    mm = MinimizeMethod(method, fi)

    # independent restarts until maxfunevals or ftarget is reached
    while not ((f.evaluations > 1 and f.fbest < f.ftarget)
               or f.evaluations > fi.maxfunevals):
        n_restarts += 1
        if n_restarts > 0:
            f.restart('independent restart')  # additional info
        maxfevals = fi.maxfunevals / (wantrestarts + 1)

        x0 = 10. * np.random.rand(fi.dim) - 5.

        class MMCallback:
            def __init__(self, fi, f, maxfevals):
                self.restarts = 0
                self.fi = fi
                self.f = f
                self.maxfevals = maxfevals
                self.basefevals = self.f.evaluations
            def __call__(self, x):
                y = self.f.evalfun(x)
                if y < self.f.ftarget:
                    raise MMCancel()
                elif self.f.evaluations - self.basefevals > self.maxfevals:
                    raise MMCancel()
                elif self.f.evaluations > self.fi.maxfunevals:
                    raise MMCancel()
        cb = MMCallback(fi, f, maxfevals)

        try:
            warnings.simplefilter("ignore") # ignore warnings about unused/ignored options
            mm(f.evalfun, x0, inner_cb = cb)
        except MMCancel:
            pass # Ok.

    return n_restarts


if __name__ == "__main__":
    method = sys.argv[1]
    wantrestarts = 0 if len(sys.argv) <= 2 else eval(sys.argv[2])
    maxfev = '1050' if len(sys.argv) <= 3 else sys.argv[3]

    if wantrestarts > 0:
        shortname = '%sr%d' % (method, wantrestarts)
        comments = 'Method %s with %d planned restarts' % (method, wantrestarts)
    else:
        shortname = '%s' % method
        comments = 'Method %s' % method
    e = Experiment(eval(maxfev), shortname, comments)

    for i in e.finstances():
        restarts = minimize_f(i, method, wantrestarts)
        e.f.finalizerun()
        e.freport(i, ' with %d restarts' % restarts)
