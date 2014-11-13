#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create a population of K solutions and iteratively minimize each
# (step-by-step) using METHOD, giving each solution the same number
# of function evaluations.  A solution is replaced in case minimization
# hits local optimum.  Stops when optimum is found or we reach the MAXFEV
# number of function evaluations.
#
# A mix of methods is also supported, if METHOD is a comma-separated
# list of methods, the methods will be uniformly assigned to population
# items. (It is wise to have K % #methods == 0.)
#
# Usage: pop-uniform.py METHOD [K] [MAXFEV]
#
# In addition, this demo extends MinimizeMethod with CMA support.
# Use `pip install cma` to get it. (Last tested with CMA-1.1.02.)
# Actually, stock MinimizeMethod class also supports CMA now, but
# we leave this code around as an example of its subclassing.
#
# Example: pop-uniform.py Powell,BFGS,SLSQP,CMA 4

import string
import sys
import time

import numpy as np
import scipy.optimize as so

sys.path.append('.')
import fgeneric
import bbobbenchmarks
from cocopf.experiment import Experiment
from cocopf.methods import MinimizeMethod
from cocopf.population import Population

import cma


class XMinimizeMethod(MinimizeMethod):
    def _setup_method(self, name):
        if name == "CMA":
            class CMAWrapper:
                """
                Wrap the reference CMA-ES Python implementation.

                N.B. the wrapper ignores the x0 parameter and instead
                lets the CMA implementation re-sample it again; this
                is for benefit of the restart strategies which won't
                start with the same point repeatedly.
                """
                def __init__(self, ftarget, maxfevals):
                    self.ftarget = ftarget
                    self.maxfevals = maxfevals

                def __call__(self, fun, x0, callback, minimizer_kwargs):
                    class InnerCMACallback:
                        def __init__(self, realcb):
                            self.realcb = realcb
                        def __call__(self, cma):
                            self.realcb(cma.best.x)
                    cb = minimizer_kwargs['callback']
                    innercb = InnerCMACallback(cb) if cb is not None else None

                    try:
                        return cma.fmin(fun, x0, 10./4.,
                                        options={'ftarget': self.ftarget, 'maxfevals': self.maxfevals,
                                                 'termination_callback': innercb, 'verb_disp': 0,
                                                 'verb_filenameprefix': '/tmp/outcmaes'})
                    except cma._Error, e:
                        print "CMA error: " + str(e)
                        return None

            self.outer_loop = CMAWrapper(self.fi.f.ftarget,
                    self.fi.maxfunevals - self.fi.f.evaluations)

        else:
            # Defer to main implementation (assume stock scipy minimizer)
            MinimizeMethod._setup_method(self, name)


def minimize_f(fi, K = None, method = None):
    """
    Minimize the ``fi`` function instance.  Returns the number of minimization
    iterations performed and method finding the optimum (if any).
    """
    f = fi.f
    optmethod = None

    pop = Population(fi, K, [XMinimizeMethod(name, fi) for name in string.split(method, ',')])

    # Iterate; make a full iteration even in case maxfunevals is reached.
    stop = False
    while not stop:
        # Always run the algorithm which has the least nfevs, to allocate
        # nfevs fairly.
        next_i = np.argmin(pop.nfevs)

        (x, y) = pop.step_one(next_i)
        #print("[%d] #%d %s=%s" % (pop.total_iters, next_i, x, y))
        if y < f.ftarget:
            optmethod = pop.minimizers[next_i].minmethod.name
            stop = True
            break # stop immediately, no point in going on

        if f.evaluations > fi.maxfunevals:
            stop = True
        pop.end_iter()

    pop.stop()

    return (pop.total_iters, optmethod)


if __name__ == "__main__":
    method = sys.argv[1]
    K = 30 if method.count(',') == 0 else method.count(',') + 1
    if len(sys.argv) > 2:
        K = eval(sys.argv[2])
    maxfev = '1050' if len(sys.argv) <= 3 else sys.argv[3]

    m_opts = dict(K = K, method = method)

    # b stands for "basinhopping"
    if method.find(',') >= 0:
        shortname = 'eUNIF%d' % K
        comments = 'Iterative UNIFORM-sampling mix (%s), pop. K=%d' % (method, K)
    else:
        shortname = 'pUNIF%d_%s' % (K, method)
        comments = 'Iterative UNIFORM-sampling %s, pop. K=%d' % (method, K)
    e = Experiment(eval(maxfev), shortname, comments)

    for i in e.finstances():
        (n_iters, optmethod) = minimize_f(i, **m_opts)
        e.f.finalizerun()

        comment = ' with %d iterations' % n_iters
        if optmethod is not None:
            comment += ' (* %s)' % optmethod
        e.freport(i, comment)
