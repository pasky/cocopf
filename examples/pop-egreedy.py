#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create a population of K solutions and iteratively minimize each
# (step-by-step) using METHOD, applying an epsilon-greedy strategy.
#
# Stops when optimum is found or we reach the MAXFEV number of function
# evaluations.
#
# A mix of methods is also supported, if METHOD is a comma-separated
# list of methods, the methods will be uniformly assigned to population
# items. (It is wise to have K % #methods == 0.)
#
# Usage: pop-egreedy.py METHOD [K] [MAXFEV] [EPS] [ACCRUAL]
#
# Example: pop-egreedy.py Powell,BFGS,SLSQP 6
# Example: pop-egreedy.py Powell,BFGS,SLSQP 3 10000 0.5 adapt0.6r

import string
import sys
import time

import numpy as np
import scipy.optimize as so

sys.path.append('.')
import fgeneric
import bbobbenchmarks
from cocopf.credit import PopulationCredit
from cocopf.experiment import Experiment
from cocopf.methods import MinimizeMethod
from cocopf.population import Population


def minimize_f(fi, K = None, method = None, eps = None, accrual = None):
    """
    Minimize the ``fi`` function instance.  Returns the number of minimization
    iterations performed and method finding the optimum (if any).
    """
    f = fi.f
    optmethod = None

    pop = Population(fi, K, [MinimizeMethod(name, fi) for name in string.split(method, ',')])
    # Credit Assignment does not matter for us as long as it's order-invariant
    popcredit = PopulationCredit(pop, "raw", accrual)

    stop = False

    # Initial iterations - evaluate each function once.
    for i in range(K):
        (x, y) = pop.step_one(i)
        #print("[%d] #%d %s=%s" % (pop.total_iters, i, x, y))
        if y < f.ftarget:
            optmethod = pop.minimizers[i].minmethod.name
            stop = True
            break # stop immediately, no point in going on
        pop.end_iter()
        popcredit.update()

    # Main set of iterations - explore/exploit.
    while not stop and f.evaluations < fi.maxfunevals:
        if np.random.rand() < eps:
            # Explore
            i = np.random.randint(K)
        else:
            # Exploit
            i = popcredit.credit.argmin()

        (x, y) = pop.step_one(i)
        #print("[%d] #%d %s=%s" % (pop.total_iters, i, x, y))
        if y < f.ftarget:
            optmethod = pop.minimizers[i].minmethod.name
            stop = True
            break # stop immediately, no point in going on
        pop.end_iter()
        popcredit.update()

    pop.stop()

    return (pop.total_iters, optmethod)


if __name__ == "__main__":
    method = sys.argv[1]
    K = 30 if method.count(',') == 0 else method.count(',') + 1
    if len(sys.argv) > 2:
        K = eval(sys.argv[2])
    maxfev = '1050' if len(sys.argv) <= 3 else sys.argv[3]
    eps = 0.5 if len(sys.argv) <= 4 else eval(sys.argv[4])
    accrual = 'latest' if len(sys.argv) <= 5 else sys.argv[5]

    m_opts = dict(K = K, method = method, eps = eps, accrual = accrual)

    if method.find(',') >= 0:
        shortname = 'mE%dG%d%s' % (int(eps*100), K, accrual if accrual != 'latest' else '')
        comments = 'Iterative epsgreedy-sampling mix (%s), eps=%s, pop. K=%d, cr. accrual = %s' % (method, eps, K, accrual)
    else:
        shortname = 'pE%dG%d%s_%s' % (int(eps*100), K, accrual if accrual != 'latest' else '', method)
        comments = 'Iterative epsgreedy-sampling %s, eps=%s, pop. K=%d, cr. accrual = %s' % (method, eps, K, accrual)
    e = Experiment(eval(maxfev), shortname, comments)

    for i in e.finstances():
        (n_iters, optmethod) = minimize_f(i, **m_opts)
        e.f.finalizerun()

        comment = ' with %d iterations' % n_iters
        if optmethod is not None:
            comment += ' (* %s)' % optmethod
        e.freport(i, comment)
