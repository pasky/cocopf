#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A generic manager of iteratively minimized population that will
keep track of solutions and respective minimizer states.

Solution search progress (in respect to method population) is
recorded to an .mdat file.
"""

import os;
import string
import sys
import time
import warnings

import numpy as np
import numpy.random as nr

from cocopf.minstep import MinimizeStepping


class SteppingData:
    def __init__(self, f, dim):
        self.f = f
        self.total_iters = 0

        # XXX: This is evil; copied from beginning of fgeneric.evalfun()
        if not self.f._is_setdim or self.f._dim != dim:
            self.f._setdim(dim)
        if not self.f._is_ready():
            self.f._readytostart()

        self.datafile = open(os.path.splitext(self.f.datafile)[0] + '.mdat', 'a')
        self.datafile.write("% function evaluation | portfolio iteration | instance index | instance method | instance invocations | instance best noise-free fitness - Fopt | best noise-free fitness - Fopt | x1 | x2...\n")

    def end_iter(self):
        self.total_iters += 1

    def record(self, i, name, iters, fitness, point):
        e = self.f.lasteval
        res = ('%d %d %d %s %d %+10.9e %+10.9e'
               % (e.num, self.total_iters, i, name, iters, fitness, e.bestf - self.f.fopt))

        tmp = []
        for x in point:
            tmp.append(' %+5.4e' % x)
        res += ''.join(tmp)

        self.datafile.write(res + '\n')
        self.datafile.flush()


class Population:
    """
    ``points`` contains the solution points of the population.
    ``minimizers`` contains the optimizer instances associated with these points.
    """
    def __init__(self, fi, K, methods):
        self.fi = fi
        self.K = K
        self.methods = methods

        # A population of solution x points
        self.points = 10. * np.random.rand(self.K, self.fi.dim) - 5.
        # A population of solution y points
        self.values = np.zeros(self.K) + 10e10
        # A population of minimizers
        self.minimizers = [self._minimizer_make(i) for i in range(self.K)]
        # A population of iteration counters
        self.iters = np.zeros(self.K, dtype = np.int)

        self.total_steps = 0
        self.total_iters = 0
        self.data = SteppingData(self.fi.f, self.fi.dim);

    def _minimizer_make(self, i):
        warnings.simplefilter("ignore") # ignore warnings about unused/ignored options
        return MinimizeStepping(self.fi.f.evalfun, self.points[i],
                self.methods[i % len(self.methods)])

    def step_one(self, i):
        """
        Perform a single minimization step with member i.
        Returns an (x,y) tuple.
        """
        for retry in [0,1]: # retry once if StopIteration
            try:
                # Step by a single iteration of the minimizer
                self.points[i] = self.minimizers[i].next()
                x = self.points[i]
                break
            except StopIteration:
                # Local optimum, pick a new random point
                x = self.points[i]
                self.restart_one(i)
                # We did no computation for [i] yet in this iteration
                # so make a step right away
                continue

        # Get the value at this point
        y = self.fi.evalfun(x)
        self.values[i] = y
        self.iters[i] += 1
        self.total_steps += 1
        self.data.record(i, self.minimizers[i].minmethod.name, self.iters[i], self.values[i] - self.fi.f.fopt, self.points[i])
        return (x, y)

    def restart_one(self, i):
        """
        Reinitialize a given population member.
        """
        self.points[i] = 10. * np.random.rand(self.fi.dim) - 5.
        self.values[i] = 10e10
        self.minimizers[i] = self._minimizer_make(i)
        self.iters[i] = 0

        #y = self.fi.f.evalfun(self.points[i]) # This is just for the debug print
        #print("#%d reached local optimum %s=%s" % (i, self.points[i], y))
        #time.sleep(1)

    def add(self):
        """
        Add another population member.
        """
        self.points = np.append(self.points, [10. * np.random.rand(self.fi.dim) - 5.], axis = 0)
        self.values = np.append(self.values, [10e10], axis = 0)
        i = len(self.points) - 1
        self.minimizers.append(self._minimizer_make(i))
        self.iters = np.append(self.iters, [0], axis = 0)

        #y = self.fi.f.evalfun(self.points[i]) # This is just for the debug print
        #print("#%d new member %s=%s" % (i, self.points[i], y))
        #time.sleep(1)
        return i


    def end_iter(self):
        """
        Notify the population that a single portfolio iteration has passed.
        This is useful in case we step multiple method instances within
        a single iteration (e.g. in MetaMax).
        """
        self.total_iters += 1
        self.data.end_iter()

    def stop(self):
        for m in self.minimizers:
            m.stop()
