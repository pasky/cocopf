#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A generic manager of iteratively minimized population that will
keep track of solutions and respective minimizer states.

Solution search progress (in respect to method population) is
recorded to an .mdat file.

Minimizer steps can be also *replayed* - when instantiating
a minimizer, we check if an existing mdat file doesn't exist
already; if so, we report that and then simply yield the steps
recorded in this mdat file instead of actually running the
algorithm in step_one().  To generate a replayable mdat file,
run a job like:

    parallel -u --gnu env BBOB_FUNSTRIPES={1}%3 BBOB_INSTRIPES={2}%5 experiments/pop-uniform.py BFGS 1 100000 ::: `seq 0 2` ::: `seq 0 4`
"""

from __future__ import print_function

import math
import sys
import warnings

import numpy as np

from cocopf.minstep import MinimizeStepping
from cocopf.methods import SteppingData


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
        self.values = np.zeros(self.K) + 1e10
        # A population of minimizers
        self.minimizers = [self._minimizer_make(i) for i in range(self.K)]
        # A population of iteration counters
        self.iters = np.zeros(self.K, dtype = np.int)

        self.total_steps = 0
        self.total_iters = 0
        self.data = SteppingData(self.fi)

    def _minimizer_make(self, i):
        try:
            return RecordedStepping(self.fi, self.methods[i % len(self.methods)])
        except IOError:
            warnings.simplefilter("ignore") # ignore warnings about unused/ignored options
            return MinimizeStepping(self.fi.f.evalfun, self.points[i],
                                    self.methods[i % len(self.methods)])

    def _step_one_call(self, i):
        base_nfevs = self.fi.f.evaluations
        best_x = None
        best_y = None
        while self.fi.f.evaluations < base_nfevs + 100:
            try:
                # Step by a single iteration of the minimizer
                self.points[i] = self.minimizers[i].next()
                x = self.points[i]
            except StopIteration:
                # Local optimum, pick a new random point
                x = self.points[i]
                self.restart_one(i)
                # We did no computation for [i] yet in this iteration
                # so just retry
                continue

            # Get the value at this point, and consider it the canonical
            # iteration value if it's best so far in this step
            y = self.fi.evalfun(x)
            if best_y is None or best_y > y:
                best_x = x
                best_y = y

        self.values[i] = best_y
        self.iters[i] += 1
        self.total_steps += 1
        self.data.record(i, self.minimizers[i].minmethod.name, self.iters[i], self.values[i] - self.fi.f.fopt, self.points[i])
        return (best_x, best_y)

    def _step_one_replay(self, i):
        orig_y = self.values[i]
        (nfevs, y) = self.minimizers[i].replay_step()
        # print(nfevs, y)
        self.values[i] = y
        self.iters[i] += 1
        self.total_steps += 1
        self.data.record(i, self.minimizers[i].minmethod.name, self.iters[i], self.values[i] - self.fi.f.fopt, None)

        # Ok, now we need to "evaluate" the function appropriate
        # number of times to get stuff logged; we exp-interpolate
        # between original and new value over the course of 1..nfevs
        base_y = min(orig_y, y) - self.fi.f.precision
        precshift = 1
        while base_y == orig_y or base_y == y:
            # It is possible that if base_y is a very big number,
            # the addition is too small to show up
            base_y -= self.fi.f.precision * (2 ** precshift)
            precshift += 1
        # print(self.minimizers[i].minmethod.name, self.nfevs[i], nfevs, self.fi.dim, self.fi.f.evaluations)
        yvals = np.exp(np.linspace(np.log(orig_y - base_y), np.log(y - base_y), nfevs)) + base_y
        # ...cutting off early if we pass the convergence criterion.
        yvals_trunc = []
        for yval in yvals:
            yvals_trunc.append(yval)
            if yval - self.fi.f.ftarget < self.fi.f.precision:
                break

        _fun_evalfull = self.fi.f._fun_evalfull
        self.fi.f._fun_evalfull = lambda x: (yvals_trunc, yvals_trunc)
        self.fi.f.evalfun(np.ones((len(yvals_trunc), self.fi.dim)))
        self.fi.f._fun_evalfull = _fun_evalfull
        # print(self.minimizers[i].minmethod.name, self.nfevs[i], nfevs, self.fi.dim, self.fi.f.evaluations)

        return (None, y)

    def step_one(self, i):
        """
        Perform a minimization step with member i.  This step takes
        at least one algorithm iteration, but also spends at least
        100 nfevs on the step (i.e. makes multiple iterations in that case).
        Returns an (x,y) tuple.

        Note that x may be None (e.g. during a replay).
        """
        if isinstance(self.minimizers[i], MinimizeStepping):
            return self._step_one_call(i)
        else:
            assert isinstance(self.minimizers[i], RecordedStepping)
            return self._step_one_replay(i)

    def restart_one(self, i):
        """
        Reinitialize a given population member.
        """
        self.points[i] = 10. * np.random.rand(self.fi.dim) - 5.
        self.values[i] = 1e10
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
        self.values = np.append(self.values, [1e10], axis = 0)
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


class RecordedStepping:
    def __init__(self, fi, minmethod):
        self.fi = fi
        self.minmethod = minmethod
        self.s = 0
        self.steps = []
        self.warn_end = True
        self._load()

    def _data_location(self):
        """
        Return a tuple to the mdat file path and number of record within
        that file.
        """
        # XXX: We hardcode disabled BBOB_FULLDIM, instance list and
        # BBOB_INSTRIPES %5
        instances = range(1, 6) + range(31, 41)
        inidx = instances.index(self.fi.iinstance)
        strmaxfev = '1e%d' % int(math.log10(self.fi.maxfunevals / self.fi.dim))
        datapath = 'data-%s/pUNIF1_%s/%d%%5' % (strmaxfev, self.minmethod.name, inidx % 5)
        mdatpath = '%s/data_f%d/bbobexp_f%d_DIM%d.mdat' % (datapath, self.fi.fun_id, self.fi.fun_id, self.fi.dim)
        return (mdatpath, int(inidx / 5))

    def _load(self):
        # If the mdat file does not exist, we throw an exception which
        # is caught in Population._minimizer_make() and triggers a
        # normal MinimizeStepping instantiation.
        (datapath, recno) = self._data_location()
        datafile = open(datapath, 'r')
        print('Loading replay %d from %s' % (recno, datapath), file=sys.stderr)
        recno += 1
        base_nfevs = 0
        for line in datafile:
            if line.startswith('%'):
                recno -= 1
                continue
            if recno < 0:
                break
            if recno > 0:
                continue
            # This is the right instance!
            # 2464 19 0 L-BFGS-B 20 +6.136498598e-07 +6.134980595e-07
            # 2574 20 0 L-BFGS-B 21 +4.228133009e+00
            items = line.split()
            try:
                nfevs = int(items[0])
                y = float(items[5]) + self.fi.f.fopt
            except ValueError:
                # 37975 96 0 BFGS 1 +1.089795774e-07 +1.089794353e-07
                # 38379 97 0 BFGS 1       +nan
                continue  # ignore such steps
            self.steps.append((nfevs - base_nfevs, y))
            base_nfevs = nfevs
        # print('\t%d steps loaded, %d nfevs total' % (len(self.steps), nfevs), file=sys.stderr)

    def replay_step(self):
        """
        Return (nfevs, y) state on the next step.
        """
        if self.s > len(self.steps) - 1:
            if self.warn_end:
                print('Warning: %d,%d %s replay hist a stop at %d #steps, %d nfevs' %
                      (self.fi.fun_id, self.fi.iinstance, self.minmethod.name, self.s, self.steps[-1][0]),
                      file=sys.stderr)
                self.warn_end = False
            # ... and keep returning the last step recorded
            self.s = len(self.steps) - 1
        step = self.steps[self.s]
        self.s += 1
        return step

    def stop(self):
        pass
