#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common routines that manage experimental setup of a single experiment,
greatly simplifying and reducing code duplication in per-experiment
executables.

This is not even specific to COCOpf. It is inspired by the BBOB
example experiment.
"""

import string
import os
import sys
import time
import math

import numpy as np
import numpy.random as nr

sys.path.append('.')
import fgeneric
import bbobbenchmarks


class FInstance:
    def __init__(self, f, dim, fun_id, iinstance, maxfunevals):
        """
        A descriptor of a single function instance.
        """
        self.f = f
        self.dim = dim
        self.fun_id = fun_id
        self.iinstance = iinstance
        self.maxfunevals = maxfunevals


class Experiment:
    def __init__(self, maxfev, shortname, comments):
        """
        Create an experiment description and initialize the experiment.

        If the environment variable $BBOB_FULLDIM is set to 1, a full
        standard set of dimensions is evaluated.  But by default, only
        dimensions 2, 5 and 20 are evaluated, which is useful for
        templateBBOBmany ECRF grahs; this is useful for comparison
        during non-final experiments, especially with 10^4 or more
        function evaluations.
        """
        self.maxfev = maxfev
        strmaxfev = '10e%d' % int(math.log10(maxfev))
        self.shortname = shortname

        if bool(os.environ.get('BBOB_FULLDIM')):
            self.dimensions = (2, 3, 5, 10, 20, 40) # Full settings
        else:
            self.dimensions = (2, 5, 20) # Just bootstrap + BBOBmany ECRF
        self.function_ids = bbobbenchmarks.nfreeIDs
        self.instances = range(1, 6) + range(31, 41)

        self.t0 = time.time()
        np.random.seed(int(self.t0))

        comments += ', FEV=%s*dim' % maxfev
        self.f = fgeneric.LoggingFunction(datapath = 'data-%s/%s'%(strmaxfev,shortname),
                algid = shortname, comments = comments)

    def finstances(self):
        """
        An iterator that generates all function instances
        to be evaluated.
        """
        for dim in self.dimensions:
            maxfunevals = self.maxfev * dim
            fevs = np.zeros(len(self.function_ids) * len(self.instances))
            fevs_i = 0
            for fun_id in self.function_ids:
                for iinstance in self.instances:
                    self.f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))
                    yield FInstance(self.f, dim, fun_id, iinstance, maxfunevals)

                    fevs[fevs_i] = self.f.evaluations
                    fevs_i += 1

                print '  % -12s      date and time: %s' % (self.shortname, time.asctime())

            overshoot_idx = np.argmax(fevs)
            overshoot = fevs[overshoot_idx]
            overshoot_f = self.function_ids[int(overshoot_idx / len(self.instances))]
            overshoot_i = self.instances[overshoot_idx % len(self.instances)]
            print('---- % -12s dimension %d-D done ----  (max FEV: %d/%d in f%d:%d)' %
                    (self.shortname, dim, overshoot, maxfunevals, overshoot_f, overshoot_i))

    def freport(self, finstance, note):
        print('  % -12s  f%d in %d-D, instance %d: FEs=%d/%d%s, '
              'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
              % (self.shortname, finstance.fun_id, finstance.dim, finstance.iinstance,
                 self.f.evaluations, finstance.maxfunevals, note,
                 self.f.fbest - self.f.ftarget, (time.time()-self.t0)/60./60.))
