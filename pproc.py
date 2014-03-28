#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common routines providing post-processing utility.  They serve mainly
as data-feeding backends for cocopf.pplot routines.  The goal is to
have all the number crunching in this module (also usable for tables
generation, machine learning etc.) while keeping all presentation
in pplot.

Their counterpart
are several scripts in the pptools/ directory, the goal here is to
have these scripts as thin as possible to enable code reuse.

We use some COCO post-processing modules, but our focus here is shifted
from raw performance (in terms of FEVs needed for convergence to
a given target) to relative performance - it is not so important for
us to see how a portfolio converges for a particular (instance,target)
pair, but rather how the convergence speed changes relative to individual
algorithms (and some reference strategies).  Also, uncertainities are
compounded with portfolios and we thus further de-emphasize statistical
estimates of unobserved ERTs.
"""

import itertools
import numpy as np
import pickle, gzip
import scipy.stats as ss
import sys

sys.path.append('.')
import bbob_pproc as bb
import bbob_pproc.algportfolio
import bbob_pproc.bestalg
import bbob_pproc.readalign as ra


class PortfolioDataSets:
    """
    This container class holds data sets of a portfolio in terms
    of both individual algorithms and a variety of strategies,
    both preloaded and generated.

    Public class attributes:

      - *algds* -- dict of per-algorithm DataSetLists
      - *stratds* -- dict of per-strategy DataSetLists

    Use *algds* and *stratds* only read-only or things may get out of sync.

    Example:

        algorithms = {}
        for a in ['CMA', 'Powell', 'SLSQP']:
            algorithms[a] = bb.load(glob.glob('ppdata/'+a+'/bbobexp_f*.info'))
        strategies = {}
        for s in ['mUNIF', 'mEG50']:
            strategies[s] = bb.load(glob.glob('ppdata/'+s+'/bbobexp_f*.info'))
        pds = PortfolioDataSets(algorithms, strategies)

    (...continued in the cocopf.pplot examples.)
    """
    def __init__(self, algorithms={}, strategies={}, pickleFile=None):
        """
        Initialize the portfolio dataset container; pass dicts
        of DataSetLists (e.g. returned by bb.load()). Alternatively,
        we can unpickle an existing portfolio.
        """
        if pickleFile is None:
            self.algds = algorithms
            self.stratds = strategies
            self._bestalg = None
            self._unifpf = None
        else:
            if pickleFile.find('.gz') < 0:
                pickleFile += '.gz'
            with gzip.open(pickleFile) as f:
                entry = pickle.load(f)
            self.algds = entry.algds
            self.stratds = entry.stratds
            self._bestalg = entry._bestalg
            self._unifpf = entry._unifpf

    def add_algorithm(self, name, ds):
        """
        Add another algorithm.
        """
        self.algds[name] = ds
        self._bestalg = None
        self._unfipf = None

    def add_strategy(self, name, ds):
        """
        Add another strategy.
        """
        self.stratds[name] = ds

    def bestalg(self, dimfun):
        """
        A BestAlgSet from all algorithms (like a DataSetList).
        Contains the best performance for each (instance,target) pair
        for a given dimfun=(dim,fun) tuple.

        The BestAlgSet represents a "left envelope" of lowest ERTs over
        all (function, target) pairs.  C.f. the oracle() method.

        Also, avoid depending on algbestfinalfunvals and similar attributes,
        which take only reached targets but not required budgets into account.
        """
        if self._bestalg is None:
            self._bestalg = bb.bestalg.generate(self.algds)
        return self._bestalg[dimfun] if dimfun is not None else self._bestalg

    def oracle(self, dimfun):
        """
        An oracle strategy is such that knows "in advance" which algorithm
        performs best on a given function and runs that one. This method
        returns a DataSetList of such a strategy - simply a DataSetList
        of the algorithm that reaches the best target fastest for the given
        dimfun=(dim,fun) tuple.

        N.B. the *bestalg* attribute is _better_ than even an oracle strategy,
        since it can switch between algorithms "for free" while the target
        moves ahead.
        """
        (dim, funcId) = dimfun
        # What is the best reachable target?
        bestfinalfunval = max(np.median(self.bestalg(dimfun).bestfinalfunvals), 1e-8)
        # How fast do various algorithms reach it?
        nametargetERT = [(name, ds.detERT([bestfinalfunval])) for (name, ds) in self.algds_dimfunc(dimfun)]
        # Pick the fastest!
        (name, targetERT) = min(nametargetERT, key = lambda k: k[1])
        return self.algds[name].dictByDimFunc()[dim][funcId][0]

    def unifpf(self):
        """
        A "uniform portfolio" strategy is generated using the COCO's
        bb.algportfolio machinery, which represents a retroactively computed
        uniform strategy on the function evaluation level; i.e. it "stops"
        each algorithm after each function evaluation.
        """
        if self._unifpf is None:
            self._unifpf = bb.algportfolio.build(self.algds)
        return self._unifpf

    def pickle(self, pickleFile):
        """
        Pickle the current portfolio dataset.  The file is automatically
        gzipped.

        Currently, this is aimed at fastest load times (to play with plotting
        stuff in different ways), therefore even redundant information like
        best and eUNIF are stored.  An alternative mode pickling only bare
        minimum (aimed at redistribution) will come in the future.
        """
        if pickleFile.find('.gz') < 0:
            pickleFile += '.gz'
        with gzip.open(pickleFile, 'w') as f:
            pickle.dump(self, f)

    def algds_dimfunc(self, dimfun):
        """
        Return an iterable of (name, DataSet) tuples corresponding
        to the given dimfun.
        """
        (dim, funcId) = dimfun
        for (algname, dset) in self.algds.iteritems():
            yield (algname, dset.dictByDimFunc()[dim][funcId][0])

    def stratds_dimfunc(self, dimfun):
        """
        Return an iterable of (name, DataSet) tuples corresponding
        to the given dimfun.
        """
        (dim, funcId) = dimfun
        for (stratname, dset) in self.stratds.iteritems():
            yield (stratname, dset.dictByDimFunc()[dim][funcId][0])

    def maxevals(self, dimfun):
        """
        Return the maximum nominal budget across all algorithms; typically,
        something around 1e5 if you run your experiments with budget 100000.
        """
        evals = [np.median(ds.maxevals) for (name, ds) in self.algds_dimfunc(dimfun)]
        return max(evals) / dimfun[0]

    def ranking(self, dimfun, groupby, ftarget=10**-8):
        """
        Produce a set of function evaluation ranks over all algorithms
        and strategies.

        Returns a set of rows where each row contains a budget as first
        element and ranks for individual algorithms and strategies
        as the second element (in the order of the strategies in the
        output of algds_dimfunc(), and stratds_dimfunc() respectively).

        The ranks are always computed based on function values after
        a particular budget.  If multiple algorithms reach ftarget,
        they are ranked by the order in which they did.
        """
        nameds = list(itertools.chain(self.algds_dimfunc(dimfun), self.stratds_dimfunc(dimfun)))
        count = len(nameds)

        # Produce "fv" items, one per dataset, containing single function value
        # for each budget
        fvset = []
        for (name, ds) in nameds:
            budgets = ds.funvals[:,0]
            f1vals = np.maximum(groupby(ds.funvals[:, 1:], axis=1), ftarget)
            fv = np.transpose(np.vstack([budgets, f1vals]))
            fvset.append(fv)

        # Align the "fv" items by budget and merge them
        fva = ra.alignArrayData(ra.VArrayMultiReader(fvset))
        budgets = fva[:,0]

        # Assign function values and rank them
        # However, we want to resolve eventual ties by ranking first
        # converging function first. So we do a trick and rewrite ftarget
        # values in increasing convergence sort order.
        values = fva[:,1:].copy()
        firstconv = np.ones(count) * (np.size(budgets)+1) # runlength+1 is default
        for i in range(count): # XXX: drop the loop
            try:
                firstconv[i] = np.nonzero(values[:,i] == ftarget)[0][0]
            except IndexError:
                continue # no rewriting needed
        firstconvranks = ss.mstats.rankdata(firstconv)
        for i in range(count):
            r = firstconvranks[i]
            values[firstconv[i]:, i] = ftarget - (1-r/count)*ftarget

        ranks = ss.mstats.rankdata(values, axis=1)

        return np.transpose(np.vstack([budgets, ranks.T]))
