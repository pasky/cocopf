#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
List a variety of values determined from "final" (converged) state
of the algorithms and strategies on a given set of functions.

Usage: table_final.py PICKLEFILE VALTYPE DIM FID...

VALTYPE can be 'rank' (average rank) and 'slowdown2<something>'
(average log-slowdown compared to <something>).  VALTYPE can
end with trailing @ to determine that std, median should be also
printed out.

In case of slowdown2<something>, an additional number is printed,
which is a portion of functions solved within the total available
budget, relative to the <something>; 1.0 means all functions that
were solved by the <something> were also solved by that algorithm.
"""

import os
import re
import sys
import time
from pylab import *

# Add the path to bbob_pproc and cocopf
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir, os.path.pardir))

from cocopf.pproc import PortfolioDataSets, resolve_fid
import cocopf.pplot as cplot
import bbob_pproc.readalign as ra


def get_stratds(pds, strat, dim, fid):
    if strat == 'oracle':
        return pds.oracle((dim, fid))
    elif strat == 'envelope':
        return pds.bestalg((dim, fid))
    else:
        return pds.stratds[strat].dictByDimFunc()[dim][fid][0]

def _pds_table_iterator(pds, dim, funcId):
    i = 0
    for (algname, ds) in pds.algds_dimfunc((dim, funcId)):
        yield ('algorithm', algname, ds)
        i += 1
    i = 0
    for (stratname, ds) in pds.stratds_dimfunc((dim, funcId)):
        yield ('strategy', stratname, ds)
        i += 1


def val_slowdown(pds, baseline_name, dim=None, funcId=None, groupby=None):
    if groupby is None: groupby = np.median
    pfsize = len(pds.algds.keys())

    try:
        funcId = iter(funcId)
    except TypeError:
        funcId = [funcId]

    avals = [list() for _ in _pds_table_iterator(pds, dim, 1)]
    # print str(len(avals)), str(len(pds.algds.keys()) + len(pds.stratds.keys()))
    baseline_solved = 0
    for fid in funcId:
        print 'fid:' + str(fid)
        baseline_ds = get_stratds(pds, baseline_name, dim, fid)
        baseline_conv_fevs = groupby(baseline_ds.detEvals([10**-8]))
        baseline_conv_lfevs = np.log(baseline_conv_fevs) / np.log(pfsize)

        if not np.isnan(baseline_conv_fevs):
            baseline_solved += 1

        i = 0
        for (kind, name, ds) in _pds_table_iterator(pds, dim, fid):
            conv_fevs = groupby(ds.detEvals([10**-8]))
            if np.isnan(baseline_conv_fevs) or np.isnan(conv_fevs):
                print name + ' \infty'
                i += 1
                continue
            conv_lfevs = np.log(conv_fevs) / np.log(pfsize)
            val = conv_fevs / baseline_conv_fevs
            avals[i].append(val)
            print name + ' ' + str(val) + ', ' + str(conv_lfevs) + '/' + str(baseline_conv_lfevs)
            i += 1

    for i in range(len(avals)):
        # print str(i), str(avals[i])
        if avals[i] == []:
            avals[i] = (np.inf, np.inf, np.inf, 0)
        else:
            avals[i] = (np.average(avals[i]), np.std(avals[i]), np.median(avals[i]), float(len(avals[i])) / baseline_solved)
        # print '>', str(avals[i])
    return avals


def val_rank(pds, dim=None, funcId=None, groupby=None):
    if groupby is None: groupby = np.median
    pfsize = len(pds.algds.keys())

    try: # funcId is array?
        # _pds_plot_iterator[] uses funcId only for things we don't care for
        fakeFuncId = funcId[0]

        manyranking = np.array([pds.ranking((dim, i), groupby) for i in funcId])
        rankcount = np.shape(manyranking[0])[1] - 1
        amanyranking = ra.alignArrayData(ra.VArrayMultiReader(manyranking))
        budget = amanyranking[:,0]
        rankings = np.hsplit(amanyranking[:,1:], len(funcId))
        avgranking = np.average(rankings, axis=0)
        ranking = np.vstack([budget, avgranking.T]).T

    except TypeError: # funcId is scalar
        fakeFuncId = funcId
        ranking = pds.ranking((dim, funcId), groupby)

    ranks = []
    i = 0
    for (kind, name, ds) in _pds_table_iterator(pds, dim, fakeFuncId):
        ranks.append(tuple(ranking[-1,1+i]))
        i += 1
    return ranks


def val_by_type(pds, valtype, dim, fid):
    fid = resolve_fid(fid)

    if valtype == "rank":
        return val_rank(pds, dim=dim, funcId=fid)

    elif valtype.startswith("slowdown2"):
        m = re.match("slowdown2(.*)", valtype)
        if m:
            strat = m.group(1)
            return val_slowdown(pds, baseline_name=strat, dim=dim, funcId=fid)
        raise ValueError('valtype ' + valtype)

    raise ValueError('valtype ' + valtype)


if __name__ == "__main__":
    picklefile = sys.argv[1]
    valtype = sys.argv[2]
    dim = int(sys.argv[3])

    if valtype.endswith('@'):
        valtype = valtype[0:-1]
        showStat = True
    else:
        showStat = False

    pds = PortfolioDataSets(pickleFile=picklefile)

    names = [name for (kind, name, ds) in _pds_table_iterator(pds, dim, 1)]

    values = np.array([val_by_type(pds, valtype, dim, fid) for fid in sys.argv[4:]])
    # print names
    # print values

    def printval(v):
        if v[0] == '\\infty':
            return v[0]
        elif showStat:
            # avg std median
            return '$%.1f^{\pm %.1f}_{|\, %.1f}$ | %.3f' % (v[0], v[1], v[2], v[3])
        else:
            return '%.1f | %.3f' % (v[0], v[3])

    print ' & '.join(['Solver'] + sys.argv[4:]) + ' \\\\'
    for i in range(len(names)):
        print ' & '.join([names[i]] + [printval(v) for v in values[:,i]]) + ' \\\\'

    #print ' & '.join(['Functions'] + names) + ' \\\\'
    #for fid in sys.argv[4:]:
    #    values = val_by_type(pds, valtype, dim, fid)
    #    print '&'.join([fid] + [str(i) for i in values) + ' \\\\'
