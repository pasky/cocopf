#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot convergence data of a portfolio on a given set of functions.

Usage: plot_conv.py PICKLEFILE PLOTTYPE DIM FID...

PLOTTYPE can be just fval_by_budget for now.
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

from cocopf.pproc import PortfolioDataSets
import cocopf.pplot as cplot

picklefile = sys.argv[1]
plottype = sys.argv[2]
dim = int(sys.argv[3])

pds = PortfolioDataSets(pickleFile=picklefile)

np.seterr(under="ignore")

def get_stratds(pds, strat):
    if strat == 'oracle':
        return pds.oracle((dim, fid))
    else:
        return pds.stratds[strat].dictByDimFunc()[dim][fid][0]

for fid in sys.argv[4:]:
    fid = int(fid)
    fig = figure('r%d'%fid)
    if plottype == "fval_by_budget":
        ax = fig.add_subplot(111)
        cplot.fval_by_budget(ax, pds, dim=dim, funcId=fid)
    elif plottype.startswith("fval2"):
        m = re.match("fval2(.*)_by_budget", plottype)
        if m:
            strat = m.group(1)
            stratds = get_stratds(pds, strat)
            ax = fig.add_subplot(111)
            cplot.fval_by_budget(ax, pds, baseline_ds=stratds, baseline_label=strat, dim=dim, funcId=fid)
        else:
            raise ValueError('plottype ' + plottype)
    elif plottype == "ert_by_target":
        ax = fig.add_subplot(111)
        cplot.ert_by_target(ax, pds, dim=dim, funcId=fid)
    elif plottype.startswith("ert2") and plottype.endswith("_by_target"):
        # e.g. ert2mUNIF7_by_target for data relative to mUNIF7
        # ert2oracle_by_target is a special case that gives nice plots!
        m = re.match("ert2(.*)_by_target", plottype)
        if m:
            strat = m.group(1)
            stratds = get_stratds(pds, strat)
            ax = fig.add_subplot(111)
            cplot.ert_by_target(ax, pds, baseline_ds=stratds, baseline_label=strat, dim=dim, funcId=fid)
        else:
            raise ValueError('plottype ' + plottype)
    else:
        raise ValueError('plottype ' + plottype)
    fig.show()

show()
