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

def get_stratds(pds, strat, dim, fid):
    if strat == 'oracle':
        return pds.oracle((dim, fid))
    else:
        return pds.stratds[strat].dictByDimFunc()[dim][fid][0]

def plot_by_type(pds, ax, plottype, dim, fid):
    if plottype == "fval_by_budget":
        cplot.fval_by_budget(ax, pds, dim=dim, funcId=fid)
    elif plottype.startswith("fval2"):
        m = re.match("fval2(.*)_by_budget", plottype)
        if m:
            strat = m.group(1)
            stratds = get_stratds(pds, strat, dim, fid)
            cplot.fval_by_budget(ax, pds, baseline_ds=stratds, baseline_label=strat, dim=dim, funcId=fid)
        else:
            raise ValueError('plottype ' + plottype)
    elif plottype == "ert_by_target":
        cplot.ert_by_target(ax, pds, dim=dim, funcId=fid)
    elif plottype.startswith("ert2") and plottype.endswith("_by_target"):
        # e.g. ert2mUNIF7_by_target for data relative to mUNIF7
        # ert2oracle_by_target is a special case that gives nice plots!
        m = re.match("ert2(.*)_by_target", plottype)
        if m:
            strat = m.group(1)
            stratds = get_stratds(pds, strat, dim, fid)
            cplot.ert_by_target(ax, pds, baseline_ds=stratds, baseline_label=strat, dim=dim, funcId=fid)
        else:
            raise ValueError('plottype ' + plottype)
    elif plottype.startswith("ert2"):
        # e.g. ert2mUNIF7_by_ert2oracle
        m = re.match("ert2(.*)_by_ert(?:2(.*))?", plottype)
        if m:
            strat1 = m.group(1)
            strat1ds = get_stratds(pds, strat1, dim, fid)
            strat2 = m.group(2)
            if strat2 is not None:
                strat2ds = get_stratds(pds, strat2, dim, fid)
            else:
                strat2 = ''
                strat2ds = None
            cplot.ert_by_ert(ax, pds,
                    baseline1_ds=strat1ds, baseline1_label=strat1,
                    baseline2_ds=strat2ds, baseline2_label=strat2,
                    dim=dim, funcId=fid)
        else:
            raise ValueError('plottype ' + plottype)
    else:
        raise ValueError('plottype ' + plottype)

if __name__ == "__main__":
    picklefile = sys.argv[1]
    plottype = sys.argv[2]
    dim = int(sys.argv[3])

    pds = PortfolioDataSets(pickleFile=picklefile)

    np.seterr(under="ignore")

    for fid in sys.argv[4:]:
        fid = int(fid)
        fig = figure('r%d'%fid)
        ax = fig.add_subplot(111)
        plot_by_type(pds, ax, plottype, dim, fid)
        fig.show()

    show()
