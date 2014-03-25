#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot convergence data of a portfolio on a given set of functions.

Usage: plot_conv.py PICKLEFILE PLOTTYPE DIM FID...

PLOTTYPE can be just fval_by_budget for now.
"""

import os
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

for fid in sys.argv[4:]:
    fid = int(fid)
    fig = figure('r%d'%fid)
    ax = fig.add_subplot(111)
    if plottype == "fval_by_budget":
        cplot.fval_by_budget(ax, pds, dim=dim, funcId=fid)
    fig.show()

show()
