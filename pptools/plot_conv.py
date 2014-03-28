#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot convergence data of a portfolio on a given set of functions.

Usage: plot_conv.py [-o FILE.PDF] PICKLEFILE PLOTTYPE DIM FID...

For now, see the plot_by_type() function for various options regarding
what PLOTTYPE can be.  Get started with fval_by_budget or overview.

("overview" meaning may change over time based on state-of-art plot
types. "overview0" will always refer to the current layout.)

In case of PLOTTYPE "rank_by_budget", FID may be multiple comma-separated
functions that will be averaged.
"""

import os
import re
import sys
import time
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages

# Add the path to bbob_pproc and cocopf
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir, os.path.pardir))

from cocopf.pproc import PortfolioDataSets
import cocopf.pplot as cplot


def get_stratds(pds, strat, dim, fid):
    if strat == 'oracle':
        return pds.oracle((dim, fid))
    elif strat == 'envelope':
        return pds.bestalg((dim, fid))
    else:
        return pds.stratds[strat].dictByDimFunc()[dim][fid][0]


def plot_by_type(pds, ax, plottype, dim, fid):
    if fid.count(',') > 0:
        fid = [int(i) for i in fid.split(',')]
    else:
        fid = int(fid)

    if plottype == "fval_by_budget":
        return cplot.fval_by_budget(ax, pds, dim=dim, funcId=fid)

    elif plottype.startswith("fval2"):
        m = re.match("fval2(.*)_by_budget", plottype)
        if m:
            strat = m.group(1)
            stratds = get_stratds(pds, strat, dim, fid)
            return cplot.fval_by_budget(ax, pds, baseline_ds=stratds, baseline_label=strat, dim=dim, funcId=fid)
        raise ValueError('plottype ' + plottype)

    elif plottype == "rank_by_budget":
        return cplot.rank_by_budget(ax, pds, dim=dim, funcId=fid)

    elif plottype == "ert_by_target":
        return cplot.ert_by_target(ax, pds, dim=dim, funcId=fid)

    elif plottype.startswith("ert2") and plottype.endswith("_by_target"):
        # e.g. ert2mUNIF7_by_target for data relative to mUNIF7
        # ert2oracle_by_target is a special case that gives nice plots!
        m = re.match("ert2(.*)_by_target", plottype)
        if m:
            strat = m.group(1)
            stratds = get_stratds(pds, strat, dim, fid)
            return cplot.ert_by_target(ax, pds, baseline_ds=stratds, baseline_label=strat, dim=dim, funcId=fid)
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
            return cplot.ert_by_ert(ax, pds,
                    baseline1_ds=strat1ds, baseline1_label=strat1,
                    baseline2_ds=strat2ds, baseline2_label=strat2,
                    dim=dim, funcId=fid)
        raise ValueError('plottype ' + plottype)
    raise ValueError('plottype ' + plottype)


def fig_by_type(pds, plottype, dim, fid):
    if plottype == "overview" or plottype == "overview0":
        return fig_overview(pds, dim, fid)

    fig = figure('%s (%s)'%(fid, plottype))
    ax = fig.add_subplot(111)
    plot_by_type(pds, ax, plottype, dim, fid)
    cplot.legend(ax)
    return fig


def fig_overview(pds, dim, fid):
    fig = figure('%s (overview)'%(fid))
    subplots = []
    # Also potentially interesting (but hard to comprehend): ert2oracle_by_ert
    for (i, plottype) in enumerate(['fval_by_budget', 'rank_by_budget', 'fval2oracle_by_budget', 'ert2oracle_by_target']):
        ax = fig.add_subplot(2, 2, 1+i)
        plot_by_type(pds, ax, plottype, dim, fid)
        subplots.append(ax)
    (handles, labels) = subplots[0].get_legend_handles_labels()
    cplot.legend(fig, handles=handles, labels=labels, loc='upper right', ncol=8)
    return fig


if __name__ == "__main__":
    if sys.argv[1] == '-o':
        sys.argv.pop(1)
        pdffile = sys.argv.pop(1)
    else:
        pdffile = None
    picklefile = sys.argv[1]
    plottype = sys.argv[2]
    dim = int(sys.argv[3])

    pds = PortfolioDataSets(pickleFile=picklefile)

    np.seterr(under="ignore")

    if pdffile is None:
        for fid in sys.argv[4:]:
            print 'figure', plottype, dim, fid
            fig = fig_by_type(pds, plottype, dim, fid)
            fig.set_tight_layout(True)
            fig.show()
        show()
    else:
        pdf = PdfPages(pdffile)
        for fid in sys.argv[4:]:
            print 'figure', plottype, dim, fid
            fig = fig_by_type(pds, plottype, dim, fid)
            fig.set_size_inches((11.692, 8.267)) # A4 landscape
            fig.set_tight_layout(True)
            pdf.savefig(fig)
        pdf.close()
