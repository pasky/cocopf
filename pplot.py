#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"Post-plotting" utilities that take post-processed performance data
(as provided by cocopf.pproc) and plot them in a variety of ways.

Their counterpart are several scripts in the pptools/ directory, the goal
here is to have these scripts as thin as possible to enable code reuse.
For the same reason, we plot everything into a given Axes object.

See cocopf.pproc for some philosophical thoughts on portfolio analysis
compared to algorithm analysis done by stock COCO.

    fig = figure('5-11')
    ax = fig.add_subplot(111)
    cocopf.pplot.fval_by_budget(ax, pds, dim=5, funcId=11)
    fig.show()
"""

import sys
import numpy as np
from pylab import *

sys.path.append('.')
import bbob_pproc as bb
import bbob_pproc.genericsettings
import bbob_pproc.pproc as pp
import bbob_pproc.readalign as ra


class GroupByMedian:
    def __call__(self, lst, **kwargs):
        return np.median(lst, **kwargs)
    def __str__(self):
        return 'median'


def _style_thickline(xstyle):
    style = { 'linestyle': 'solid', 'zorder': -1, 'linewidth': 6 }
    style.update(xstyle)
    return style

def _style_algorithm(name, i):
    # Automatic colors are fine, no markers used
    return { 'linestyle': 'dashed' }

def _style_oracle():
    return _style_thickline({ 'color': '#DAFFE4' })

def _style_unifpf():
    return _style_thickline({ 'color': '#D0E4FF' })

def _style_strategy(name, i):
    if name.startswith('mUNIF'):
        return _style_thickline({ 'color': 'wheat' })

    styles = bb.genericsettings.line_styles
    style = styles[i % len(styles)].copy()
    del style['linestyle']

    style['markersize'] = 12.
    style['markeredgewidth'] = 1.5
    style['markerfacecolor'] = 'None'
    style['markeredgecolor'] = style['color']

    style['linestyle'] = 'solid'
    style['zorder'] = 1
    style['linewidth'] = 2
    return style


def _pds_plot_iterator(pds, dim, funcId):
    """
    An iterator that will in turn yield all drawable curves
    in the form of (kind, name, ds, style) tuples (where kind
    is one of 'algorithm', 'oracle', 'unifpf', 'strategy').
    """
    i = 0
    for (algname, ds) in pds.algds_dimfunc((dim, funcId)):
        yield ('algorithm', algname, ds, _style_algorithm(algname, i))
        i += 1
    yield ('oracle', 'oracle', pds.oracle((dim, funcId)), _style_oracle())
    yield ('unifpf', 'eUNIF', pds.unifpf().dictByDimFunc()[dim][funcId][0], _style_unifpf())
    i = 0
    for (stratname, ds) in pds.stratds_dimfunc((dim, funcId)):
        yield ('strategy', stratname, ds, _style_strategy(stratname, i))
        i += 1

def _legend(ax):
    legendfont = matplotlib.font_manager.FontProperties()
    legendfont.set_size('small')
    ax.legend(loc='best', ncol=3, fancybox=True, prop=legendfont)


def _fval_label(baseline_ds, baseline_label):
    if baseline_ds:
        if baseline_label:
            return 'Function Values rel. to ' + baseline_label
        else:
            return 'Function Values (rel.)'
    else:
        return 'Best Function Values'

def fval_by_budget(ax, pds, baseline_ds=None, baseline_label="", dim=None, funcId=None, groupby=None):
    """
    Plot a classic "convergence plot" that shows how the function value
    approaches optimum as time passes, in terms of raw performance.

    groupby is the method of aggregating results of multiple instances --
    a callable, stringable object, GroupByMedian by default.

    By default, raw function values (as difference to optimum) are shown,
    but relative values to some baseline dataset can be shown instead.
    """
    if groupby is None: groupby = GroupByMedian()
    pfsize = len(pds.algds.keys())

    if baseline_ds:
        baseline_budgets = baseline_ds.funvals[:, 0]
        baseline_funvals = groupby(baseline_ds.funvals[:, 1:], axis=1)
        # fvb is matrix with each row being [budget,funval]
        baseline_fvb = np.transpose(np.vstack([baseline_budgets, baseline_funvals]))

    for (kind, name, ds, style) in _pds_plot_iterator(pds, dim, funcId):
        #print name, ds
        budgets = ds.funvals[:, 0]
        funvals = groupby(ds.funvals[:, 1:], axis=1)
        fvb = np.transpose(np.vstack([budgets, funvals]))

        if baseline_ds:
            # Relativize by baseline
            fvba = ra.alignArrayData(ra.VArrayMultiReader([fvb, baseline_fvb]))
            budgets = fvba[:, 0]
            funvals = fvba[:, 1] / fvba[:, 2]

        style['markevery'] = 16
        ax.loglog(budgets, funvals, label=str(groupby)+' '+name, basex=pfsize, **style)
    if baseline_ds:
        ax.set_yticks([1], minor=True)
    ax.set_xlabel('Budget')
    ax.set_ylabel(_fval_label(baseline_ds, baseline_label))
    ax.grid()
    if baseline_ds:
        ax.yaxis.grid(True, which = 'minor')
    _legend(ax)


def _ert_label(baseline_ds, baseline_label):
    if baseline_ds:
        if baseline_label:
            return 'ERT rel. to ' + baseline_label
        else:
            return 'Relative ERT'
    else:
        return 'Absolute ERT'

def ert_by_target(ax, pds, baseline_ds=None, baseline_label="", dim=None, funcId=None):
    """
    Plot a rotated convergence plot.  It is essentially like fval_by_budget(),
    but rotated by 90 degrees, showing how big budget is required to reach
    every target.

    While this is a little less intuitive at first, it allows better judgement
    of performance impact of each strategy.  With fval_by_budget(), performance
    change is represented by a curve phase shift, while in ert_by_target(),
    it simply translates position on the y axis.

    By default, absolute ERT is shown, but relative values to some baseline
    dataset can be shown instead.
    """
    pfsize = len(pds.algds.keys())

    runlengths = 10**np.linspace(0, np.log10(pds.maxevals((dim, funcId))), num=500)
    target_values = pp.RunlengthBasedTargetValues(runlengths,
            reference_data=pds.bestalg(None), force_different_targets_factor=10**0.004)
    targets = target_values((funcId, dim))

    if baseline_ds:
        baseline_fevs = np.array(baseline_ds.detERT(targets))

    for (kind, name, ds, style) in _pds_plot_iterator(pds, dim, funcId):
        #print name, ds
        fevs = ds.detERT(targets)
        if baseline_ds:
            fevs /= baseline_fevs
        style['markevery'] = 64
        ax.loglog(targets, fevs, label=name, basey=pfsize, **style)
    ax.grid()
    ax.set_xlim(10**2, 10**(np.log10(targets[-1])-0.2))
    ax.set_xlabel('Function Value Targets')
    ax.set_ylabel(_ert_label(baseline_ds, baseline_label))
    _legend(ax)


def ert_by_ert(ax, pds, baseline1_ds=None, baseline1_label="", baseline2_ds=None, baseline2_label="", dim=None, funcId=None):
    """
    Plot the evolution of relative ERT based on changing absolute ERT.
    It's not clear whether this will eventually be useful at all, but it
    offers another perspective that might aid some analysis.
    """
    pfsize = len(pds.algds.keys())

    runlengths = 10**np.linspace(0, np.log10(pds.maxevals((dim, funcId))), num=500)
    target_values = pp.RunlengthBasedTargetValues(runlengths,
            reference_data=pds.bestalg(None), force_different_targets_factor=10**0.004)
    targets = target_values((funcId, dim))

    if baseline1_ds:
        baseline1_fevs = np.array(baseline1_ds.detERT(targets))
    if baseline2_ds:
        baseline2_fevs = np.array(baseline2_ds.detERT(targets))

    for (kind, name, ds, style) in _pds_plot_iterator(pds, dim, funcId):
        #print name, ds
        fevs1 = ds.detERT(targets)
        if baseline1_ds:
            fevs1 /= baseline1_fevs
        fevs2 = ds.detERT(targets)
        if baseline2_ds:
            fevs2 /= baseline2_fevs

        infsx = np.nonzero(fevs1 == inf)
        infs = infsx[0]
        if np.size(infs) > 0:
            #print infs
            fevs1 = fevs1[:infs[0]-1]
            fevs2 = fevs2[:infs[0]-1]

        print name, fevs1, fevs2
        style['markevery'] = 64
        ax.loglog(fevs2, fevs1, label=name, basex=pfsize, basey=pfsize, **style)
    ax.grid()
    ax.set_xlim(0, runlengths[-1])
    ax.set_ylabel(_ert_label(baseline1_ds, baseline1_label))
    ax.set_xlabel(_ert_label(baseline2_ds, baseline2_label))
    _legend(ax)
