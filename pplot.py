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
import bbob_pproc.pproc as pp
import bbob_pproc.genericsettings


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
    style['markevery'] = 16
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
    for (algname, dset) in pds.algds.iteritems():
        ds = dset.dictByDimFunc()[dim][funcId][0]
        yield ('algorithm', algname, ds, _style_algorithm(algname, i))
        i += 1
    yield ('oracle', 'oracle', pds.oracle((dim, funcId)), _style_oracle())
    yield ('unifpf', 'eUNIF', pds.unifpf().dictByDimFunc()[dim][funcId][0], _style_unifpf())
    i = 0
    for (stratname, dset) in pds.stratds.iteritems():
        ds = dset.dictByDimFunc()[dim][funcId][0]
        yield ('strategy', stratname, ds, _style_strategy(stratname, i))
        i += 1

def _legend(ax):
    legendfont = matplotlib.font_manager.FontProperties()
    legendfont.set_size('small')
    ax.legend(loc='best', ncol=3, fancybox=True, prop=legendfont)


def fval_by_budget(ax, pds, dim=None, funcId=None, groupby=None):
    """
    Plot a classic "convergence plot" that shows how the function value
    approaches optimum as time passes, in terms of raw performance.

    groupby is the method of aggregating results of multiple instances --
    a callable, stringable object, GroupByMedian by default.
    """
    if groupby is None: groupby = GroupByMedian()
    pfsize = len(pds.algds.keys())

    for (kind, name, ds, style) in _pds_plot_iterator(pds, dim, funcId):
        #print name, ds
        budgets = ds.funvals[:, 0]
        funvals = ds.funvals[:, 1:]
        ax.loglog(budgets, groupby(funvals, axis=1), label=str(groupby)+' '+name, basex=pfsize, **style)
    ax.grid()
    ax.set_xlabel('Budget')
    ax.set_ylabel('Best Function Values')
    _legend(ax)
