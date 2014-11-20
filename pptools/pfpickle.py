#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load performance data on a portfolio of given set of algorithms and
given set of strategies, compute the oracle and uniform portfolio
strategy and pickle the resulting portfolio data so that it can
be quickly used for various plotting activities.

Usage: pickle.py PICKLEFILE ALGORITHM... -- STRATEGY...

Note that if PICKLEFILE already exists, the given datasets are appended
to the data file.  If you specify datasets by path, only the basename
is used as the algorithm/strategy name in portfolio.
"""

import glob
import os
import sys

# Add the path to bbob_pproc and cocopf
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir, os.path.pardir))

import bbob_pproc as bb
from cocopf.pproc import PortfolioDataSets

picklefile = sys.argv[1]

dashidx = sys.argv.index('--')
algs = sys.argv[2:dashidx]
strats = sys.argv[dashidx+1:]

if os.path.exists(picklefile) or os.path.exists(picklefile + '.gz'):
    pds = PortfolioDataSets(pickleFile=picklefile)
else:
    pds = PortfolioDataSets()

newalg = False
for apath in algs:
    aname = os.path.basename(os.path.normpath(apath))
    if aname not in pds.algds:
        print apath
        pds.add_algorithm(aname, bb.load(glob.glob(apath+'/bbobexp_f*.info') + glob.glob(apath+'/*/bbobexp_f*.info')))
        newalg = True
    else:
        print apath, 'already imported, skipping...'
for spath in strats:
    sname = os.path.basename(os.path.normpath(spath))
    if sname not in pds.stratds:
        print spath
        pds.add_strategy(sname, bb.load(glob.glob(spath+'/bbobexp_f*.info') + glob.glob(spath+'/*/bbobexp_f*.info')))
    else:
        print spath, 'already imported, skipping...'

if newalg:
    # Force regen of "static strategies", the set of algorithms changed
    pds._bestalg = None
    pds._unifpf = None

print "bestalg"
pds.bestalg(None)
print "unifpf"
pds.unifpf()

# TODO: Pickle to tmp file and rename()?
print "Pickling..."
pds.pickle(picklefile)
