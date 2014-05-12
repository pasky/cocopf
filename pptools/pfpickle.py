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

for apath in algs:
    print apath
    aname = os.path.basename(os.path.normpath(apath))
    pds.add_algorithm(aname, bb.load(glob.glob(apath+'/bbobexp_f*.info')))
for spath in strats:
    print spath
    sname = os.path.basename(os.path.normpath(spath))
    snamemap = {
            'mUNIF7': 'UNIF',
            'mE50G7': 'EG',
            'mPM0.025_0.1_7': 'PM',
            'mAP0.075_0.5_0.1_7': 'AP',
            'mT1_1A7': 'TA',
            'mMMK14bs': 'MMK',
            'mMM8_7': 'MML',
            'mHUC4.0B7sr_0.95_50m': 'SR',
            'mHUC2.0B7auc_0.75_50m': 'AUC',
            'mUC8.0B7xr_log,adapt0.9': 'RUCB',
            'mUC16.0B7x_log,adapt0.7': 'LUCB',
        }
    pds.add_strategy(snamemap[sname], bb.load(glob.glob(spath+'/bbobexp_f*.info')))

# TODO: Make generating these optional?
print "bestalg"
pds.bestalg(None)
print "unifpf"
pds.unifpf()

# TODO: Pickle to tmp file and rename()?
print "Pickling..."
pds.pickle(picklefile)
