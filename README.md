HyperBBOB: An Algorithm Portfolio BBOB Platform
===============================================

**HyperBBOB** is a Python-based platform for research and development of
hyperheuristics maintaining portfolios of optimization algorithms within
the excellent [BBOB benchmarking framework](http://coco.gforge.inria.fr/doku.php).

Using **HyperBBOB**, you can immediately start experimenting with ways to
combine multiple heuristic black-box optimization algorithms to achieve
the best combined performance on the BBOB benchmark function set.

**HyperBBOB** brings the following on the table:

  * A ready-made ``Experiment`` class that will shorten and simplify your
    experiment scripts.

  * A generic ``MinimizeMethod`` interface that will allow you to maintain
    a common set of custom optimization algorithms across all your
    hyperheuristic experiments.

  * The pre-made ``MinimizeMethod`` interface gives you instant access to
    seven minimization algorithms that are part of *scipy.optimize*
    (wrapped in the basinhopping global optimization algorithm),
    plus there is an example incorporating *CMA-ES*, allowing you to
    focus just on your hyperheuristics.

  * As a unique feature, **HyperBBOB** allows you to single-step
    the minimization algorithms iteration by iteration (via the
    ``MinimizeStepping`` wrapper), allowing usage of a flexible suspend
    / resume schedule.  This requires no modification of the used
    minimization algorithms as long as they can simply execute
    a callback function after each iteration!  (I.e. you can do this
    with the SciPy minimizers and CMA-ES, without code modifications.)

  * A ready-made ``Population`` class for easy maintenance of a portfolio
    of multiple concurrently executed (algorithm, solution) pairs.

HyperBBOB has been tested just on Debian Linux so far.


Getting Started
---------------

First, get and unpack the latest bbob distribution, e.g.:

	http://coco.lri.fr/downloads/download13.09/bbobexp13.09.tar.gz

Change to the ``python/`` subdirectory and ``git clone
https://github.com/pasky/hyperbbob`` here - that should
create a ``python/hyperbbob/`` directory.

Well, that's it!  It is time to run some examples - find them all
in the ``examples/`` subdirectory and run them all from within the
``python/`` directory.  Description and instructions are at the
top of each example.  Let's try the epsilon-greedy strategy:

	bbob/python$ hyperbbob/examples/pop-egreedy.py Powell,BFGS,SLSQP 3

You can use GNU Parallel to easily execute multiple experiments in
parallel.  Let's compare the raw performance of the three algorithms
run outside a portfolio, generating some nice graphs while we are at it:

	bbob/python$ parallel -u --gnu hyperbbob/examples/single.py {} ::: Powell BFGS SLSQP
	bbob/python$ cd data-10e3
	bbob/python/data-10e3$ ../bbob_pproc/rungenericmany.py Powell BFGS SLSQP
	bbob/python/data-10e3$ cd ../../latextemplates
	bbob/latextemplates$ ln -s ../python/data-10e3/ppdata .
	bbob/latextemplates$ pdflatex templateBBOBmany.tex
	bbob/latextemplates$ evince templateBBOBmany.pdf

(If you need to interrupt the experiment, use Ctrl-\ (SIGQUIT) instead
of Ctrl-C.  You will need to wipe out the generated data directories
before restarting the experiment!)


Documentation
-------------

There is some inline documentation, but it's still a work in progress.
However, the code is short and commented.  You can use the examples
as templates for your own experiments.  Contributions are welcome!
