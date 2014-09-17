COCOpf: An Algorithm Portfolio COCO/BBOB Platform
=================================================

**COCOpf** is a Python-based platform for research and development of
hyperheuristics maintaining portfolios of optimization algorithms within
the excellent [COCO benchmarking framework for BBOB workshops](http://coco.gforge.inria.fr/doku.php).

Algorithm portfolios represent an approach where multiple heuristic
optimization algorithms, each most suitable for a different class of
problems, are combined together in a single general solver that can
choose the best algorithm to run for each problem instance on the input.

Using **COCOpf**, you can immediately start experimenting with ways to
combine multiple heuristic black-box optimization algorithms to achieve
the best combined performance on the COCO/BBOB benchmark function set.

**COCOpf** brings the following on the table:

  * A ready-made ``Experiment`` class that will shorten and simplify your
    experiment scripts.

  * A generic ``MinimizeMethod`` interface that will allow you to maintain
    a common set of custom optimization algorithms across all your
    hyperheuristic experiments.

  * The pre-made ``MinimizeMethod`` interface gives you instant access to
    seven minimization algorithms that are part of *scipy.optimize*
    (wrapped in the basinhopping global optimization algorithm),
    and to the state-of-art CMA-ES population-based minimization method.
    You can easily add more custom minimization methods using subclassing.

  * As a unique feature, **COCOpf** allows you to single-step
    the minimization algorithms iteration by iteration (via the
    ``MinimizeStepping`` wrapper), allowing usage of a flexible suspend
    / resume schedule.  This requires no modification of the used
    minimization algorithms as long as they can simply execute
    a callback function after each iteration!  (I.e. you can do this
    with the SciPy minimizers and CMA-ES, without code modifications.)

  * A ready-made ``Population`` class for easy maintenance of a portfolio
    of multiple concurrently executed (algorithm, solution) pairs.

COCOpf has been tested just on Debian Linux so far.


Getting Started
---------------

First, get and unpack the latest bbob COCO distribution, e.g.:

	http://coco.lri.fr/downloads/download13.09/bbobexp13.09.tar.gz

Change to the ``python/`` subdirectory and ``git clone
https://github.com/pasky/cocopf`` here - that should
create a ``python/cocopf/`` directory.

Well, that's it!  It is time to run some examples - find them all
in the ``examples/`` subdirectory and run them all from within the
``python/`` directory.  Description and instructions are at the
top of each example.  Let's try the epsilon-greedy strategy:

	bbob/python$ cocopf/examples/pop-egreedy.py Powell,BFGS,SLSQP 3

You can use GNU Parallel to easily execute multiple experiments in
parallel.  Let's compare the raw performance of the three algorithms
run outside a portfolio, generating some nice graphs while we are at it:

	bbob/python$ parallel -u --gnu cocopf/examples/single.py {} ::: Powell BFGS SLSQP
	bbob/python$ cd data-1e3
	bbob/python/data-1e3$ ../bbob_pproc/rungenericmany.py Powell BFGS SLSQP
	bbob/python/data-1e3$ cd ../../latextemplates
	bbob/latextemplates$ ln -s ../python/data-1e3/ppdata .
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


Other Approaches
----------------

COCOpf itself comes with a ``bbob_pproc.algportfolio`` module that allows
one to simulate parallel run of multiple tested algorithms.  Back-testing
against past obtained results is also popular when researching algorithm
portfolios with offline selection and it certainly is useful in that
scenario.

COCOpf's approach is different in three important aspects:

  * During online selection, algorithms are switched between their
    full iterations in COCOpf.  However, algorithm iterations are
    not recorded in the COCO/BBOB dat files and back-testing using the
    algportfolio approach therefore allows for algorithm switching
    only between individual function evaluations (either after each
    evaluation, or after every N iterations).

    We argue that COCOpf's approach is more realistic in the online
    selection setting as switching algorithms may not be appropriate
    after an arbitrary function evaluation - algorithms may evaluate
    the function a variable number of times, we may try to switch
    from a genetic algorithm instance multiple times during a single
    population evaluation, and by checking too often, we can incur
    a significant computation overhead from the algorithm selection
    itself.

  * It makes it easier to implement own online algorithm selection
    strategies than modifying the code of algportfolio, which can be
    a little opaque.

  * It allows not just for benchmarking, but also for reusing
    the same code for practical black-box optimization of functions
    that are not part of COCO/BBOB.


Post-Processing
---------------

In addition to the **bbob_pproc** tools of COCO that you have on your
disposal (e.g. ``rungeneric.py``), COCOpf includes also its own
analysis and post-processing utilities that are geared at exploring
the effect of strategies relative to the portfolio rather than on
raw optimization performance.

The first step is to pickle your current portfolio dataset collection.
Let's assume ``data-alg`` directory contains per-algorithm datasets
and ``data-strat`` directory contains per-strategy datasets:

	cocopf/pptools/pfpickle.py bestmix.pickle.gz data-alg/* -- data-strat/*

Now, just to get started, let's show some plots of convergence
to optimum for 5D functions 2, 7 and 11 - how the function value
changes in time (IOW, budget):

	cocopf/pptools/plot_conv.py bestmix.pickle.gz fval_by_budget 5  2 7 11

Also, we can generate TeX-formatted tables that display averaged
performance over various function classes:

	cocopf/pptools/table_final.py bestmix7.pickle.gz slowdown2oracle 5 all volatile all-volatile

More to come soon!

The heavy lifting is all done by the ``pproc`` and ``pplot`` modules,
easy to use from ipython or your custom scripts e.g. preparing
figures for publication.
