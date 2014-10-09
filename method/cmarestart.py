#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A wrapper to CMA restart strategies (IPOP, BIPOP) that picks initial
x0 values according to the basinhopping algorithm.

Achieving this is a little complicated as we need to run two functions
in parallel:

    basinhopping
      `-------> calls our wrapper with x0
      :            `---> fmin CMA restarts>0
      :                   | CMA runs for one iteration
      :                   fmin callbacks x0 parameter
      :       v---------' :
      : return to bhop    :
      v-------'           :
      | step_func         :
      `-------> wrapper   :
      :            `------> fmin returns from x0 callback
      :                   | CMA runs for one iteration
      .....................

Essentially, basinhopping and fmin would be coroutines, but we can't
use Python coroutines since the yield cannot be nested in the callback.
So instead we use the same mechanism as MinStep, running fmin() in
a separate Thread and obtaining x0 values over a Queue.  We archive
the last best x reported and use that during the request for new x0.

N.B. the MinimizeStepping machinery is happenning *outside* the
basinhopping routine, therefore we don't need to worry too much,
but we do need to re-throw the callback exception from fmin thread
also in the basinhopping thread.
"""

import threading
from Queue import Queue
import scipy.optimize as so
import sys

sys.path.insert(0, '.')
import cma


def cma_wrapper(fun, x0, cb, minimizer_kwargs):
    """
    This function actually executes cma.fmin() in a sort of
    user friendly way.
    """
    class InnerCMACallback:
        def __init__(self, realcb):
            self.realcb = realcb
        def __call__(self, cma):
            self.realcb(cma.best.x)

    fminargs = minimizer_kwargs['fminargs']
    fminargs['options']['termination_callback'] = InnerCMACallback(cb) if cb is not None else None

    try:
        return cma.fmin(fun, x0, 10./4., **fminargs)
    except cma._Error, e:
        print "CMA error: " + str(e)
        return None


class CMAThread(threading.Thread):
    """
    This is the thread that runs CMA, stopping whenever CMA wants to restart
    and asking for new x0 over a queue.
    """

    def __init__(self, fun, minimizer_kwargs):
        threading.Thread.__init__(self)

        self.fun = fun

        minimizer_kwargs = dict(minimizer_kwargs)
        self.origcb = minimizer_kwargs.pop('callback')
        self.minimizer_kwargs = minimizer_kwargs

        # x0request sends last_x to the basinhopping thread when we are
        # restarting and need an x0...
        self.x0request = Queue(maxsize=1)
        # ...and after that the x0 is obtained from x0receive.
        self.x0receive = Queue(maxsize=1)

        # stopev is used to signalize to basinhopping that the minimization
        # should stop...
        self.stopev = threading.Event()
        # ...which is always because of an exception (typically
        # minstep.ThreadCancel but also examples.single.MMCancel etc.)
        # stored in this attribute.
        self.stop_e = None

        self.last_x = None

    def run(self):
        # We define some callbacks which are just essentially method pointers:

        class IterCallback:
            """
            This callback is invoked whenever a single iteration of CMA
            is done, yielding the best x0 found.
            """
            def __init__(self, cmathread):
                self.cmathread = cmathread
            def __call__(self, xk):
                self.cmathread.one_iter(xk)

        class X0Callback:
            """
            This callback is called when we need to restart.
            """
            def __init__(self, cmathread):
                self.cmathread = cmathread
            def __call__(self):
                return self.cmathread.x0cb()

        # Now, we simply repeatedly call the CMA fmin:

        try:
            while True:
                cma_wrapper(self.fun, X0Callback(self), IterCallback(self),
                            self.minimizer_kwargs)
                # If we get here, the CMA returned naturally; either we
                # had no restart strategy or (unusually) at least ran out
                # of permitted restarts.  Well, simply start again from
                # the beginning!

        except Exception as e:
            # Optimization run cancellation by the main thread.
            # Let basinhopping thread know it's over...
            self.stopev.set()
            self.stop_e = e
            # ...and unblock it.
            self.x0request.put(self.last_x)
            self.x0request.join()
            return

    def one_iter(self, xk):
        """
        This callback is invoked whenever a single iteration of CMA
        is done, yielding the best x0.  We archive that as our return
        value when we need to come back to basinhopping.

        Also, we call the original one_iter callback; it may throw
        minstep.ThreadCancel, but we deal with that in run().
        """
        self.last_x = xk
        self.origcb(xk)

    def x0cb(self):
        """
        This callback is invoked when we need to obtain a new x0.
        """
        self.x0request.put(self.last_x)
        self.x0request.join()
        x0 = self.x0receive.get(True)
        self.x0receive.task_done()
        return x0


class CMAController:
    """
    This is the controller of CMAThread that communicates with it.
    At the same time, it is a callable that serves as scipy.optimize
    compatible minimize function method, being called (via the __call__
    method) repeatedly from basinhopping.
    """
    def __init__(self):
        # We will spawn the thread on the first __call__ when we
        # get all the required arguments.
        self.cmathread = None

    def __call__(self, fun, x0, **kwargs):
        """
        Perform a single CMA run up to a restart (at that point,
        we return from here but not from CMA's fmin(); when we get
        called here again, we pass the new x0 to the CMA's fmin()
        innards and go on.
        """
        if self.cmathread is None:
            # Spawn the thread and acknowledge the first x0 request.
            self.cmathread = CMAThread(fun, kwargs)
            self.cmathread.start()
            self.cmathread.x0request.get(True)
            self.cmathread.x0request.task_done()
        cma = self.cmathread

        # Send the x0 we have
        cma.x0receive.put(x0)
        cma.x0receive.join()

        # Wait until next x0 is requested... (after single CMA run)
        best_x = cma.x0request.get(True)
        cma.x0request.task_done()

        # Test whether we aren't actually done
        if cma.stopev.is_set():
            cma.join()
            raise cma.stop_e

        # Return best_x; we will be called again with new x0
        if best_x is not None:
            return so.OptimizeResult(fun=fun(best_x), x=best_x, success=True)
        else:
            return so.OptimizeResult(success=False)
