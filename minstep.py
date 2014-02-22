#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A wrapper to scipy.optimize.minimize that returns after each iteration
(and resumes again on demand), i.e. provides an iteration by iteration
stepping functionality.
"""

import threading
import traceback
from Queue import Queue

import numpy as np
import numpy.random as nr
import scipy.optimize as so


class ThreadCancel(Exception):
    pass

class MinimizeThread(threading.Thread):
    def __init__(self, fun, x0, minmethod):
        threading.Thread.__init__(self)

        self.fun = fun
        self.x0 = x0
        self.minmethod = minmethod

        # iterq passes tuples from the minimization to caller
        self.iterq = Queue(maxsize = 1)
        self.iterq_first = object()
        self.iterq_iter = object()
        self.iterq_finished = object()

        # stopev is used to signalize the minimization should stop
        self.stopev = threading.Event()

        self.last_x = self.x0

    def run(self):
        class OICallback:
            # A shim to call self.one_iter()
            def __init__(self, thread):
                self.thread = thread
            def __call__(self, xk):
                self.thread.one_iter(xk)
        callback = OICallback(self)

        try:
            # Wake the MinimizeStepping constructor
            self.iterq.put((self.iterq_first, 0))

            self.iterq.join() # wait for unblocking by the first next()
            if self.stopev.is_set():
                raise ThreadCancel()

            r = self.minmethod(self.fun, self.x0, inner_cb = callback)

            # Report the final result (XXX: or is it a dupe?)
            x = getattr(r, 'x', self.x0)
            if np.any(x != self.last_x):
                self.one_iter(x)
            self.iterq.put((self.iterq_finished, 0))

        except ThreadCancel:
            return

    def one_iter(self, xk):
        """
        Called after every iteration of minimize.
        """
        # TODO: Possibly pass a whole OptimizeResult?
        self.last_x = xk
        self.iterq.put((self.iterq_iter, xk))
        self.iterq.join() # wait for unblocking by next()

        if self.stopev.is_set():
            raise ThreadCancel()


class MinimizeStepping:
    """
    Minimization of scalar function of one or more variables. Just like
    scipy.optimize.minimize(), but it creates an object via which you can
    step, instead of a final OptimizeResult object. Example:

    >>> mm = MinimizeMethod("Powell", fi)
    >>> ms = MinimizeStepping(fun, x0, mm)
    >>> ms.next()
    [1.0, 2.0]
    >>> ms.next()
    [1.5, 2.1]
    >>> ms.stop() # This is necessary, not automatic!
    """

    def __init__(self, fun, x0, minmethod):
        """
        Initialize the object and also start up the thread.
        """
        self.minmethod = minmethod

        # Our design is thread-based, but there is no concurrency!
        # There is always *only one* thread running (either the main
        # thread or MinimizeThread), everything else blocks.
        self.thread = MinimizeThread(fun, x0, minmethod)
        self.thread.start()

        # Now block until the thread is initialized...
        self.thread.iterq.get(True)
        # ...and now self.thread is blocked.

        # Using isAlive() is racy with deinitialization
        self.thread_alive = True

    def next(self):
        """
        Run for a single iteration and return the current x.
        Throws StopIteration if the minimizer finished (no need to call stop()).
        """
        # Unblock self.thread
        self.thread.iterq.task_done()
        # Block us on self.thread
        msg = self.thread.iterq.get(True)
        # ...and now self.thread is blocked again.

        if msg[0] is self.thread.iterq_iter:
            return msg[1]
        elif msg[0] is self.thread.iterq_finished:
            self.thread.join()
            self.thread_alive = False
            raise StopIteration()
        else:
            raise RuntimeError('unknown message %s' % msg)

    def stop(self):
        """
        Calling this function is *required* in case minimization is
        interrupted early.
        """
        if not self.thread_alive:
            return
        # First set stopev...
        self.thread.stopev.set()
        # ...then unblock self.thread, there will be no more messages...
        self.thread.iterq.task_done()
        self.thread_alive = False
        # ...and collect the thread.
        self.thread.join()
