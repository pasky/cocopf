#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Management of "credits" assigned to method population members, that can
then be used for method choice - accrued credits determine fitness of
the members. (We are still minimizing the credit, i.e. smaller is better.)

Credit assignment mostly involves rescaling raw results to a [0,1]
interval - either working with the fitness itself or just its rank
(sorted position) within the population.

Credit accrual can commonly be simply resetting credit of the whole
population - keeping no history and instead choosing just based on
the last result.  Alternatively, it can be a simple average, window
based average, exponential back-off, etc.

To use the credits infrastructure, simply maintain a PopulationCredit
instance together with Population and choose based on
popcredit.credit[i] instead of pop.values[i]. Consider allowing
the user to override the method names.
"""

import numpy as np
import re


class PopulationCredit(object):
    """
    A container describing the credit of a population tied to
    particular assignment and accrual methods.

    The assign_method works on the population object, returning
    a credit array corresponding to current population values.

    The accrual_method works on individual population members,
    accepting current credit, n number of iterations and newly
    added credit, and returning new credit for n+1.
    If accrual_method is a name ending with 'r', the 'r' is
    interpreted as reset_on_restart = True.

    The method parameters may be either callables or strings.
    If you want to provide custom credit management in your
    experiment, it is a better idea to subclass PopulationCredit
    extending the _assign_resolve() and/or _accrual_resolve()
    methods.
    """
    def __init__(self, pop, assign_method, accrual_method,
            reset_on_restart = False):
        self.pop = pop
        self.reset_on_restart = reset_on_restart

        self.credit = np.zeros(self.pop.K) + 1
        self.iters = self.pop.iters.copy()

        if not callable(accrual_method) and accrual_method.endswith("r"):
            accrual_method = accrual_method[:-1]
            self.reset_on_restart = True

        self.assign_method = assign_method if callable(assign_method) else self._assign_resolve(assign_method)
        self.accrual_method = accrual_method if callable(accrual_method) else self._accrual_resolve(accrual_method)

    def _assign_resolve(self, name):
        """
        Resolve assignment operator names, possibly allowing
        the user to specify extra parameters. Feel free to extend
        this in your subclasses.
        """
        if name == "raw":
            return CreditAssignRaw()
        elif name == "ranked":
            return CreditAssignRanked()
        else:
            raise ValueError("Unknown credit assignment operator " + str(name))

    def _accrual_resolve(self, name):
        """
        Resolve accrual operator names, possibly allowing
        the user to specify extra parameters. Feel free to extend
        this in your subclasses.
        """
        if name == "latest":
            return CreditAccrualLatest()
        elif name == "average":
            return CreditAccrualAverage()
        m = re.match("adapt(.*)", name)
        if m: return CreditAccrualAdapt(float(m.group(1)))
        raise ValueError("Unknown credit accrual operator " + str(name))

    def add(self):
        """
        Add another population member.
        """
        self.credit = np.append(self.credit, [1], axis = 0)
        self.iters = np.append(self.iters, [0], axis = 0)

    def update(self):
        """
        Update credit information after pop.end_iter() has been called.
        We will assign fresh credit to the population, then proceed
        with accrual for members which were stepped.
        """
        self.pop.values[np.isnan(self.pop.values)] = 10e8
        new_credit = self.assign_method(self.pop)

        for i in range(self.pop.K):
            if self.pop.iters[i] == self.iters[i]:
                continue
            if self.pop.iters[i] < self.iters[i] and self.reset_on_restart:
                self.iters[i] = 0
            if self.iters[i] > 0:
                self.credit[i] = self.accrual_method(self.credit[i], self.iters[i], new_credit[i])
            else:
                self.credit[i] = new_credit[i]
            self.iters[i] += 1


class CreditAssignRaw(object):
    """
    The most naive credit assignment method is to consider just
    the raw results, with no rescaling.
    """
    def __init__(self):
        pass
    def __call__(self, pop):
        return pop.values.copy()

class CreditAssignRanked(object):
    """
    A popular credit assignment method that is scale-invariant
    is ranking the results.  The functions get assigned credit
    according to their rank, linearly rescaled to [0,1].
    """
    def __init__(self):
        pass
    def __call__(self, pop):
        idx = np.argsort(pop.values)
        credit = np.zeros(pop.K)
        for i in range(pop.K):
            credit[idx[i]] = i/(pop.K-1.0)
        return credit


class CreditAccrualLatest(object):
    """
    The most naive credit accrual method is to consider just the latest
    optimization result, with no history.
    """
    def __init__(self):
        pass
    def __call__(self, credit_old, iters, credit_new):
        return credit_new

class CreditAccrualAverage(object):
    """
    Another fairly naive credit accrual method is to simply average
    all credits over the history.
    """
    def __init__(self):
        pass
    def __call__(self, credit_old, iters, credit_new):
        return credit_old + (credit_new - credit_old) / iters

class CreditAccrualAdapt(object):
    """
    A common idea is to compute an average that exponentially fast
    discounts older samples, by using a fixed adaptation constant.
    """
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, credit_old, iters, credit_new):
        return credit_old + (credit_new - credit_old) * self.alpha
