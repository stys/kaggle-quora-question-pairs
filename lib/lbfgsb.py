# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
from sklearn.utils.fixes import expit


class LogisticRegressionLBFGSB(object):
    def __init__(self, l1=0.0, l2=0.0, tol=1.e-3, max_iter=15000):
        self.l1 = l1
        self.l2 = l2
        self.tol = tol
        self.max_iter = max_iter
        self.we = None
        self.info = None

    def fit(self, X, y):
        self.we, _, self.info = LogisticRegressionLBFGSB.optimize(X, y, self.l1, self.l2, self.tol, self.max_iter)

    def predict(self, X):
        n = (len(self.we) - 1) / 2
        w0 = self.we[0]
        w = self.we[1:n+1] - self.we[n+1:]
        return expit(w0 + safe_sparse_dot(X, w))

    @staticmethod
    def fgrad(we, X, y, l1, l2):
        nsamples, nfactors = X.shape

        w0 = we[0]
        w = we[1:(nfactors+1)] - we[(nfactors+1):]
        yz = y * (safe_sparse_dot(X, w) + w0)
        f = - np.sum(log_logistic(yz)) + l1 * np.sum(we[1:]) + 0.5 * l2 * np.dot(w, w)

        e = (expit(yz) - 1) * y
        g = safe_sparse_dot(X.T, e) + l2 * w
        g0 = np.sum(e)

        grad = np.concatenate([g, -g]) + l1
        grad = np.insert(grad, 0, g0)

        return f, grad

    @staticmethod
    def optimize(X, y, l1, l2, tol, max_iter):
        nsamples, nfactors = X.shape
        we = np.zeros(1 + 2 * nfactors)
        wb = [(None, None)] + [(0, None)] * nfactors * 2

        we, f, info = fmin_l_bfgs_b(
            func=LogisticRegressionLBFGSB.fgrad,
            x0=we,
            bounds=wb,
            fprime=None,
            pgtol=tol,
            maxiter=max_iter,
            args=(X, y, l1, l2),
            iprint=99
        )

        return we, f, info
