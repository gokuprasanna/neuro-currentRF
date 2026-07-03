"""Empirical-Bayes / MNE-style initialization for the NCRF source covariance."""
from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from ._typing import FloatArray


# Functions used for initialize \Gamma
def find_mu(
        s: FloatArray,
        y: FloatArray,
        eta: float = 1,
        tol: float = 1e-8,
        max_iter: int = 1000,
) -> float:
    """Solve for the empirical-Bayes noise parameter used in initialization."""
    logger = logging.getLogger(__name__)
    e = s ** 2
    z = y ** 2
    TM = z.size
    eta = eta * TM
    z2 = z.sum(axis=1)
    mu = 0
    diff = []

    logger.info('please wait: calculating mu...')
    for _ in range(max_iter):
        temp = 1 + mu * e
        fmu = z2 / (temp ** 2)
        f = fmu.sum() - eta
        dfmu = (-2) * fmu * e / temp
        diff.append(f / dfmu.sum())
        if (mu == 0 and f < 0) or abs(diff[-1] / diff[0]) < tol:
            logger.info(f"thanks for waiting, (mu: {mu}) calculation complete after:"
                        f"iteration # {len(diff)} with relative error {diff[-1] / diff[0]}")
            return mu
        mu -= diff[-1]

    logger.info(f"maximum iteration {max_iter} reached, consider more iterations for convergence!")
    return mu


def wls(
        y: FloatArray,
        l: FloatArray,
        w: FloatArray,
        return_ecov: bool = False,
) -> tuple[FloatArray, float] | tuple[FloatArray, float, FloatArray]:
    """Solve the weighted least-squares problem used for NCRF initialization."""
    w = np.squeeze(w)
    if w.ndim == 1:
        lw = l * w[None, :]
    else:
        lw = l @ w
    u, s, vh = linalg.svd(lw, full_matrices=False)
    yw = u.T @ y
    mu = find_mu(s, yw, eta=1)
    if mu:
        gamma = s / (s ** 2 + 1 / mu)
    else:
        gamma = 1 / s

    if w.ndim == 1:
        im = w[:, None] * vh.T
    else:
        im = w @ vh.T

    im = im * gamma[None, :]

    if return_ecov is True:
        ecov = np.eye(w.shape[0]) - vh.T @ ((gamma * s)[:, None] * vh)
        ecov *= mu
        if w.ndim == 1:
            ecov *= w[:, None]
            ecov *= w[None, :]
        else:
            ecov = ecov @ w.T
            ecov = w @ ecov
        return im @ yw, mu, ecov

    return im @ yw, mu


def mne_initialization(
        y: FloatArray,
        l: FloatArray,
        use_depth_prior: bool = True,
        exp: float = 0.8,
) -> tuple[FloatArray, FloatArray]:
    """Build the initial Gamma and sensor covariance from an MNE-style estimate."""
    N, M = l.shape
    T = y.shape[1]

    if use_depth_prior:
        dw = 1.0 / (l ** 2).sum(axis=0)
        limit = dw.min() * 10.0
        depth_weighting = (np.minimum(dw / limit, 1)) ** exp
    else:
        depth_weighting = np.ones(M)

    w = np.ones(M)
    w *= depth_weighting
    inv, mu, ecov = wls(y, l, w, return_ecov=True)
    Gamma = np.diag((inv @ inv.T) / T + ecov)
    data_cov = l * Gamma[None, :] @ l.T
    return Gamma, data_cov
