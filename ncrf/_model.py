"""The NCRF estimator, its fitted model, and the fit report.

:class:`NCRF` drives the fit (regularization selection and optimization),
:class:`NCRFModel` is the frozen, reusable result that can be applied to new
data, and :class:`NCRFResult` bundles the model with training-set evaluation
and provenance.
"""
# Authors: Proloy Das <email:proloyd94@gmail.com>
#          Christian Brodbeck <email:brodbecc@mcmaster.ca>
# License: BSD (3-clause)
from __future__ import annotations

import logging
from functools import cached_property
from operator import attrgetter

from eelbrain import NDVar, fmtxt
import numpy as np
from scipy.signal import find_peaks

from ._crossvalidation import CVResult, crossvalidate
from ._data import RegressionData
from ._forward import ForwardModel
from ._reconstruction import TRFDesign
from ._solver import FitHistory, Solver, _evaluate_objective, find_mu_range
from ._typing import FloatArray, MuArg, MusArg


class NCRFModel:
    """Frozen, fitted NCRF model that can be applied to arbitrary datasets.

    Holds the estimated weights together with the forward model and stimulus
    metadata needed to evaluate (and, in the future, predict) on any
    :class:`RegressionData`.  Reusable and picklable; produced by :meth:`NCRF.fit`
    and exposed as :attr:`NCRFResult.model`.

    Attributes
    ----------
    forward
        The :class:`ForwardModel` (lead field, whitening filter, source/sensor/space).
    theta
        NCRF coefficients over the Gabor basis; the frozen weights.
    Gamma
        Per-trial source covariance estimates.
    Sigma_b
        Per-trial data covariance estimates (used by :meth:`eval_obj`).
    mu
        Regularization parameter used to fit the weights.
    tstart, tstep, tstop, basis_std
        TRF timing and Gaussian-basis width.
    """
    _name = 'cTRFs estimator'

    def __init__(
            self,
            *,
            forward: ForwardModel,
            theta: FloatArray,
            Gamma: list,
            Sigma_b: list,
            mu: float,
            design: TRFDesign,
    ) -> None:
        self.forward = forward
        self.theta = theta
        self.Gamma = Gamma
        self.Sigma_b = Sigma_b
        self.mu = mu
        self._design = design

    @property
    def tstart(self) -> list[float]:
        return self._design.tstart

    @property
    def tstep(self) -> float:
        return self._design.tstep

    @property
    def tstop(self) -> list[float]:
        return self._design.tstop

    @property
    def basis_std(self) -> float:
        return self._design.basis_std

    @classmethod
    def _from_solver(cls, solver: Solver, data: RegressionData) -> NCRFModel:
        """Freeze a finished solver together with the fitted data's metadata."""
        return cls(
            forward=solver.forward,
            theta=solver.theta,
            Gamma=solver.Gamma,
            Sigma_b=solver.Sigma_b,
            mu=solver.mu,
            design=data.trf_design,
        )

    def __repr__(self) -> str:
        orientation = 'free' if self.forward.space else 'fixed'
        return f"<[{orientation} orientation] {self._name} on {self.forward.source!r}>"

    def _whiten(self, data: RegressionData) -> RegressionData:
        """Whiten ``data`` unless it is already whitened (no-op for fitted/CV data)."""
        if data.is_whitened:
            return data
        return data.whiten(self.forward.whitening_filter)

    def _predict_whitened(self, covariate: FloatArray) -> FloatArray:
        """Predicted whitened sensor data for one trial's covariate matrix."""
        return np.dot(np.dot(self.forward.whitened_lead_field, self.theta), covariate.T)

    def eval_obj(
            self,
            data: RegressionData,
            return_wl2: bool = False,
    ) -> float | tuple[float, float]:
        """Evaluate the model's objective value on a dataset.

        Parameters
        ----------
        data
            Dataset on which to evaluate the objective.
        return_wl2
            Also return the weighted L2 term.

        Returns
        -------
        float | tuple[float, float]
            Objective value, or a pair containing the objective value and the
            weighted L2 term when ``return_wl2`` is true.
        """
        data = self._whiten(data)
        return _evaluate_objective(self.forward, self.theta, self.Sigma_b, data, return_wl2)

    def explained_variance(self, data: RegressionData) -> float:
        """Compute the global explained-variance score on a dataset."""
        logger = logging.getLogger('NCRF: Explained Variance')
        data = self._whiten(data)
        temp = 0
        for key, (meg, covariate) in enumerate(data):
            W_meg = meg
            y = W_meg - self._predict_whitened(covariate)
            temp += np.nansum(np.var(y, axis=1) / np.var(W_meg, axis=1)) / y.shape[0]

        logger.debug(f'{self.mu}: {1 - temp / len(data)}')
        return 1 - temp / len(data)

    def voxelwise_explained_variance(self, data: RegressionData) -> NDVar:
        """Compute each source's contribution to explained variance."""
        data = self._whiten(data)
        W_leadfield = self.forward.whitened_lead_field
        temp = np.zeros(len(self.forward.source))
        for meg, covariate in data:
            total_var = np.var(meg, axis=1)
            y_full = meg - np.dot(np.dot(W_leadfield, self.theta), covariate.T)
            base_var = np.var(y_full, axis=1)
            for i in range(len(self.forward.source)):
                # Zeroing source i's weights just removes its (linear) contribution
                # to the prediction, so add that contribution back to the residual.
                block = self.forward.source_block(i)
                contribution = np.dot(np.dot(W_leadfield[:, block], self.theta[block]), covariate.T)
                y_i = y_full + contribution
                temp[i] += np.nansum((np.var(y_i, axis=1) - base_var) / total_var) / meg.shape[0]

        return NDVar(temp / len(data), self.forward.source)

    @cached_property
    def h_scaled(self) -> NDVar | list[NDVar]:
        """Return ``h`` with the original stimulus scaling restored."""
        return self._design.reconstruct_scaled(self.h)

    @cached_property
    def h(self) -> NDVar | list[NDVar]:
        """Return the spatio-temporal response function as Eelbrain NDVars."""
        return self._design.reconstruct(self.theta, self.forward)


class NCRF:
    """Estimator for neuro-current response functions (cTRFs).

    Construct with a forward model and noise covariance, then call :meth:`fit`
    with a :class:`RegressionData` instance to obtain an :class:`NCRFResult`.

    Parameters
    ----------
    lead_field
        Forward solution a.k.a. lead-field matrix, with ``sensor`` and ``source``
        dimensions and an optional ``space`` dimension for free orientation.
    noise_covariance
        Noise covariance matrix in sensor space, typically estimated from empty-room
        recordings.
    n_iter
        Number of outer iterations of the algorithm.
    n_iterc
        Number of Champagne iterations within each outer iteration.
    n_iterf
        Number of FASTA iterations within each outer iteration.

    Notes
    -----
    Usage:

    1. Use :meth:`RegressionData.from_data` to construct a prepared dataset
       from MEG and stimulus segments.
    2. Initialize :class:`NCRF` with the lead field and noise covariance.
    3. Call :meth:`NCRF.fit` with the :class:`RegressionData` instance; it
       returns an :class:`NCRFResult` with the estimated cortical TRFs.
    """
    _name = 'cTRFs estimator'

    def __init__(
            self,
            lead_field: NDVar,
            noise_covariance: FloatArray,
            n_iter: int = 30,
            n_iterc: int = 10,
            n_iterf: int = 100,
    ) -> None:
        self.forward = ForwardModel.from_lead_field(lead_field, noise_covariance)
        self.n_iter = n_iter
        self.n_iterc = n_iterc
        self.n_iterf = n_iterf

    def __repr__(self) -> str:
        orientation = 'free' if self.forward.space else 'fixed'
        return f"<[{orientation} orientation] {self._name} on {self.forward.source!r}>"

    def _new_solver(self) -> Solver:
        return Solver(self.forward, self.n_iter, self.n_iterc, self.n_iterf)

    def fit(
            self,
            data: RegressionData,
            mu: MuArg = 'auto',
            do_crossvalidation: bool = False,
            tol: float = 1e-5,
            verbose: bool = False,
            use_ES: bool = False,
            mus: MusArg = None,
            n_splits: int = None,
            n_workers: int = None,
            compute_explained_variance: bool = False,
            accept_whitening: bool = False,
            store_theta: bool = False,
            store_gamma: bool = False,
            store_sigma_b: bool = False,
    ) -> NCRFResult:
        """Fit the NCRF model to prepared regression data.

        Estimate both TRFs and source variance from the observed MEG data by solving
        the Bayesian optimization problem formulated in :cite:`das2020neuro`.

        Parameters
        ----------
        data
            M/EEG data and the corresponding stimulus variables. Not mutated.
        mu
            Regularization parameter; promote sparsity and guard against over-fitting
        do_crossvalidation
            if True, from a wide range of regularizing parameters, the one resulting in
            the least generalization error in a k-fold cross-validation procedure is chosen.
            Unless specified the range and k is chosed from cofig.py. The user can also pass
            several keyword arguments to overwrite them.
        tol
            tolerence parameter. Decides when to stop outer iterations.
        verbose
            If set True prints intermediate values of the cost functions (default ``False``).
        use_ES
            use estimation stability criterion :cite:`limEstimationStabilityCrossValidation2016`
            to choose the best ``mu`` (default ``False``).
        mus
            range of mu to be considered for cross-validation
        n_splits
            k value used in k-fold cross-validation
        n_workers
            Number of workers to use for cross-validation.
            ``None`` to use ``cpu_count/2`` (default).
            ``0`` to run without :mod:`multiprocessing`.
        compute_explained_variance
            Compute voxel-wise explained variance.
        accept_whitening
            Accept pre-whitened data. This is intended for internal workflows
            that slice an already-whitened dataset, such as cross-validation.
        store_theta
            Store the ``theta`` estimate after each outer iteration in the
            result's :class:`FitHistory`.
        store_gamma
            Store the source covariances after each outer iteration.
        store_sigma_b
            Store the data covariances after each outer iteration.

        Returns
        -------
        NCRFResult
            The fitted model and estimated cortical TRFs.
        """
        if data.is_whitened:
            if not accept_whitening:
                raise ValueError("data is already whitened; pass accept_whitening=True to accept it")
        else:
            data = data.whiten(self.forward.whitening_filter)

        history = FitHistory(store_theta=store_theta, store_gamma=store_gamma, store_sigma_b=store_sigma_b)
        mu, cv_results = self._select_mu(data, mu, do_crossvalidation, mus, tol, n_splits, n_workers, use_ES)

        solver = self._new_solver()
        solver.run(data, mu, tol, history, verbose)
        model = NCRFModel._from_solver(solver, data)

        residual = model.eval_obj(data)
        explained_var = model.explained_variance(data)
        if compute_explained_variance:
            voxelwise = model.voxelwise_explained_variance(data)
        else:
            voxelwise = None

        return NCRFResult(
            model, explained_var=explained_var, voxelwise_explained_variance=voxelwise,
            residual=residual, history=history, cv_results=cv_results,
        )

    def _select_mu(
            self,
            data: RegressionData,
            mu: MuArg,
            do_crossvalidation: bool,
            mus: MusArg,
            tol: float,
            n_splits: int,
            n_workers: int,
            use_ES: bool,
    ) -> tuple[float, list[CVResult] | None]:
        """Choose the regularization parameter, running cross-validation if requested.

        Returns the chosen ``mu`` and, when cross-validation was performed, the
        list of :class:`CVResult`.
        """
        logger = logging.getLogger(__name__)
        if not do_crossvalidation:
            if mu is None:
                raise TypeError(f'{mu=}: fit needs mu to be a number or "auto"')
            return mu, None

        if mus == 'auto':
            mus = find_mu_range(self._new_solver().gradient(data))
        logger.info('Crossvalidation initiated!')
        cv_results = crossvalidate(self, data, mus, tol, n_splits, n_workers)
        best_cv = min(cv_results, key=attrgetter('cross_fit'))
        if best_cv.mu == min(mus):
            logger.info(f'CVmu is {best_cv.mu}: extending range of mu towards left')
            new_mus = np.logspace(np.log10(best_cv.mu) - 1, np.log10(best_cv.mu), 4)[:-1]
        elif best_cv.mu == max(mus):
            logger.info(f'CVmu is {best_cv.mu}: extending range of mu towards right')
            new_mus = np.logspace(np.log10(best_cv.mu), np.log10(best_cv.mu) + 1, 4)[1:]
        else:
            new_mus = None

        if new_mus is not None:
            cv_results.extend(crossvalidate(self, data, new_mus, tol, n_splits, n_workers))
            best_cv = min(cv_results, key=attrgetter('cross_fit'))

        mu = best_cv.mu
        if use_ES:
            cv_results_ = sorted(cv_results, key=attrgetter('mu'))
            if mu == cv_results[-1].mu:
                logger.info(f'\nCVmu is {best_cv.mu}: could not find mu based on estimation stability criterion\nContinuing with cross-validation only.')
            else:
                best_es = None
                for i, res in enumerate(cv_results_):
                    if res.mu < mu:
                        continue
                    else:
                        try:
                            if res.estimation_stability < cv_results_[i + 1].estimation_stability:
                                best_es = res
                                break
                        except IndexError:
                            best_es = None
                if best_es is None:
                    logger.warning('\nNo ES minima found: could not find mu based on estimation stability criterion.\nContinuing with cross-validation only.')
                else:
                    mu = best_es.mu
        return mu, cv_results


class NCRFResult:
    """Report produced by :meth:`NCRF.fit`.

    Bundles the fitted :class:`NCRFModel` with the training-set evaluation and the
    fitting provenance.  Model-level quantities (``h``, ``theta``, ``mu``, …) live
    on :attr:`model`; this object holds only what is specific to *this* fit.

    Attributes
    ----------
    model
        The fitted :class:`NCRFModel` (frozen weights + prediction/evaluation API).
    explained_var
        Fraction of total variance explained, evaluated on the training data. For
        an arbitrary dataset use :meth:`model.explained_variance`.
    voxelwise_explained_variance
        Source-wise contributions to explained variance on the training data
        (``None`` unless requested at fit time).
    residual
        The fit error, i.e. ``model.eval_obj`` on the training data.
    history
        Per-iteration :class:`FitHistory` accumulated during fitting.
    """
    _name = 'cTRFs estimator'

    def __init__(
            self,
            model: NCRFModel,
            *,
            explained_var: float,
            voxelwise_explained_variance: NDVar | None,
            residual: float,
            history: FitHistory,
            cv_results: list[CVResult] | None,
    ) -> None:
        self.model = model
        self.explained_var = explained_var
        self.voxelwise_explained_variance = voxelwise_explained_variance
        self.residual = residual
        self.history = history
        self._cv_results = cv_results

    def __repr__(self) -> str:
        forward = self.model.forward
        orientation = 'free' if forward.space else 'fixed'
        return f"<[{orientation} orientation] {self._name} on {forward.source!r}>"

    def cv_info(self) -> fmtxt.Table:
        """Summarize stored cross-validation scores in a table."""
        if self._cv_results is None:
            raise ValueError("CV: no cross-validation was performed. Use mu='auto' to perform cross-validation.")
        cv_results = sorted(self._cv_results, key=attrgetter('mu'))
        criteria = ('cross-fit', 'l2/mu')
        best_mu = {criterion: self.cv_mu(criterion) for criterion in criteria}

        table = fmtxt.Table('lllll')
        table.cells('mu', 'cross-fit', 'l2-error', 'weighted l2-error', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in cv_results:
            table.cell(fmtxt.stat(result.mu, fmt=fmt))
            star = 1 if result.mu is best_mu['cross-fit'] else 0
            table.cell(fmtxt.stat(result.cross_fit, fmt, star, 1))
            star = 1 if result.mu is best_mu['l2/mu'] else 0
            table.cell(fmtxt.stat(result.l2_error, fmt, star, 1))
            table.cell(fmtxt.stat(result.weighted_l2_error, fmt=fmt))
            table.cell(fmtxt.stat(result.estimation_stability, fmt=fmt))
        # warnings
        mus = [res.mu for res in self._cv_results]
        warnings = []
        if self.model.mu == min(mus):
            warnings.append("Best mu is smallest mu")
        if warnings:
            table.caption(f"Warnings: {'; '.join(warnings)}")
        return table

    def cv_mu(self, criterion: str = 'cross-fit') -> float:
        """Retrieve best mu based on cross-validation

        Parameters
        ----------
        criterion
            Criterion for best fit. Possible values:

            - ``'cross-fit'``: The smallest cross-fit value (default)
            - ``'l2'``: The smallest l2 error
            - ``'l2/mu'``: The local minimum in the l2 error with smallest mu
        """
        if criterion == 'cross-fit':
            best_cv = min(self._cv_results, key=attrgetter('cross_fit'))
        elif criterion == 'l2':
            best_cv = min(self._cv_results, key=attrgetter('l2_error'))
        elif criterion == 'l2/mu':
            cv_results = sorted(self._cv_results, key=attrgetter('mu'))
            peaks, _ = find_peaks([-result.l2_error for result in cv_results])  # find local minima
            if len(peaks) > 0:
                # higher mu -> smaller trf
                best_cv = max([cv_results[peak] for peak in peaks], key=attrgetter('mu'))
            else:
                best_cv = min(cv_results, key=attrgetter('l2_error'))
        else:
            raise ValueError(f'criterion={criterion}')
        return best_cv.mu
