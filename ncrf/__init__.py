"""Public package surface for the NCRF fitting pipeline.

The package is organized around a small top-level API: :func:`fit_ncrf`
coordinates input normalization and model fitting, while :class:`NCRF` and
:class:`RegressionData` expose the lower-level object-oriented workflow.
"""

from ._data import RegressionData
from ._model import NCRF, NCRFModel, NCRFResult
from ._solver import FitHistory
from ._ncrf import fit_ncrf
