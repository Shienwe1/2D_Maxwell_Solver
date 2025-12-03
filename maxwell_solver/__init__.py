"""
FEniCSx Maxwell Solver Package
"""
__version__ = "1.0.0"

from .solver import MaxwellSolver2D
from .pml import PMLCoefficients
from .convergence import ConvergenceAnalyzer
from .visualization import FieldVisualizer
from .sources import GaussianPulse, PlaneWave, DipoleSources

__all__ = [
    'MaxwellSolver2D',
    'PMLCoefficients',
    'ConvergenceAnalyzer',
    'FieldVisualizer',
    'GaussianPulse',
    'PlaneWave',
    'DipoleSources'
]
