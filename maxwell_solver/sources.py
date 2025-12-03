# ============================================================================
# FILE: maxwell_solver/sources.py (COMPLETE & FIXED)
# ============================================================================
import numpy as np

class SourceBase:
    """Base class for electromagnetic sources"""
    
    def __init__(self, amplitude=1.0, frequency=1e9):
        self.amplitude = amplitude
        self.omega = 2 * np.pi * frequency
        self.frequency = frequency
        
    def __call__(self, x, t):
        raise NotImplementedError

class GaussianPulse(SourceBase):
    """
    Gaussian pulse source with selectable polarization.
    """
    def __init__(self, center, width, amplitude=1.0, frequency=1e9, polarization='y'):
        super().__init__(amplitude, frequency)
        self.center = center
        self.width = width
        self.polarization = polarization.lower()
        
    def __call__(self, x, t):
        # x is a numpy array of coordinates with shape (3, num_points)
        r2 = (x[0] - self.center[0])**2 + (x[1] - self.center[1])**2
        
        # Spatial Profile
        spatial = np.exp(-r2 / (2 * self.width**2))
        
        # Temporal Profile
        t0 = 4.0 / self.omega 
        sigma_t = 1.5 / self.omega
        temporal = np.sin(self.omega * t) * np.exp(-((t - t0)**2) / (2 * sigma_t**2))
        
        val = self.amplitude * spatial * temporal
        
        # Return Vector Tuple (Ex, Ey)
        zeros = np.zeros_like(val)
        if self.polarization == 'x':
            return (val, zeros)
        elif self.polarization == 'y':
            return (zeros, val)
        else:
            return (val, val)

class PlaneWave(SourceBase):
    """
    Plane wave source (approximate implementation for localized injection).
    """
    def __init__(self, direction, amplitude=1.0, frequency=1e9, phase=0.0):
        super().__init__(amplitude, frequency)
        # Normalize direction vector
        d = np.array(direction)
        norm = np.linalg.norm(d)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero")
        self.direction = d / norm
        self.phase = phase
        
    def __call__(self, x, t):
        # k = omega / c (approximate in free space)
        c0 = 299792458.0
        k = self.omega / c0
        
        # Phase term k * (d . x)
        spatial_phase = k * (self.direction[0] * x[0] + self.direction[1] * x[1])
        
        # Wave equation
        val = self.amplitude * np.sin(self.omega * t - spatial_phase + self.phase)
        
        # Determine polarization perpendicular to propagation
        # If propagating in X, E is in Y
        if abs(self.direction[0]) > abs(self.direction[1]):
            return (np.zeros_like(val), val) # Ey
        else:
            return (val, np.zeros_like(val)) # Ex

class DipoleSources(SourceBase):
    """
    Electric dipole source (Point source singularity).
    """
    def __init__(self, position, moment=1.0, amplitude=1.0, frequency=1e9, polarization='y'):
        super().__init__(amplitude, frequency)
        self.position = position
        self.moment = moment
        self.polarization = polarization.lower()
        
    def __call__(self, x, t):
        r = np.sqrt((x[0] - self.position[0])**2 + (x[1] - self.position[1])**2)
        # Avoid division by zero
        r = np.maximum(r, 1e-6)
        
        # Decay 1/r^2
        spatial = self.moment / (r**2 + 0.01)
        temporal = np.sin(self.omega * t)
        
        val = self.amplitude * spatial * temporal
        
        zeros = np.zeros_like(val)
        if self.polarization == 'x':
            return (val, zeros)
        else:
            return (zeros, val)