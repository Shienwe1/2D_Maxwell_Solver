"""
Example 4: Interference Pattern from Two Sources
-------------------------------------------------
Two Gaussian pulses creating interference patterns.
Best for: Understanding wave interference, testing linearity
"""
import sys
import os
import numpy as np

# 1. ROBUST PATH SETUP
# Ensures we can find 'maxwell_solver' regardless of where we run this script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from maxwell_solver import MaxwellSolver2D, FieldVisualizer
from maxwell_solver.sources import SourceBase

class DualGaussianSource(SourceBase):
    """Two Gaussian pulses separated in space with optional phase shift"""
    
    def __init__(self, center1, center2, width, amplitude=1.0, frequency=1e9, phase_diff=0):
        super().__init__(amplitude, frequency)
        self.center1 = center1
        self.center2 = center2
        self.width = width
        self.phase_diff = phase_diff
    
    def __call__(self, x, t):
        # --- Source 1 Logic ---
        r1_sq = (x[0] - self.center1[0])**2 + (x[1] - self.center1[1])**2
        spatial1 = np.exp(-r1_sq / (2 * self.width**2))
        
        # --- Source 2 Logic ---
        r2_sq = (x[0] - self.center2[0])**2 + (x[1] - self.center2[1])**2
        spatial2 = np.exp(-r2_sq / (2 * self.width**2))
        
        # --- Temporal Logic (Correct Phase Shift) ---
        t0 = 4.0 / self.omega
        sigma_t = 1.5 / self.omega
        
        # Base temporal pulse
        envelope = np.exp(-((t - t0)**2) / (2 * sigma_t**2))
        
        # Source 1 oscillates at sin(wt)
        wave1 = np.sin(self.omega * t) * envelope
        
        # Source 2 oscillates at sin(wt + phase)
        # This is the physically correct way to shift phase
        wave2 = np.sin(self.omega * t + self.phase_diff) * envelope
        
        # Combine magnitudes
        total_magnitude = self.amplitude * (spatial1 * wave1 + spatial2 * wave2)
        
        # --- CRITICAL FIX: VECTOR RETURN ---
        # We polarize the wave in the Y-direction (Ey) to match the TE solver expectations.
        zeros = np.zeros_like(total_magnitude)
        return (zeros, total_magnitude)


def run_dual_sources():
    """Two sources creating interference"""
    print("\n" + "="*60)
    print("Example 4: Two-Source Interference (Vector Mode)")
    print("="*60)
    
    # Setup Output Directory
    output_dir = os.path.join(current_dir, "..", "..", "output")
    if not os.path.exists(output_dir):
        os.makedirs
