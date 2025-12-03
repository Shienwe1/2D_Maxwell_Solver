import numpy as np
from dolfinx import fem
import basix.ufl

class PMLCoefficients:
    """Calculate PML absorption coefficients for centered domain"""
    
    def __init__(self, domain_limit, pml_width, sigma_max=20.0, alpha=2.0):
        """
        Parameters:
        -----------
        domain_limit : float OR tuple
            The extent of the domain.
            If float: symmetric domain [-L, L]
            If tuple: (Lx, Ly) implies domain [0, Lx] x [0, Ly] based on solver setup
        """
        # --- CRITICAL FIX: Handle Tuple vs Float ---
        if isinstance(domain_limit, (tuple, list, np.ndarray)):
            # Assuming solver uses [0, Lx] x [0, Ly], the "limit" for centered math 
            # might need adjustment. 
            # BUT, your solver creates [0, Lx], so center is Lx/2.
            # Your current PML math assumes centered at 0 ([-L, L]).
            
            # Let's standardize: Use the tuple directly.
            self.Lx = domain_limit[0]
            self.Ly = domain_limit[1]
        else:
            self.Lx = domain_limit
            self.Ly = domain_limit

        self.w = pml_width
        self.sigma_max = sigma_max
        self.alpha = alpha
        
    def _profile(self, dist):
        """Helper to calculate the polynomial ramp"""
        val = np.maximum(0, dist)
        return self.sigma_max * (val / self.w) ** self.alpha

    def sigma_x(self, x):
        """PML absorption in x-direction"""
        # Solver domain is [0, Lx]
        # Left Wall: x < w
        # Right Wall: x > Lx - w
        
        val = np.zeros_like(x[0])
        
        # Left PML (x approaches 0)
        dist_left = self.w - x[0]
        val = np.maximum(val, self._profile(dist_left))
        
        # Right PML (x approaches Lx)
        dist_right = x[0] - (self.Lx - self.w)
        val = np.maximum(val, self._profile(dist_right))
        
        return val
    
    def sigma_y(self, x):
        """PML absorption in y-direction"""
        # Solver domain is [0, Ly]
        
        val = np.zeros_like(x[1])
        
        # Bottom PML (y approaches 0)
        dist_bottom = self.w - x[1]
        val = np.maximum(val, self._profile(dist_bottom))
        
        # Top PML (y approaches Ly)
        dist_top = x[1] - (self.Ly - self.w)
        val = np.maximum(val, self._profile(dist_top))
        
        return val

    def sigma_combined(self, x):
        """Returns max(sigma_x, sigma_y) for isotropic approximation"""
        sx = self.sigma_x(x)
        sy = self.sigma_y(x)
        return np.maximum(sx, sy)
    
    def get_pml_functions(self, domain):
        """Create FEniCS functions for PML coefficients"""
        cell_name = domain.topology.cell_name()
        el_sigma = basix.ufl.element("DG", cell_name, 1)
        Q = fem.functionspace(domain, el_sigma)
        
        s_x = fem.Function(Q)
        s_y = fem.Function(Q)
        s_comb = fem.Function(Q)
        
        s_x.interpolate(self.sigma_x)
        s_y.interpolate(self.sigma_y)
        s_comb.interpolate(self.sigma_combined)
        
        return s_x, s_y, s_comb