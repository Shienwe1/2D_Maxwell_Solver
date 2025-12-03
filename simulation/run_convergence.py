# ============================================================================
# FILE: simulation/run_convergence.py (FIXED FOR DOLFINX v0.8+)
# ============================================================================
import sys
import os
import numpy as np

# Path fix to find the package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from maxwell_solver.convergence import ConvergenceAnalyzer
from maxwell_solver.sources import GaussianPulse
from maxwell_solver import MaxwellSolver2D
from dolfinx import fem
# --- FIXED IMPORT ---
from dolfinx.fem import create_interpolation_data
import ufl
from mpi4py import MPI

def interpolate_nonmatching(target_function, source_function):
    """
    Safely interpolate source_function onto target_function 
    using the modern DOLFINx v0.8+ API.
    """
    V_target = target_function.function_space
    V_source = source_function.function_space
    
    # 1. Identify cells to interpolate (all cells on this process)
    # The new API requires identifying which cells we are mapping to.
    mesh_target = V_target.mesh
    tdim = mesh_target.topology.dim
    num_cells = mesh_target.topology.index_map(tdim).size_local + \
                mesh_target.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)

    # 2. Create Interpolation Data (The "Collision Map")
    # This replaces the old 'create_nonmatching_meshes_interpolation_data'
    interpolation_data = create_interpolation_data(
        V_target, 
        V_source, 
        cells, 
        padding=1.0e-6
    )
    
    # 3. Perform the Interpolation
    # Try the v0.9 method first, fall back to v0.8 style if needed
    if hasattr(target_function, "interpolate_nonmatching"):
        # DOLFINx v0.9+ style
        target_function.interpolate_nonmatching(
            source_function, 
            cells, 
            interpolation_data=interpolation_data
        )
    else:
        # DOLFINx v0.8 style
        target_function.interpolate(
            source_function, 
            cells=cells, 
            nmm_interpolation_data=interpolation_data
        )

def run_self_convergence():
    print("--- 📉 Starting Self-Convergence Analysis ---")
    
    # Define Parameters
    domain = (1.0, 1.0)
    pml = 0.2
    resolutions = [10, 20, 40]
    T_final = 1.0e-9 
    CFL = 0.05
    
    source = GaussianPulse(center=(0.5, 0.5), width=0.1, frequency=1e9, polarization='y')
    
    solutions = []
    
    # 1. Run Simulations
    for res in resolutions:
        print(f"\nRunning Resolution: {res}...")
        solver = MaxwellSolver2D(domain, pml, res)
        solver.set_source(source)
        dt = CFL / (res * solver.c0)
        solver.solve(T_final, dt, output_interval=100000, verbose=False)
        solutions.append(solver.E_n)

    # 2. Compute Errors Relative to Finest Mesh (N=40)
    fine_sol = solutions[-1]
    errors = []
    
    print("\nCalculating Errors relative to finest mesh (N=40)...")
    V_fine = fine_sol.function_space
    
    for i in range(len(resolutions) - 1):
        res = resolutions[i]
        sol_coarse = solutions[i]
        
        # Project coarse solution onto fine mesh
        sol_projected = fem.Function(V_fine)
        interpolate_nonmatching(sol_projected, sol_coarse)
        
        # Calculate Difference
        diff = sol_projected - fine_sol
        error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * ufl.dx)), 
            op=MPI.SUM
        ))
        
        errors.append(error_L2)
        print(f"Res {res} vs Res 40: Error L2 = {error_L2:.6e}")

    # 3. Calculate Convergence Rate
    if len(errors) >= 2:
        e1 = errors[0] # Error of N=10
        e2 = errors[1] # Error of N=20
        r_ratio = resolutions[1] / resolutions[0] 
        
        rate = np.log(e1 / e2) / np.log(r_ratio)
        print(f"\nEstimated Convergence Rate: {rate:.2f}")
        
        if rate > 1.5:
            print("✅ Excellent! (Approaching 2nd Order)")
        elif rate > 0.9:
            print("✅ Good. (Linear to Super-linear)")
        else:
            print("⚠️ Low convergence rate.")

if __name__ == "__main__":
    run_self_convergence()