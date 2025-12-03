
"""
Convergence analysis and error estimation for 2D Maxwell Solver.
"""
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import fem
import ufl
from mpi4py import MPI

from .solver import MaxwellSolver2D

class ConvergenceAnalyzer:
    """Analyze convergence rates and error estimates."""
    
    @staticmethod
    def compute_errors(solver, E_exact_func=None):
        """
        Compute L2 error norm against an exact solution (if provided).
        If E_exact_func is None, returns the L2 norm of the solution itself (Energy check).
        
        Parameters:
        -----------
        solver : MaxwellSolver2D
            The solver instance with the final state E_n.
        E_exact_func : function or None
            A Python function f(x, t) returning the exact vector field (Ex, Ey, Ez).
        """
        domain = solver.domain
        
        # 1. Get Numerical Solution
        E_num = solver.E_n
        
        # 2. Define Error Form
        if E_exact_func is not None:
            # Interpolate exact solution onto the same space
            E_exact = fem.Function(solver.V_E)
            E_exact.interpolate(lambda x: E_exact_func(x, solver.times[-1]))
            
            # Error = E_num - E_exact
            diff = E_num - E_exact
            # L2 Error = sqrt( integral( |diff|^2 ) )
            # ufl.inner for vectors is (dx*dx + dy*dy)
            error_form = fem.form(ufl.inner(diff, diff) * ufl.dx)
            
        else:
            # If no exact solution, just compute L2 norm of the field (Energy proxy)
            error_form = fem.form(ufl.inner(E_num, E_num) * ufl.dx)
            
        # 3. Assemble and calculate root
        local_error = fem.assemble_scalar(error_form)
        global_error = domain.comm.allreduce(local_error, op=MPI.SUM)
        
        return np.sqrt(global_error)

    @staticmethod
    def convergence_study(domain_size, pml_width, resolutions, source, 
                          T_final, CFL=0.2, verbose=True):
        """
        Perform a mesh refinement study to calculate convergence rate.
        
        Parameters:
        -----------
        resolutions : list of int
            List of mesh resolutions to test (e.g. [10, 20, 40]).
        source : SourceBase
            The source object to use for simulation.
        """
        results = []
        
        print(f"Running Convergence Study on resolutions: {resolutions}")
        
        for res in resolutions:
            if verbose:
                print(f"\n--- Testing Resolution: {res} cells/unit ---")
            
            # 1. Setup Solver
            solver = MaxwellSolver2D(domain_size, pml_width, res)
            solver.set_source(source)
            
            # 2. Calculate DT based on fixed CFL
            # Note: For convergence studies, we usually scale dt with h (dt ~ 1/N)
            dt = CFL / (res * solver.c0)
            
            # 3. Solve
            solver.solve(T_final, dt, output_interval=100000, verbose=False)
            
            # 4. Compute 'Error' (L2 Norm of solution as proxy for stability/conservation)
            # In a real rigorous test, you would compare against an analytical plane wave.
            norm_l2 = ConvergenceAnalyzer.compute_errors(solver)
            
            h = 1.0 / res
            results.append({
                'h': h,
                'resolution': res,
                'error_norm': norm_l2,
                'dt': dt
            })
            
            if verbose:
                print(f"   h = {h:.4f}, L2 Norm = {norm_l2:.6e}")
                
        return results

    @staticmethod
    def compute_convergence_rate(results):
        """
        Compute empirical convergence rate from results list.
        Rate p approx log2( error(2h) / error(h) )
        """
        if len(results) < 2:
            return 0.0
            
        # We use the last two data points (finest meshes)
        e1 = results[-2]['error_norm']
        e2 = results[-1]['error_norm']
        h1 = results[-2]['h']
        h2 = results[-1]['h']
        
        # Avoid division by zero
        if e2 == 0 or e1 == 0:
            return 0.0
            
        # Formula: p = log(e1/e2) / log(h1/h2)
        # Note: If checking pure energy conservation, this might be flat (0).
        # If checking error against exact solution, should be ~2.
        rate = np.log(e1 / e2) / np.log(h1 / h2)
        return rate

    @staticmethod
    def plot_convergence(results, output_file="convergence.png"):
        """Plot the convergence graph (log-log)."""
        h_values = [r['h'] for r in results]
        errors = [r['error_norm'] for r in results]
        
        plt.figure(figsize=(8, 6))
        plt.loglog(h_values, errors, 'o-', linewidth=2, label='Simulation Result')
        
        # Add reference slope (O(h^2))
        if len(h_values) > 0:
            ref_slope = [errors[0] * (h / h_values[0])**2 for h in h_values]
            plt.loglog(h_values, ref_slope, 'k--', alpha=0.5, label='Reference O(h²)')
        
        plt.xlabel('Mesh Size (h)')
        plt.ylabel('L2 Norm / Error')
        plt.title('Convergence Analysis')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        
        plt.savefig(output_file)
        plt.close()
        print(f"Convergence plot saved to {output_file}")
