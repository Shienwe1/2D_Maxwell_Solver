# File: scripts/run_convergence.py
import sys
import os
import numpy as np

# Path magic to find the package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from maxwell_solver import ConvergenceAnalyzer, GaussianPulse

def main():
    print("--- 📉 Starting Convergence Analysis ---")
    
    # 1. Define the Test Case
    # We use a smaller domain and shorter time to keep the test fast
    domain_size = (1.0, 1.0)
    pml_width = 0.2
    
    # Define resolutions to test (doubling each time)
    # e.g., 10, 20, 40 cells per meter
    resolutions = [10, 20, 40]
    
    # Define Source
    pulse = GaussianPulse(
        center=(0.5, 0.5), 
        width=0.1, 
        frequency=1e9, 
        polarization='y'
    )
    
    # Define Time Parameters
    # We run for a short duration so the pulse exists but hasn't hit the boundary yet
    T_final = 2.0e-9 
    
    # 2. Run the Study
    # This will run the solver 3 times (once for each resolution)
    results = ConvergenceAnalyzer.convergence_study(
        domain_size, 
        pml_width, 
        resolutions, 
        source=pulse,
        T_final=T_final,
        CFL=0.2,       # Keep CFL constant to ensure stability
        verbose=True
    )
    
    # 3. Compute Convergence Rate
    rate = ConvergenceAnalyzer.compute_convergence_rate(results)
    
    print("\n" + "="*40)
    print(f"RESULTS SUMMARY")
    print("="*40)
    for res in results:
        print(f"Res: {res['resolution']} | h: {res['h']:.4f} | Error (L2): {res['error_norm']:.6e}")
        
    print("-" * 40)
    print(f"Empirical Convergence Rate: {rate:.2f}")
    print("="*40)
    
    if rate > 1.5:
        print("✅ SUCCESS: Solver is converging at ~2nd order.")
    elif rate > 0.8:
        print("⚠️ WARNING: Solver is converging, but rate is low (1st order?).")
    else:
        print("❌ FAILURE: Solver is not converging properly.")

    # 4. Plot
    ConvergenceAnalyzer.plot_convergence(results, "convergence_plot.png")

if __name__ == "__main__":
    main()