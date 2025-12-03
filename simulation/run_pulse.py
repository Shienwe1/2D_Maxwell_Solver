# File: scripts/run_pulse.py
import sys
import os

# 1. Add the parent directory to Python path so it can find 'maxwell_solver'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from maxwell_solver import MaxwellSolver2D, GaussianPulse, FieldVisualizer

def main():
    print("--- Starting Gaussian Pulse Simulation ---")

    # 2. Define Parameters
    # A 2x2 meter box with 20 cells per meter resolution
    domain = (2.0, 2.0)
    pml_thickness = 0.3
    resolution = 20  
    
    # 3. Initialize Solver
    solver = MaxwellSolver2D(domain, pml_thickness, resolution)
    
    # 4. Set Source (Polarized in Y-direction)
    pulse = GaussianPulse(
        center=(1.0, 1.0), 
        width=0.15, 
        frequency=1e9, 
        polarization='y'
    )
    solver.set_source(pulse)
    
    # 5. Run Simulation (10 nanoseconds)
    # Calculate stable time step (CFL condition)
    dt = 0.2 / (resolution * solver.c0)
    T_final = 10e-9
    
    print(f"Time step: {dt:.3e} s")
    solver.solve(T_final, dt, output_interval=10)
    
    # 6. Visualize
    print("Generating GIF...")
    vis = FieldVisualizer()
    vis.create_animation(solver, "gaussian_pulse.gif")
    print("Done! Check 'gaussian_pulse.gif'.")

if __name__ == "__main__":
    main()