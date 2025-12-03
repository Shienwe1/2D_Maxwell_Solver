"""
Example 1: Gaussian Pulse Propagation
--------------------------------------
A localized pulse emanating from the center of the domain.
Best for: Testing solver stability, vector field orientation, and PML absorption.
"""
import sys
import os

# 1. Path Magic: Ensure we can import the package from the parent directory
# This allows running the script from anywhere (IDE or Terminal)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can import our package
from maxwell_solver import MaxwellSolver2D, GaussianPulse, FieldVisualizer

def run_gaussian_pulse():
    """Simple Gaussian pulse in empty space"""
    print("\n" + "="*60)
    print("Example 1: Gaussian Pulse Propagation (Vector TE Mode)")
    print("="*60)
    
    # 2. Setup Output Directory
    output_dir = os.path.join(current_dir, "..", "..", "output")
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # 3. Domain Parameters
    # A 2m x 2m box
    domain_size = (2.0, 2.0)
    # A 0.3m absorbing layer on all sides
    pml_width = 0.3
    # Resolution: 20 cells per meter (higher = more accurate but slower)
    resolution = 20
    
    # 4. Initialize Solver
    print(f"[1/4] Initializing Solver (Size: {domain_size}, Res: {resolution})...")
    solver = MaxwellSolver2D(domain_size, pml_width, resolution)
    
    # 5. Setup Source
    print("[2/4] Configuring Gaussian Source...")
    # We place a pulse at the exact center (1.0, 1.0)
    # Polarization='y' means the Electric field oscillates vertically (Ey)
    source = GaussianPulse(
        center=(1.0, 1.0),
        width=0.15,             # Spatial width (sigma)
        frequency=1e9,          # 1 GHz frequency
        amplitude=1.0,          # 1 V/m peak
        polarization='y'        # Excite Ey component
    )
    solver.set_source(source)
    
    # 6. Time Integration Setup
    # CFL (Courant-Friedrichs-Lewy) condition for stability
    # For unstructured triangular meshes, 0.2 is a safe starting point.
    CFL_safety = 0.2
    dt = CFL_safety / (resolution * solver.c0)
    
    # Total simulation time: 10 nanoseconds (enough for pulse to reach the wall)
    T_final = 10e-9
    
    print(f"      Time step: {dt:.3e} s")
    print(f"      Total steps: {int(T_final/dt)}")
    
    # 7. Run the Simulation Loop
    print("[3/4] Running Time Stepping...")
    # solve() returns the final state functions, but we mostly care about the history list it builds
    solver.solve(T_final, dt, output_interval=10, verbose=True)
    
    # 8. Visualization
    print("[4/4] Generating Visualization...")
    visualizer = FieldVisualizer()
    
    # Define file paths
    fields_path = os.path.join(output_dir, "example01_fields.png")
    anim_path = os.path.join(output_dir, "example01_animation.gif")
    
    # A. Save a static snapshot of the final state
    visualizer.visualize_fields(solver, fields_path)
    
    # B. Generate the GIF animation
    # frame_skip=5 means we only animate every 5th saved frame to keep file size small
    if len(solver.Ez_history) > 0:
        visualizer.create_animation(solver, anim_path, frame_skip=2)
    else:
        print("Warning: Not enough history data to create animation.")
    
    print("\n" + "="*60)
    print("✅ Simulation Complete!")
    print(f"Check your output files here:\n  {output_dir}")
    print("="*60)

if __name__ == "__main__":
    run_gaussian_pulse()