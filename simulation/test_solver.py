import sys
import os

# --- PATH FIX START ---
# Get the absolute path to the folder where this script lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go UP one level to the project root ('2D-Maxwell-Solver')
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to Python's search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- PATH FIX END ---

# Now the import will work because Python knows where to look
from maxwell_solver import MaxwellSolver2D, GaussianPulse, FieldVisualizer

def main():
    print("--- 🧪 Running Simple Solver Test ---")

    # 1. Setup a small 2m x 2m box
    domain_size = (2.0, 2.0)
    pml_width = 0.3
    resolution = 15  # Low resolution for speed
    
    print(f"Initializing solver ({resolution} cells/m)...")
    solver = MaxwellSolver2D(domain_size, pml_width, resolution)
    
    # 2. Add a Pulse in the center (Polarized Vertical 'y')
    pulse = GaussianPulse(
        center=(1.0, 1.0),
        width=0.15,
        frequency=1e9,
        polarization='y' 
    )
    solver.set_source(pulse)
    
    # 3. Run for a short time (10 nanoseconds)
    dt = 0.2 / (resolution * 3e8)
    T_final = 10e-9 
    
    print(f"Simulating {int(T_final/dt)} time steps...")
    solver.solve(T_final, dt, output_interval=10, verbose=True)
    
    # 4. Save the Result
    print("Saving GIF...")
    vis = FieldVisualizer()
    vis.create_animation(solver, "test_result.gif", frame_skip=2)
    print("✅ Done! Open 'test_result.gif' to see the wave.")

if __name__ == "__main__":
    main()