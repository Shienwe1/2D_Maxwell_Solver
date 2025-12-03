"""
Example 2: Plane Wave Propagation
----------------------------------
A uniform wave traveling across the domain.
Best for: Testing absorption at boundaries, studying wave propagation
"""
import sys
sys.path.insert(0, '..')

from maxwell_solver import MaxwellSolver2D, PlaneWave, FieldVisualizer


def run_plane_wave():
    """Plane wave traveling at 45 degrees"""
    print("\n" + "="*60)
    print("Example 2: Plane Wave at 45°")
    print("="*60)
    
    # Domain setup
    domain_size = (3.0, 3.0)  # Larger domain for plane wave
    pml_width = 0.4
    resolution = 15           # Lower resolution for speed
    
    # Create solver
    print("Creating solver...")
    solver = MaxwellSolver2D(domain_size, pml_width, resolution)
    
    # Setup plane wave traveling at 45 degrees
    source = PlaneWave(
        direction=(1, 1),       # Will be normalized to (0.707, 0.707)
        frequency=2e9,          # 2 GHz (wavelength ~15cm)
        amplitude=1.0,
        phase=0.0               # Starting phase
    )
    solver.set_source(source)
    
    # Time integration
    CFL = 0.5
    dt = CFL / (resolution * solver.c0)
    T_final = 20e-9  # 20 ns to see wave travel across domain
    
    print(f"Wavelength: {solver.c0/2e9:.3f} m")
    print(f"Time step: {dt:.3e} s")
    
    # Solve
    solver.solve(T_final, dt, output_interval=30)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualizer = FieldVisualizer()
    visualizer.visualize_fields(solver, "../output/example02_fields.png")
    visualizer.create_animation(solver, "../output/example02_animation.gif", frame_skip=2)
    
    print("\n✅ Complete! Watch the plane wave cross the domain.")


if __name__ == "__main__":
    run_plane_wave()