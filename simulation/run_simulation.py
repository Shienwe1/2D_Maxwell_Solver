"""
Main simulation script - Run this file to execute simulations
"""
import sys
sys.path.insert(0, '..')  # Add parent directory to path

from maxwell_solver import (MaxwellSolver2D, GaussianPulse, 
                            FieldVisualizer, ConvergenceAnalyzer)


def main():
    """Main simulation workflow"""
    print("="*60)
    print("FEniCSx 2D Maxwell Solver")
    print("="*60)
    
    # Domain parameters
    domain_size = (2.0, 2.0)
    pml_width = 0.3
    resolution = 20
    
    # Create solver
    print("\n[1/4] Initializing solver...")
    solver = MaxwellSolver2D(domain_size, pml_width, resolution)
    
    # Setup source
    print("[2/4] Setting up Gaussian pulse source...")
    source = GaussianPulse(
        center=(1.0, 1.0),
        width=0.1,
        frequency=1e9,
        amplitude=1.0
    )
    solver.set_source(source)
    
    # Time integration
    print("[3/4] Running time integration...")
    CFL = 0.5
    dt = CFL / (resolution * solver.c0)
    T_final = 10e-9
    
    print(f"  Time step: {dt:.3e} s")
    print(f"  Total steps: {int(T_final/dt)}")
    
    Ez, H = solver.solve(T_final, dt, output_interval=20)
    
    # Visualization
    print("[4/4] Generating visualizations...")
    visualizer = FieldVisualizer()
    visualizer.visualize_fields(solver, "output/fields.png")
    
    if len(solver.Ez_history) > 10:
        visualizer.create_animation(solver, "output/evolution.gif")
    
    # Convergence study
    print("\n" + "="*60)
    print("Running Convergence Analysis")
    print("="*60)
    
    analyzer = ConvergenceAnalyzer()
    resolutions = [10, 20, 30]
    errors = analyzer.convergence_study(
        domain_size, pml_width, resolutions, source,
        T_final=5e-9, dt_factor=0.5
    )
    
    rate = analyzer.compute_convergence_rate(errors)
    print(f"\nEmpirical convergence rate: {rate:.2f}")
    
    analyzer.plot_convergence(errors, "output/convergence.png")
    
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("="*60)
    print("Output files:")
    print("  - output/fields.png")
    print("  - output/evolution.gif")
    print("  - output/convergence.png")


if __name__ == "__main__":
    main()