
import numpy as np
import pyvista as pv
import dolfinx.plot
from dolfinx import fem  
import basix.ufl         

class FieldVisualizer:
    """Visualize electromagnetic fields"""
    
    @staticmethod
    def create_animation(solver, output_file="animation.gif", frame_skip=2, clim=None):
        if len(solver.E_history) < 5:
            print("Not enough frames for animation")
            return

        print(f"Creating animation with {len(solver.E_history)} frames...")
        plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
        plotter.open_gif(output_file, fps=10)

        # 1. Setup Interpolation Space (DG1 Vector - 2D)
        # --- FIXED: Use shape=(2,) to match the 2D Nedelec Solver ---
        cell_name = solver.domain.topology.cell_name()
        el_plot = basix.ufl.element("DG", cell_name, 1, shape=(2,)) 
        V_plot = fem.functionspace(solver.domain, el_plot)
        u_plot = fem.Function(V_plot)
        
        # 2. Setup Reconstruction Function
        u_reconstruct = fem.Function(solver.V_E)

        # 3. Setup Grid
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V_plot)
        
        # Determine Color Limits
        if clim is None:
            u_reconstruct.x.array[:] = solver.E_history[0]
            u_plot.interpolate(u_reconstruct)
            
            # Reshape as 2D vectors
            vecs_2d = u_plot.x.array.reshape((-1, 2))
            mags = np.linalg.norm(vecs_2d, axis=1)
            clim = [0, mags.max() if mags.max() > 0 else 1.0]

        # 4. Loop Frames
        for i, data_snapshot in enumerate(solver.E_history):
            if i % frame_skip != 0:
                continue

            # A. Load history data
            u_reconstruct.x.array[:] = data_snapshot
            
            # B. Interpolate Nedelec (2D) -> DG (2D)
            # This now works because both spaces are 2D!
            u_plot.interpolate(u_reconstruct)
            
            # C. Extract 2D Vectors
            vecs_2d = u_plot.x.array.reshape((-1, 2))
            
            # --- CRITICAL FIX: Pad with Zeros for PyVista (2D -> 3D) ---
            # PyVista expects (N, 3) for vectors, even if Z is zero.
            zeros = np.zeros((vecs_2d.shape[0], 1))
            vecs_3d = np.hstack((vecs_2d, zeros))
            
            visual_mags = np.linalg.norm(vecs_3d, axis=1)

            # D. Update Grid
            grid = pv.UnstructuredGrid(topology, cell_types, geometry)
            grid.point_data["Magnitude"] = visual_mags
            grid.point_data["E_Field"] = vecs_3d  # Store full vectors just in case
            
            plotter.clear()
            plotter.add_mesh(grid, scalars="Magnitude", cmap="inferno", 
                           clim=clim, show_edges=False)
            
            time_val = solver.times[i] if i < len(solver.times) else 0.0
            plotter.add_text(f"Time: {time_val:.3e} s", font_size=12)
            plotter.view_xy()
            plotter.write_frame()

        plotter.close()
        print(f"Animation saved to {output_file}")