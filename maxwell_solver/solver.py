# ============================================================================
# FILE: maxwell_solver/solver.py (FIXED FOR DOLFINX v0.8)
# ============================================================================
import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import Function, petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl 

from .pml import PMLCoefficients

class MaxwellSolver2D:
    """
    2D Maxwell's equations solver (TE Mode: E is vector, H is scalar).
    Uses Nedelec (1st kind) elements for E and DG for H.
    """
    
    def __init__(self, domain_size, pml_width, resolution, element_degree=1):
        self.Lx, self.Ly = domain_size
        self.pml_width = pml_width
        self.resolution = resolution
        self.degree = element_degree
        
        # Physical constants
        self.c0 = 299792458.0
        self.mu0 = 4 * np.pi * 1e-7
        self.eps0 = 8.854187817e-12
        
        self._create_mesh()
        self.pml = PMLCoefficients(domain_size, pml_width)
        self._setup_function_spaces()
        
        # State variables
        self.E_n = Function(self.V_E)   # Current E (Vector)
        self.E_np1 = Function(self.V_E) # Next E
        self.H_n = Function(self.V_H)   # Current H (Scalar)
        self.H_np1 = Function(self.V_H) # Next H
        
        self.source = None

    def _create_mesh(self):
        nx = int(self.Lx * self.resolution)
        ny = int(self.Ly * self.resolution)
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array([0.0, 0.0]), np.array([self.Lx, self.Ly])],
            [nx, ny],
            cell_type=mesh.CellType.triangle
        )

    def _setup_function_spaces(self):
        cell_name = self.domain.topology.cell_name()
        
        # Nedelec Element (1st kind) for Electric Field Vector
        el_E = basix.ufl.element("N1curl", cell_name, self.degree)
        self.V_E = fem.functionspace(self.domain, el_E)
        
        # Discontinuous Lagrange for Magnetic Field Scalar
        el_H = basix.ufl.element("DG", cell_name, self.degree - 1)
        self.V_H = fem.functionspace(self.domain, el_H)

    def set_source(self, source):
        self.source = source

    def _source_term(self, t):
        """Interpolate source into vector space using fem.Expression"""
        J_src = Function(self.V_E)
        if self.source is None:
            J_src.x.array[:] = 0.0
            return J_src
            
        # Define the source function wrapper (2D)
        def source_wrapper(x):
            # source returns (Ex, Ey) tuple
            vals = self.source(x, t)
            
            # Return 2D Array (2, num_points)
            return np.array([vals[0], vals[1]], dtype=PETSc.ScalarType)

        # Use 2D Lagrange Space for Intermediate Interpolation
        V_temp = fem.functionspace(self.domain, ("Lagrange", 2, (2,)))
        u_temp = Function(V_temp)
        u_temp.interpolate(source_wrapper)
        
        # Project/Interpolate from Lagrange (2D) -> Nedelec (2D)
        J_src.interpolate(u_temp)
        
        return J_src

    def setup_variational_forms(self, dt):
        self.dt = dt
        
        u_E = ufl.TrialFunction(self.V_E) 
        v_E = ufl.TestFunction(self.V_E)
        
        u_H = ufl.TrialFunction(self.V_H) 
        v_H = ufl.TestFunction(self.V_H)
        
        sx, sy, s_comb = self.pml.get_pml_functions(self.domain)
        
        # --- Step 1: Solve E ---
        E_mid_unknown = 0.5 * (u_E + self.E_n) 
        F_E = (self.eps0 * ufl.inner(u_E - self.E_n, v_E) * ufl.dx
               - self.dt * ufl.inner(self.H_n, ufl.curl(v_E)) * ufl.dx 
               + self.dt * ufl.inner(s_comb * E_mid_unknown, v_E) * ufl.dx)
               
        # --- Step 2: Solve H ---
        E_mid_known = 0.5 * (self.E_np1 + self.E_n) 
        F_H = (self.mu0 * ufl.inner(u_H - self.H_n, v_H) * ufl.dx
               + self.dt * ufl.inner(ufl.curl(E_mid_known), v_H) * ufl.dx)

        # Boundary Conditions
        self.bcs = []
        fdim = self.domain.topology.dim - 1
        facets = mesh.locate_entities_boundary(self.domain, fdim, lambda x: np.full(x.shape[1], True))
        dofs = fem.locate_dofs_topological(self.V_E, fdim, facets)
        
        u_bc = Function(self.V_E)
        u_bc.x.array[:] = 0.0
        self.bcs.append(fem.dirichletbc(u_bc, dofs))

        # Forms
        self.a_E = fem.form(ufl.lhs(F_E))
        self.L_E = fem.form(ufl.rhs(F_E))
        self.a_H = fem.form(ufl.lhs(F_H))
        self.L_H = fem.form(ufl.rhs(F_H))
        
        # Matrices
        self.A_E = petsc.assemble_matrix(self.a_E, bcs=self.bcs)
        self.A_E.assemble()
        self.A_H = petsc.assemble_matrix(self.a_H)
        self.A_H.assemble()
        
        # Vectors
        self.b_E_vec = self.A_E.createVecRight()
        self.b_H_vec = self.A_H.createVecRight()
        
        self._setup_solvers()

    def _setup_solvers(self):
        self.solver_E = PETSc.KSP().create(self.domain.comm)
        self.solver_E.setOperators(self.A_E)
        self.solver_E.setType("preonly")
        self.solver_E.getPC().setType("lu")
        
        self.solver_H = PETSc.KSP().create(self.domain.comm)
        self.solver_H.setOperators(self.A_H)
        self.solver_H.setType("preonly")
        self.solver_H.getPC().setType("lu")

    def time_step(self, t):
        # 1. Update E
        J = self._source_term(t + self.dt/2)
        
        with self.b_E_vec.localForm() as loc:
            loc.set(0)
        
        petsc.assemble_vector(self.b_E_vec, self.L_E)
        petsc.assemble_vector(self.b_E_vec, fem.form(-self.dt * ufl.inner(J, ufl.TestFunction(self.V_E)) * ufl.dx))
        
        petsc.apply_lifting(self.b_E_vec, [self.a_E], [self.bcs])
        self.b_E_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self.b_E_vec, self.bcs)
        
        # --- FIXED: Use .x.petsc_vec instead of .vector ---
        self.solver_E.solve(self.b_E_vec, self.E_np1.x.petsc_vec)
        self.E_np1.x.scatter_forward()
        
        # 2. Update H
        with self.b_H_vec.localForm() as loc:
            loc.set(0)
            
        petsc.assemble_vector(self.b_H_vec, self.L_H)
        self.b_H_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # --- FIXED: Use .x.petsc_vec ---
        self.solver_H.solve(self.b_H_vec, self.H_np1.x.petsc_vec)
        self.H_np1.x.scatter_forward()
        
        # 3. Rotate state
        self.E_n.x.array[:] = self.E_np1.x.array
        self.H_n.x.array[:] = self.H_np1.x.array

    def solve(self, T_final, dt, output_interval=10, verbose=True):
        self.setup_variational_forms(dt)
        num_steps = int(T_final / dt)
        
        self.E_history = [] 
        self.times = []
        
        print(f"Solving: {num_steps} steps")
        
        for i in range(num_steps):
            t = (i+1)*dt
            self.time_step(t)
            
            if i % output_interval == 0:
                self.times.append(t)
                self.E_history.append(self.E_n.x.array.copy())
                if verbose:
                    print(f"Step {i}/{num_steps}")
                    
        return self.E_n, self.H_n