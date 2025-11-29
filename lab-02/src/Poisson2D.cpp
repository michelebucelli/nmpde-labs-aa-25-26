/**
 * @file Poisson2D.cpp
 * @brief Implementation of Poisson2D methods: mesh/setup, assembly, solve, output,
 *        and error computation for a 2D Poisson problem using simplex elements.
 *
 * The implementation uses deal.II constructs (FE_SimplexP, QGaussSimplex,
 * FEValues, VectorTools, MatrixTools) to build and solve the finite element
 * linear system.
 *
 * Documentation follows a compact Doxygen style: each public method is
 * documented with a short description, parameters, and return behavior.
 */

#include "Poisson2D.hpp"

/**
 * @brief Prepare the problem: create mesh, finite element, quadrature,
 *        DoF handler and linear system containers.
 *
 * This method performs all one-time initialization steps required before
 * assembling the linear system:
 *  - create a subdivided hypercube mesh on [0,1]^2 with N_el elements per side,
 *  - instantiate the FE_SimplexP finite element of degree r,
 *  - create a QGaussSimplex quadrature rule (degree r+1),
 *  - initialize the DoFHandler and distribute DoFs,
 *  - initialize sparsity pattern, system matrix, RHS and solution vectors.
 *
 * Side effects:
 *  - writes a VTK mesh file named "mesh-<N_el>.vtk".
 */
void
Poisson2D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    // 1. Create a temporary mesh of squares (Quads)
    Triangulation<dim> quad_mesh;
    GridGenerator::subdivided_hyper_cube(quad_mesh, N_el, 0.0, 1.0, true);

    // 2. Convert the squares to triangles (Simplex) and store in the main
    // 'mesh'
    GridGenerator::convert_hypercube_to_simplex_mesh(quad_mesh, mesh);

    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    // Write the mesh to file.
    const std::string mesh_file_name = "mesh-" + std::to_string(N_el) + ".vtk";
    GridOut           grid_out;
    std::ofstream     grid_out_file(mesh_file_name);
    grid_out.write_vtk(mesh, grid_out_file);
    std::cout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    std::cout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity_pattern);

    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

/**
 * @brief Assemble the global linear system for the Poisson problem.
 *
 * The assembly computes element contributions for:
 *  - volume integrals: stiffness matrix using mu * grad(phi_i) . grad(phi_j)
 *    and load vector using f * phi_i,
 *  - Neumann boundary integrals on boundary IDs 2 and 3 adding contributions
 *    from the boundary function h(x,y) = y,
 *  - Dirichlet boundary conditions on boundary IDs 0 and 1 with g(x,y) = x+y.
 *
 * After assembly, Dirichlet conditions are enforced via
 * MatrixTools::apply_boundary_values.
 *
 * No return value; the global system_matrix and system_rhs are modified in
 * place.
 */
void
Poisson2D::assemble()
{
  std::cout << "===============================================" << std::endl;
  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  // Volume FEValues
  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // We define the quadrature for the face (dim-1)
  const QGauss<dim - 1> face_quadrature_formula(r + 1);

  // We initialize the Face FEValues using that quadrature
  FEFaceValues<dim> fe_face_values(*fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // 1. VOLUME INTEGRALS (Stiffness matrix and RHS f)
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu_loc = mu(fe_values.quadrature_point(q));
          const double f_loc  = f(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += mu_loc * fe_values.shape_grad(i, q) *
                                       fe_values.shape_grad(j, q) *
                                       fe_values.JxW(q);
                }
              cell_rhs(i) +=
                f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      // 2. NEUMANN BOUNDARY INTEGRALS
      for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell;
           ++face_n)
        {
          if (cell->face(face_n)->at_boundary())
            {
              const types::boundary_id b_id = cell->face(face_n)->boundary_id();

              // Neumann on Gamma_2 (bottom, ID 2) and Gamma_3 (top, ID 3)
              if (b_id == 2 || b_id == 3)
                {
                  fe_face_values.reinit(cell, face_n);

                  for (unsigned int q = 0; q < face_quadrature_formula.size();
                       ++q)
                    {
                      // h(x,y) = y.
                      const double h_loc =
                        fe_face_values.quadrature_point(q)[1];

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs(i) += h_loc *
                                         fe_face_values.shape_value(i, q) *
                                         fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      // Add local contributions to global system
      cell->get_dof_indices(dof_indices);
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Dirichlet boundary conditions (Gamma_0 left, Gamma_1 right).
  {
    std::map<types::global_dof_index, double> boundary_values;

    // g(x,y) = x + y
    class DirichletFunction : public Function<dim>
    {
    public:
      double
      value(const Point<dim> &p, const unsigned int = 0) const override
      {
        return p[0] + p[1];
      }
    };

    DirichletFunction g_function;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &g_function;
    boundary_functions[1] = &g_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

/**
 * @brief Solve the assembled linear system using Conjugate Gradient.
 *
 * The solver uses ReductionControl to limit iterations and tolerance,
 * and PreconditionIdentity (no preconditioning). The computed solution
 * is stored in the member vector 'solution'.
 *
 * Side effects:
 *  - prints the number of CG iterations used.
 */
void
Poisson2D::solve()
{
  std::cout << "===============================================" << std::endl;

  ReductionControl         solver_control(1000, 1.0e-16, 1.0e-6);
  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

/**
 * @brief Write the computed finite element solution to a VTK file.
 *
 * The file name follows the pattern "output-<N_el>.vtk".
 * The method adds the solution vector to a DataOut and writes it on disk.
 */
void
Poisson2D::output() const
{
  std::cout << "===============================================" << std::endl;

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  const std::string output_file_name =
    "output-" + std::to_string(N_el) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;
  std::cout << "===============================================" << std::endl;
}

/**
 * @brief Compute a global error norm between the finite element solution
 *        and a provided exact solution.
 *
 * @param[in] norm_type The norm type (L2, H1, Linfty) defined in
 *                      VectorTools::NormType.
 * @param[in] exact_solution A Function object representing the exact solution.
 * @return The computed global error as a double.
 *
 * The routine uses a QGaussSimplex quadrature of degree r+2 for error
 * integration and calls VectorTools::integrate_difference and
 * VectorTools::compute_global_error.
 */
double
Poisson2D::compute_error(const VectorTools::NormType &norm_type,
                         const Function<dim>         &exact_solution) const
{
  const QGaussSimplex<dim> quadrature_error(r + 2);
  Vector<double>           error_per_cell(mesh.n_active_cells());

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  return VectorTools::compute_global_error(mesh, error_per_cell, norm_type);
}