#include "Poisson1D.hpp"

void
Poisson1D::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    /**
     * @brief Creates a 1D mesh of the segment [0,1] divided into N_el uniform intervals.
     *
     * Each active cell corresponds to an element, so `mesh.n_active_cells()`
     * will return `N_el`. The last parameter (true) assigns different boundary
     * IDs to the extremities:
     * - Left boundary: `ID 0`
     * - Right boundary: `ID 1`
     * This is useful for applying boundary conditions on both extremities.
     */
    GridGenerator::subdivided_hyper_cube(mesh, N_el, 0.0, 1.0, true);
    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;

    /*
     * @brief Writes the mesh to a `VTK` file for visualization.
     *
     * Since the mesh is generated internally, it is written to a file for user
     * inspection. This is not necessary if the mesh is read from file. Open the
     * file in ParaView to visualize the `1D` partition.
     */
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

    /*
     * @brief Note on finite element classes in `1D`.
     *
     * Finite elements in one dimension are obtained with the `FE_Q` or
     * `FE_SimplexP` classes. The former is meant for hexahedral elements, the
     * latter for tetrahedra, but they are equivalent in `1D`. We use
     * `FE_SimplexP` here for consistency with the next labs.
     */

    /**
     * @brief Creates Lagrange finite elements of degree r on 1D segments.
     *
     * `FE_SimplexP<dim>(r)` represents Lagrange finite elements of degree `r`.
     * For `r=1`, these are linear elements with `2` degrees of freedom per cell
     * (at the extremities). In `1D`, `dofs_per_cell` is `r + 1`.
     */
    fe = std::make_unique<FE_SimplexP<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

    /**
     * @brief Creates a Gauss quadrature rule on the simplex with `r + 1` points per cell.
     *
     * This quadrature integrates polynomials up to degree `2r + 1` exactly in
     * `1D`, ensuring sufficient accuracy for the products in the system,
     * particularly the mass matrix terms. For the stiffness term, it is also
     * adequate.
     */
    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    // Initialize the DoF handler with the mesh we constructed.
    dof_handler.reinit(mesh);

    /**
     * @brief Distributes the degrees of freedom (DoFs) across nodes and cells.
     *
     * For a given finite element space, initializes information on the control
     * variables: how many they are, where they are collocated, their global
     * indices, etc. In `1D` with linear finite elements, there is one DoF per
     * internal node and two DoFs at the boundaries (which will be constrained
     * by boundary conditions).
     */
    dof_handler.distribute_dofs(*fe);

    // `dof_handler.n_dofs()` indicates the number of unknowns in the system.
    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    /*
     * Initializes the sparsity pattern for the system matrix.
     *
     * A sparsity pattern is a data structure that indicates which entries of
     * the matrix are zero and which are non-zero. First, a
     * DynamicSparsityPattern is created (memory-inefficient and
     * access-inefficient but fast to write), then it is converted to a
     * SparsityPattern (more efficient but immutable -> hence cannot be
     * modified).
     */
    std::cout << "  Initializing the sparsity pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Then, we use the sparsity pattern to initialize the system matrix.
    std::cout << "  Initializing the system matrix" << std::endl;
    // Will be the stiffness matrix `K`.
    system_matrix.reinit(sparsity_pattern);

    // Finally, we initialize the right-hand side and solution vectors.
    std::cout << "  Initializing the system right-hand side" << std::endl;
    // The vector of known terms.
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    // Will contain the vector of unknowns `U`.
    solution.reinit(dof_handler.n_dofs());
  }
}


// How it builds `K` and `F`
void
Poisson1D::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element -> in `1D`, with `r = 1`, it is `2`.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  /**
   * @brief Creates an FEValues instance.
   *
   * This object allows computing basis functions, their derivatives, the
   * reference-to-current element mapping, and its derivatives on all quadrature
   * points of all elements. The update flags specify what quantities to compute
   * on quadrature points:
   * - `update_values`: values of shape functions.
   * - `update_gradients`: derivatives of shape functions.
   * - `update_quadrature_points`: positions of quadrature points.
   * - `update_JxW_values`: quadrature weights.
   */
  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Each cell has its own local matrix and its own local vector, then
  // accumulated into the global matrix. Local matrix and right-hand side
  // vector. These will be overwritten for each element within the loop.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // Vector to store the global indices of the DoFs of the current element
  // within the loop.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  /**
   * @brief Loop over all active cells to assemble the global linear system.
   *
   * For each cell, computes the local contributions to the stiffness matrix and
   * right-hand side vector, then adds them to the global matrix and vector.
   */
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // Reinitialize the FEValues object on the current element. This
      // precomputes all the quantities we requested when constructing FEValues
      // (see the `update_* flags above`) for all quadrature nodes of the
      // current cell.
      fe_values.reinit(cell);

      // Reset the cell matrix and vector, discarding any leftovers from the
      // previous element.
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      /**
       * @brief Assembles the local matrix and vector for the current cell.
       *
       * Loops over quadrature points to compute the local contributions to the
       * stiffness matrix and right-hand side vector.
       */
      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we assemble the local contribution for current cell and
          // current quadrature point, filling the local matrix and vector.
          const double mu_loc = mu(fe_values.quadrature_point(q));
          const double f_loc  = f(fe_values.quadrature_point(q));

          // Here we iterate over *local* DoF indices.
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += mu_loc *                     //
                                       fe_values.shape_grad(i, q) * //
                                       fe_values.shape_grad(j, q) * //
                                       fe_values.JxW(q);
                }

              cell_rhs(i) += f_loc *                       //
                             fe_values.shape_value(i, q) * //
                             fe_values.JxW(q);
            }
        }

      // At this point the local matrix and vector are constructed: we need
      // to sum them into the global matrix and vector. To this end, we need
      // to retrieve the global indices of the DoFs of current cell.
      cell->get_dof_indices(dof_indices);

      // Then, we add the local matrix and vector into the corresponding
      // positions of the global matrix and vector.
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }


  /**
   * @brief Applies boundary conditions to the linear system.
   *
   * So far we assembled the matrix as if there were no Dirichlet conditions.
   * Now we want to replace the rows associated to nodes on which Dirichlet
   * conditions are applied with equations like `u_i = b_i`.
   */
  {
    // We construct a map that stores, for each DoF corresponding to a Dirichlet
    // condition, the corresponding value. E.g., if the Dirichlet condition is
    // `u_i = b_i`, the map will contain the pair `(i, b_i)`.
    std::map<types::global_dof_index, double> boundary_values;

    // This object represents our boundary data as a real-valued function (that
    // always evaluates to zero). Other functions may require to implement a
    // custom class derived from `dealii::Function<dim>`.
    Functions::ZeroFunction<dim> bc_function;

    // Then, we build a map that, for each boundary tag, stores a pointer to the
    // corresponding boundary function.
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &bc_function;
    boundary_functions[1] = &bc_function;

    // `interpolate_boundary_values` fills the boundary_values map.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    // Finally, we modify the linear system to apply the boundary conditions.
    // This replaces the equations for the boundary DoFs with the corresponding
    // `u_i = 0` equations.
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

/**
 * @brief Solves the linear system using the conjugate gradient method.
 *
 * The system matrix is symmetric and positive definite, so `CG` is appropriate.
 * Uses the identity preconditioner and specified tolerances.
 */
void
Poisson1D::solve()
{
  std::cout << "===============================================" << std::endl;

  // Specify the maximum number of iterations, absolute tolerance, and relative
  // reduction.
  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  // Create the conjugate gradient solver.
  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;
  // Solve using the identity preconditioner.
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

/**
 * @brief Outputs the solution to a `VTK` file for visualization.
 *
 * Uses the `DataOut` class to write the solution vector to a file that can be
 * opened in ParaView.
 */
void
Poisson1D::output() const
{
  std::cout << "===============================================" << std::endl;

  // The `DataOut` class manages writing the results to a file.
  DataOut<dim> data_out;

  // Add the solution vector to the `DataOut` object.
  data_out.add_data_vector(dof_handler, solution, "solution");

  // Finalize the `DataOut` object by building patches.
  data_out.build_patches();

  // Write the data to a `VTK` file.
  const std::string output_file_name =
    "output-" + std::to_string(N_el) + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}

/**
 * @brief Computes the error in the specified norm.
 *
 * The error is an integral, approximated using a quadrature formula with r + 2
 * points per cell to ensure sufficient accuracy.
 */
double
Poisson1D::compute_error(const VectorTools::NormType &norm_type,
                         const Function<dim>         &exact_solution) const
{
  // Use a quadrature formula with one node more than in assembly for accuracy.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  // Compute the norm on each element and store in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Compute the global error by summing over all cells.
  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}