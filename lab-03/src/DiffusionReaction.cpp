#include "DiffusionReaction.hpp"

void
DiffusionReaction::setup()
{
  std::cout << "===============================================" << std::endl;

  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;

    // Read the mesh from file.
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh);

    std::ifstream mesh_file(mesh_file_name);
    grid_in.read_msh(mesh_file);

    std::cout << "  Number of elements = " << mesh.n_active_cells()
              << std::endl;
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

void
DiffusionReaction::assemble()
{
  std::cout << "===============================================" << std::endl;

  std::cout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu_loc    = mu(fe_values.quadrature_point(q));
          const double sigma_loc = sigma(fe_values.quadrature_point(q));
          const double f_loc     = f(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += mu_loc *                     //
                                       fe_values.shape_grad(i, q) * //
                                       fe_values.shape_grad(j, q) * //
                                       fe_values.JxW(q);

                  cell_matrix(i, j) += sigma_loc *                   //
                                       fe_values.shape_value(i, q) * //
                                       fe_values.shape_value(j, q) * //
                                       fe_values.JxW(q);
                }

              cell_rhs(i) += f_loc *                       //
                             fe_values.shape_value(i, q) * //
                             fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    Functions::ZeroFunction<dim>              bc_function;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (unsigned int i = 0; i < 6; ++i)
      boundary_functions[i] = &bc_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void
DiffusionReaction::solve()
{
  std::cout << "===============================================" << std::endl;

  // PreconditionIdentity preconditioner;

  // PreconditionJacobi preconditioner;
  // preconditioner.initialize(system_matrix);

  // PreconditionSOR preconditioner;
  // preconditioner.initialize(
  //   system_matrix,
  //   PreconditionSOR<SparseMatrix<double>>::AdditionalData(1.0));

  PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverCG<Vector<double>> solver(solver_control);

  std::cout << "  Solving the linear system" << std::endl;

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  std::cout << "  " << solver_control.last_step() << " CG iterations"
            << std::endl;
}

void
DiffusionReaction::output() const
{
  std::cout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");
  data_out.build_patches();

  // Use std::filesystem to construct the output file name based on the
  // mesh file name.
  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string           output_file_name =
    "output-" + mesh_path.stem().string() + ".vtk";
  std::ofstream output_file(output_file_name);
  data_out.write_vtk(output_file);

  std::cout << "Output written to " << output_file_name << std::endl;

  std::cout << "===============================================" << std::endl;
}

double
DiffusionReaction::compute_error(const VectorTools::NormType &norm_type,
                                 const Function<dim> &exact_solution) const
{
  const QGaussSimplex<dim> quadrature_error(r + 2);

  FE_SimplexP<dim> fe_linear(1);
  MappingFE        mapping(fe_linear);

  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
    VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}