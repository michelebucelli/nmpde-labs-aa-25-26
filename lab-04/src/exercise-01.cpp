#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto mu = [](const Point<dim> & /*p*/) { return 0.1; };
  const auto f  = [](const Point<dim>  &/*p*/, const double  &/*t*/) {
    return 0.0;
  };

  Heat problem(/*mesh_filename = */ "../mesh/mesh-cube-10.msh",
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 0.0,
               /* delta_t = */ 0.0025,
               mu,
               f);

  problem.run();

  return 0;
}