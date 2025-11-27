#include "DiffusionReaction.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = DiffusionReaction::dim;

  const std::string  mesh_filename = "../mesh/mesh-cube-40.msh";
  const unsigned int degree        = 1;

  const auto mu = [](const Point<dim> &p) {
    if (p[0] < 0.5)
      return 100.0;
    else
      return 1.0;
  };

  const auto f     = [](const Point<dim>     &/*p*/) { return 1.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };

  DiffusionReaction problem(mesh_filename, degree, mu, sigma, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}