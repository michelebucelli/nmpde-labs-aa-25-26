#include <iostream>

#include "Poisson2D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = Poisson2D::dim;

  const std::string  mesh_file_name = "../mesh/mesh-square-20.msh";
  const unsigned int r              = 1;

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto f  = [](const Point<dim>  &/*p*/) { return -5.0; };
  const auto h  = [](const Point<dim> &p) { return p[1]; };

  Poisson2D problem(mesh_file_name, r, mu, f, h);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}