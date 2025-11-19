#include <iostream>

#include "Poisson1D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  /**
   * @brief Defines the spatial dimension for the problem.
   *
   * The dimension is retrieved from `Poisson1D.hpp`, which is set to `1` for
   * `1D` problems. This constant is used to specify the template parameter for
   * `Point<dim>`.
   */
  constexpr unsigned int dim = Poisson1D::dim;

  /**
   * @brief Number of elements in the mesh partition `(0,1)`.
   *
   * This value determines the discretization level for the finite element
   * method.
   * @note changed from 40 -> 20 since the text is asking for `N_el = 20`.

   */
  const unsigned int N_el = 20;

  /**
   * @brief The finite element polynomial degree.
   *
   * For `r=1`, the elements are linear `(P_1)`.
   */
  const unsigned int r = 1;

  /**
   * @brief Lambda function implementing the diffusion coefficient mu(x).
   *
   * The function returns a constant value of 1.0, ignoring the input point p.
   */
  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };

  /**
   * @brief Lambda function implementing the source term f(x).
   *
   * The function defines f(x) as:
   * - `0` if `x <= 1/8 or x > 1/4`,
   * - `-1` if `1/8 < x <= 1/4`.
   * This is the point-wise function specified in the text.
   */
  const auto f = [](const Point<dim> &p) {
    if (p[0] <= 0.125 || p[0] > 0.25)
      return 0.0;
    else
      return -1.0;
  };

  /**
   * @brief Builds the object that represents the `FEM` problem with:
   * - mesh divided into `N_el` elements,
   * - FE of grade `r`,
   * coefficients `mu` and `f` given by the lambda function.
   */
  Poisson1D problem(N_el, r, mu, f);

  // Sets up the finite element problem: builds mesh, FE space, DoF handler,
  // matrices, and vectors.
  problem.setup();
  // Assembles the linear system `KU = F` and applies boundary conditions: `u(0)
  // = u(1) = 0`.
  problem.assemble();
  // Writes the solution to a .vtk file for visualization (using Paraview in
  // this course).
  problem.solve();
  // Solves the linear system numerically to find the unknowns U.
  problem.output();

  return 0;
}
