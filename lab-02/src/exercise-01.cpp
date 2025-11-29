/**
 * @file exercise-01.cpp
 * @brief Minimal driver for the Poisson2D example.
 *
 * This program sets up and solves a model 2D Poisson problem using the
 * Poisson2D class. It demonstrates a typical workflow:
 *  - construct the problem with mesh resolution and polynomial degree,
 *  - provide coefficient and forcing functions as lambdas,
 *  - run setup, assembly, solve, and write the solution to disk.
 *
 * The file is intentionally minimal: all configuration is hard-coded to
 * keep the example concise for instructional purposes.
 */

#include <iostream>

#include "Poisson2D.hpp"

/**
 * @brief Program entry point.
 *
 * The main routine:
 *  - defines mesh resolution (N_el) and FE degree (r),
 *  - provides mu(p) and f(p) as lightweight lambdas,
 *  - constructs a Poisson2D instance and executes the standard sequence
 *    of methods: setup(), assemble(), solve(), output().
 *
 * @return int exit status (0 on success).
 */
int
main(int /*argc*/, char * /*argv*/[])
{
  // Number of elements per side of the unit square mesh.
  const unsigned int N_el = 5;

  // Polynomial degree of the simplex finite element space.
  const unsigned int r = 1;

  // Diffusion coefficient mu(x,y) = 1.0.
  // Parameter name intentionally omitted (/*p*/) to avoid unused-parameter
  // warnings.
  auto mu = [](const Point<2> & /*p*/) { return 1.0; };

  // Right-hand side f(x,y) = -5.0.
  // Parameter name intentionally omitted (/*p*/) to avoid unused-parameter
  // warnings.
  auto f = [](const Point<2> & /*p*/) { return -5.0; };

  // Construct the Poisson problem and run the solver pipeline.
  Poisson2D problem(N_el, r, mu, f);

  problem.setup();    // mesh, FE, DoFs, system containers
  problem.assemble(); // build matrix and RHS, apply boundary conditions
  problem.solve();    // solve the linear system
  problem.output();   // write solution to VTK

  return 0;
}