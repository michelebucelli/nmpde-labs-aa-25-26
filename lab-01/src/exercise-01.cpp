#include <iostream>

/**
 * @brief deal.II ParameterHandler utilities.
 *
 * The header <deal.II/base/parameter_handler.h> provides the
 * deal::ParameterHandler class used to declare, document, parse and validate
 * named runtime parameters (typically read from a .prm file). Typical
 * responsibilities when including this header:
 *  - declare parameters with declare_entry(name, default, pattern,
 * description),
 *  - read/parse parameter files via parse_input_file(filename) or
 * parse_input(),
 *  - query parameter values with get(name) and convert them to the required
 * type (use utility functions or std::stringstream for conversion),
 *  - validate values using regex-style patterns or custom checks.
 *
 * Practical notes:
 *  - Use clear parameter names and descriptions so auto-generated .prm files
 * are self-documenting.
 *  - Prefer strict patterns when declaring entries to catch configuration
 * errors early (e.g., integer, double, boolean patterns).
 *  - After parsing, retrieve values and convert to the appropriate C++ types
 *    before using them in numerical setup.
 *
 * See the deal.II documentation for full API details of deal::ParameterHandler.
 */
#include <deal.II/base/parameter_handler.h>

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

  // -----------------------------------------------------------------------
  // NEW: PARAMETER HANDLING SECTION
  // -----------------------------------------------------------------------

  // 1. Create the ParameterHandler object.
  ParameterHandler prm;

  // 2. Declare the entries.
  // We tell the handler: "Expect a parameter named `Mesh elements`".
  // - `Default value`: "10"
  // - `Pattern`: It must be an `Integer` (`Patterns::Integer`)
  // - `Documentation`: A help string describing it.
  prm.declare_entry("Mesh elements",
                    "10",
                    Patterns::Integer(),
                    "Number of elements in the mesh partition (0,1)");

  prm.declare_entry("Polynomial degree",
                    "1",
                    Patterns::Integer(),
                    "The finite element polynomial degree");

  // 3. Read the file.
  // We assume the file is named `parameters.prm` and is in the execution dir.
  prm.parse_input("parameters.prm");
  
  // 4. Retrieve the values
  // The handler reads them as strings/ints, we cast them to what we need.
  const unsigned int N_el = prm.get_integer("Mesh elements");
  const unsigned int r    = prm.get_integer("Polynomial degree");

  // Optional but useful: Print to the console to verify
  std::cout << "Reading parameters from file:" << std::endl
            << " - N_el: " << N_el << std::endl
            << " - r:    " << r << std::endl;
  // std::cout << "Reading parameters form file: " << '\n'
  //           << " - N_el: " << N_el << '\n'
  //           << " - r: " << r << '\n';

  // -----------------------------------------------------------------------
  // END PARAMETER HANDLING
  // -----------------------------------------------------------------------

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
   *
   * @note the problem is now initialized using the variables `N_el` and `r` we just read.
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
