#include <deal.II/base/convergence_table.h>

#include <iostream>

#include "DiffusionReaction.hpp"

static constexpr unsigned int dim = DiffusionReaction::dim;

// Exact solution.
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return std::sin(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;

    result[0] =
      2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) * std::sin(4.0 * M_PI * p[1]);
    result[1] =
      4.0 * M_PI * std::sin(2.0 * M_PI * p[0]) * std::cos(4.0 * M_PI * p[1]);

    return result;
  }

  static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
};

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  ConvergenceTable table;

  const std::vector<unsigned int> N_el_values = {5, 10, 20, 40};
  const unsigned int              r           = 2;

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };

  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };

  const auto f = [](const Point<dim> &p) {
    return (20.0 * M_PI * M_PI + 1.0) * std::sin(2.0 * M_PI * p[0]) *
           std::sin(4.0 * M_PI * p[1]);
  };

  const ExactSolution exact_solution;

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (const auto &N_el : N_el_values)
    {
      const std::string mesh_file_name =
        "../mesh/mesh-square-" + std::to_string(N_el) + ".msh";

      DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

      problem.setup();
      problem.assemble();
      problem.solve();
      problem.output();

      const double h = 1.0 / N_el;

      const double error_L2 =
        problem.compute_error(VectorTools::L2_norm, exact_solution);
      const double error_H1 =
        problem.compute_error(VectorTools::H1_norm, exact_solution);

      table.add_value("h", h);
      table.add_value("L2", error_L2);
      table.add_value("H1", error_H1);

      convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  table.set_scientific("L2", true);
  table.set_scientific("H1", true);

  table.write_text(std::cout);

  return 0;
}