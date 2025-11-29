#ifndef POISSON_1D_HPP
#define POISSON_1D_HPP

#include <deal.II/base/quadrature_lib.h>
/**
 * @brief deal.II timing utilities.
 *
 * This header provides lightweight timing and profiling helpers used to
 * measure the execution time of code regions. The most commonly used classes
 * are:
 *  - dealii::Timer: low-level timer with start()/stop() and value() methods
 *    (useful for short, manually controlled measurements).
 *  - dealii::TimerOutput: higher-level utility that collects timing data from
 *    multiple named sections and prints a readable summary (supports nested
 *    regions and RAII helper types such as ScopedTimer).
 *
 * Practical guidance:
 *  - Use TimerOutput when you want aggregated, formatted timing reports for
 *    setup/assembly/solve phases. Instantiate a single TimerOutput (typically
 *    in the top-level object) and pass/borrow it where needed.
 *  - Prefer RAII helpers (ScopedTimer) to ensure timers are stopped even on
 *    early returns or exceptions.
 *  - TimerOutput is thread-aware when constructed with appropriate flags; check
 *    the deal.II documentation if using in parallel contexts.
 *  - Example:
 *      TimerOutput timer_output(std::cout, TimerOutput::summary,
 * TimerOutput::wall);
 *      {
 *        TimerOutput::ScopedTimer t(timer_output, "assembly");
 *        // code to time
 *      }
 *
 * See the deal.II reference for exact API and constructor options.
 */
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson1D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;

  // Constructor.
  Poisson1D(const unsigned int                              &N_el_,
            const unsigned int                              &r_,
            const std::function<double(const Point<dim> &)> &mu_,
            const std::function<double(const Point<dim> &)> &f_)
    : N_el(N_el_)
    , r(r_)
    , mu(mu_)
    , f(f_)
    // `std::cout` print results to the terminal.
    // `TimerOutput::summary` print a table at the end. 
    // `TimerOutput::wall_times` measure real-word time (wall clock). 
    , computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

protected:
  // Number of elements.
  const unsigned int N_el;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  std::function<double(const Point<dim> &)> mu;

  // Forcing term.
  std::function<double(const Point<dim> &)> f;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  //
  // We use a unique_ptr here so that we can choose the type and degree of the
  // finite elements at runtime (the degree is a constructor parameter).
  //
  // The class FiniteElement<dim> is an abstract class from which all types of
  // finite elements implemented by deal.ii inherit. Using the abstract class
  // makes it very easy to switch between different types of FE space among the
  // many that deal.ii provides.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  //
  // We use a unique_ptr here so that we can choose the type and order of the
  // quadrature formula at runtime (the order is a constructor parameter).
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;

  /**
   * @brief TimerOutput a timer used for profiling and aggregating timing information.
   *
   * TimerOutput collects timing data for named code regions (for example
   * "setup", "assembly", "solve") and prints a formatted summary. It supports
   * wall-clock and CPU timing and is intended to be instantiated once per
   * top-level object so timings are aggregated across calls.
   *
   * Usage (RAII):
   *   {
   *     TimerOutput::ScopedTimer t(computing_timer, "assembly");
   *     // code to time
   *   }
   *
   * Best practices:
   *  - Keep a single TimerOutput per top-level object and reuse it across
   *    member functions to produce a coherent report.
   *  - Declare it mutable if you need to time operations inside const member
   *    functions (as done here).
   *  - Construct the TimerOutput in the implementation (cpp) with an output
   *    stream and preferred reporting options, e.g.:
   *      computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall);
   *  - Prefer RAII ScopedTimer to ensure timers are stopped even on exceptions.
   *
   * See the deal.II reference for constructor options and advanced usage.
   */
  mutable TimerOutput computing_timer;
};

#endif