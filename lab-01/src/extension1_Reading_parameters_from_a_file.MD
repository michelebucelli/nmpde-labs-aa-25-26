In professional simulation software (like Ansys, OpenFOAM, or academic research codes), you **never** hard-code parameters like mesh size or material coefficients. You always read them from a configuration file. This allows you to run hundreds of simulations on a cluster without recompiling the code a single time.

---

### Step 1: Create the Parameter File

First, we need a text file that acts as the "menu" for your program.
Create a new file named **`parameters.prm`** in your main directory (where the `.cpp` files are).

**Content of `parameters.prm`:**

```text
# Parameter file for Poisson 1D

# The polynomial degree of the finite element (1 for linear, 2 for quadratic)
set Polynomial degree = 1

# The number of elements in the mesh
set Mesh elements     = 20
```

*Note: The lines starting with `#` are comments.*

-----

### Step 2: Modify the C++ Code

We need to modify `exercise-01.cpp` to stop using `const unsigned int N_el = 20;` and start asking the `ParameterHandler` for the values.

**Changes required in `exercise-01.cpp`:**

1.  **Include the header:** You need the library that handles parameters.
2.  **Declare parameters:** Tell deal.II what parameters to look for and what kind of data they are (Integer? Double? String?).
3.  **Parse the file:** Read `parameters.prm`.
4.  **Retrieve values:** Assign them to your C++ variables.

Here is the updated `main` function. Replace the content of your `exercise-01.cpp` with this:

```cpp
#include <iostream>
#include <deal.II/base/parameter_handler.h> // <--- NEW INCLUDE

#include "Poisson1D.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  const unsigned int dim = Poisson1D::dim;

  // -----------------------------------------------------------------------
  // NEW: PARAMETER HANDLING SECTION
  // -----------------------------------------------------------------------
  
  // 1. Create the ParameterHandler object
  ParameterHandler prm;

  // 2. Declare the entries
  // We tell the handler: "Expect a parameter named 'Mesh elements'".
  // - Default value: "10"
  // - Pattern: It must be an Integer (Patterns::Integer)
  // - Documentation: A help string describing it.
  prm.declare_entry("Mesh elements", "10", Patterns::Integer(),
                    "Number of elements in the mesh partition (0,1)");

  prm.declare_entry("Polynomial degree", "1", Patterns::Integer(),
                    "The finite element polynomial degree");

  // 3. Read the file
  // We assume the file is named 'parameters.prm' and is in the execution dir.
  prm.parse_input("parameters.prm");

  // 4. Retrieve the values
  // The handler reads them as strings/ints, we cast them to what we need.
  const unsigned int N_el = prm.get_integer("Mesh elements");
  const unsigned int r    = prm.get_integer("Polynomial degree");

  // Optional: Print to console to verify
  std::cout << "Reading parameters from file:" << std::endl
            << " - N_el: " << N_el << std::endl
            << " - r:    " << r << std::endl;

  // -----------------------------------------------------------------------
  // END PARAMETER HANDLING
  // -----------------------------------------------------------------------

  const auto mu = [](const Point<dim> & /*p*/) { return 1.0; };
  
  // Using the 'Point 5' forcing term from Exercise 2 for consistency, 
  // or you can keep the original Exercise 1 forcing term.
  const auto f = [](const Point<dim> &p) {
    if (p[0] <= 0.125 || p[0] > 0.25)
      return 0.0;
    else
      return -1.0;
  };

  // Initialize the problem using the variables N_el and r we just read
  Poisson1D problem(N_el, r, mu, f);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}
```

-----

### Step 3: Compile and Test

This is the important part. We will compile the code **once**, but run it **twice** with different results, without touching the C++ code in between.

1.  **Recompile:**

    ```bash
    make exercise-01
    ```

2.  **Run with default parameters (N=20, r=1):**
    Ensure your `parameters.prm` has `Mesh elements = 20`.

    ```bash
    ./exercise-01
    ```

    *Check the terminal output. It should say "Number of elements = 20".*

3.  **Modify the parameter file (The "Magic" Step):**
    Open `parameters.prm` in a text editor (like `nano` or `vim` or VS Code) and change:

    ```text
    set Mesh elements = 50
    set Polynomial degree = 2
    ```

    **Save the file.**

4.  **Run AGAIN (Do NOT recompile):**

    ```bash
    ./exercise-01
    ```

    *Check the terminal output. It should now say "Number of elements = 50" and "Degree = 2".*

### Why this is "Granular Detail" important

You used `prm.declare_entry`. This adds a layer of safety.
If you try to write `set Mesh elements = banana` in your text file, deal.II will throw an error saying "I expected an Integer but got 'banana'". This prevents your simulation from crashing silently with bad data.