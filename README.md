# Ideal Magnetohydrodynamics & Grad-Shafranov Solver

A finite-volume ideal magnetohydrodynamics (MHD) solver and a Grad-Shafranov
(GS) equilibrium solver, developed to accompany the MPhil dissertation
**"Magnetohydrodynamic Simulations in Complex Geometries"**.

**Author** &emsp;&emsp; Uzair Abdullah\
**Institution** &nbsp; Centre for Scientific Computing\
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; Department of Physics\
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; University of Cambridge\
**Contact** &emsp;&nbsp;&nbsp;ua247@cam.ac.uk\
&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;ujaira02@gmail.com\
**Supervisor** &nbsp;Dr Maria Nikodemou\
**Date** &emsp;&emsp;&emsp; 07 August 2026

## Acknowledgements

This solver builds on decades of published work in computational
magnetohydrodynamics and finite-volume methods for hyperbolic conservation
laws, and on the CRATOS-GS Grad-Shafranov solver developed at the
Laboratory for Scientific Computing. Thanks go to the wider open-source
scientific computing community, and to OpenMP for the shared-memory
parallelism used throughout the solver.

Dr Maria Nikodemou\
Laboratory for Scientific Computing\
Department of Physics\
University of Cambridge

---

## Overview

The MHD solver integrates the time-dependent ideal magnetohydrodynamics
equations on a range of one, two, and three-dimensional test problems,
starting from simple flat-walled domains and building up to realistic
tokamak vessel geometries. The GS solver computes axisymmetric plasma
equilibria for the same reactor vessels, sharing their boundary
representation with the MHD reactor tests.

Both solvers are reached from a single interactive menu, or can be run
directly from the command line for scripting and batch use.

## Project Structure

```
├── main.cpp
├── mhd/
│   ├── 1d/                    1D Riemann problems
│   └── 2d/
│   │   ├── blast/             MHD blast wave
│   │   ├── briowu/            Brio-Wu shock tube
│   │   ├── kelvinhelmholtz/   Kelvin-Helmholtz instability
│   │   ├── orszagtang/        Orszag-Tang vortex
│   │   ├── reverseb/          Reverse-B shock tube
│   │   └── rotor/             MHD rotor
│   └── 3d/                    Reactor-vessel rotor and blast wave
├── gs/                        Grad-Shafranov equilibrium solver
├── sdf/                       Precomputed vessel signed distance functions
│   ├── ITER/
│   ├── MAST/
│   ├── ST40/
│   └── sdf.py                 Script to compute vessel signed distance functions
├── README.md
├── LICENCE-GPL3               Licence for the source code
└── LICENCE-CC-BY              Licence for the dissertation text and figures
```

Each test lives in its own translation unit named after the test, for
example `Rotor2DRef.cpp`, and the menu structure in `main.cpp` mirrors this
layout directly. A test's folder tells you where to find it in the menu.

## Building

The solver requires a C++17 compiler, OpenMP, and the Eigen linear algebra
library. Install the dependencies as below:

#### macOS (Homebrew)

```bash
brew install libomp eigen python
```

#### Linux (GCC)

```bash
sudo apt update
sudo apt install g++ libeigen3-dev python3
```

The project is then built from the repository root with:

#### macOS

```
g++ -O3 -std=c++17 -Xpreprocessor -fopenmp \
    -I$(brew --prefix libomp)/include \
    -L$(brew --prefix libomp)/lib \
    -I/opt/homebrew/opt/eigen/include/eigen3 \
    -Igs \
    -Imhd/1d \
    -Imhd/2d/blast -Imhd/2d/briowu -Imhd/2d/kelvinhelmholtz \
    -Imhd/2d/orszagtang -Imhd/2d/reverseb -Imhd/2d/rotor \
    -Imhd/3d \
    -lomp -march=native \
    main.cpp $(find gs mhd -name '*.cpp') \
    -o solver
```

#### Linux

```
g++ -O3 -std=c++17 \
    -fopenmp \
    -I/usr/include/eigen3 \
    -Igs \
    -Imhd/1d \
    -Imhd/2d/blast -Imhd/2d/briowu -Imhd/2d/kelvinhelmholtz \
    -Imhd/2d/orszagtang -Imhd/2d/reverseb -Imhd/2d/rotor \
    -Imhd/3d \
    -march=native \
    main.cpp $(find gs mhd -name '*.cpp') \
    -o solver
```

This produces a single binary, `solver`, in the repository root.

## Running

**Interactive menu**

```
./solver
```

This opens the top-level menu, from which the MHD Solver and the GS Solver
can be reached. Every menu offers a `?` option for a written explanation of
the tests available there, and a `0` option to go back.

**Jumping straight into a solver**

```
./solver --solver mhd
./solver --solver gs
```

These skip the top-level menu entirely and drop straight into the chosen
solver's own menu. The program exits once you back out of that menu.

**Running a single test directly**

```
./solver <test_id>
```

for example:

```
./solver rotor2dreact
```

Here is a full list of the experiments.

```
briowu1d
```
```
rarefac1d
```
```
reverseb1d
```
```
shocktrans1d
```
```
briowu2drot
```
```
briowu2dref
```
```
reverseb2dref
```
```
orszag2d
```
```
rotor2d
```
```
rotor2dref
```
```
rotor2dcirc
```
```
rotor2dreact
```
```
rotor3dreact
```
```
blast2d
```
```
blast2dref
```
```
blast2dcirc
```
```
blast2dreact
```
```
blast3dreact
```
```
kh2d
```
```
validationgs
```
```
equilibriumgs
```
```
perturbedgs
```

The available test IDs are also listed in the Read Me option inside the
program, and correspond to the 1D, 2D, and 3D test menus.

## Numerical Methods

The MHD solver combines a shared numerical core across every test:

- **Riemann solver.** The Harten-Lax-van Leer-Discontinuities (HLLD)
  solver of Miyoshi and Kusano resolves contact discontinuities and
  intermediate wave states more accurately than simpler HLL solvers.
- **Spatial reconstruction.** Third-order Weighted Essentially
  Non-Oscillatory (WENO3) reconstruction gives high-order accuracy while
  suppressing spurious oscillations near discontinuities.
- **Time integration.** Third-order Strong Stability-Preserving
  Runge-Kutta (SSP-RK3) stepping preserves the stability properties of the
  spatial discretisation under a Courant-Friedrichs-Lewy (CFL) number of
  0.6.
- **Divergence cleaning.** The Generalised Lagrange Multiplier (GLM)
  method of Dedner et al. maintains the divergence-free constraint

$$
\nabla \cdot \mathbf{B} = 0
$$

  &emsp;&nbsp;&nbsp;&nbsp; by introducing an auxiliary scalar field that carries spurious magnetic
  monopoles away from the domain.
- **Boundary treatment.** The Ghost Fluid Method (GFM) of Fedkiw et al.
  handles rigid boundaries by mirroring boundary-normal velocity and
  magnetic field components while preserving tangential components.

### Boundary Geometries

Every reflective wall, whether square, circular, or reactor-shaped, is
represented implicitly by a signed distance function (SDF) $\phi(\mathbf{x})$, which
satisfies the Eikonal equation

$$
\left|\nabla \phi(\mathbf{x})\right| = 1
$$

almost everywhere, so that $\left|\phi(\mathbf{x})\right|$ is the exact Euclidean distance from $\mathbf{x}$
to the nearest boundary point. By convention $\phi$ is negative inside the
fluid domain, positive in the solid region, and exactly zero on the wall.
The square and circular walls admit closed-form expressions for $\phi$.
Realistic tokamak cross-sections do not, so each vessel's SDF is instead
precomputed offline from digitised wall data and stored as a lookup table,
bilinearly interpolated at run time. Since ITER, MAST, and ST40 are all
axisymmetric, the three-dimensional SDF for the 3D tests is obtained by
revolving the same tabulated poloidal cross-section about the device axis
rather than tabulating a full volume.

To generate a signed distance function for the reactor vessels available
in this program (MAST, ITER, and ST40) run the following scripts:

#### macOS

```
python sdf/sdf.py
```

#### Linux

```
python3 sdf/sdf.py
```

from there, follow the instructions printed to the terminal.

## The Grad-Shafranov Solver

Assuming negligible flow, the ideal MHD momentum equation reduces to the
static force balance

$$
\nabla p = \mathbf{J} \times \mathbf{B}
$$

Imposing axisymmetry and writing the field in terms of the poloidal flux
$\Psi\left(R, Z\right)$ and toroidal field function $f\left(R, Z\right)$ yields the Grad-Shafranov
equation

$$
\Delta^{*}\Psi=-\mu_0R^2\frac{dp}{d\Psi}-f\frac{df}{d\Psi}
$$

a nonlinear elliptic PDE in which $\Psi$ appears both as the dependent variable
and, through the plasma profiles $p\left(\Psi\right)$ and $f\left(\Psi\right)$, as an implicit argument of
its own right-hand side. The solver resolves this nonlinearity by Picard
iteration, and is validated against the analytic Solov'ev and
Cerfon-Freidberg equilibria before being applied to the ITER, MAST, and
ST40 vessels.

## Licensing

The source code in this repository is released under the GPL-3.0 licence, set
out in full in [`LICENsE`](LICENSE).

The dissertation text, figures, and other written material accompanying
this repository are released under a Creative Commons Attribution 4.0
International licence, set out in full in
[`LICENCE-CC-BY`](LICENCE-CC-BY).

## References

- Brio, M. and Wu, C. C. (1988). An upwind differencing scheme for the
  equations of ideal magnetohydrodynamics. *J. Comput. Phys.*
- Ryu, D. and Jones, T. W. (1995). Numerical magnetohydrodynamics in
  astrophysics: algorithm and tests for one-dimensional flow.
  *Astrophys. J.*, 442:228-258. doi:10.1086/175437
- Miyoshi, T. and Kusano, K. (2005). A multi-state HLL approximate Riemann
  solver for ideal magnetohydrodynamics. *J. Comput. Phys.*
- Jiang, G.-S. and Shu, C.-W. (1996). Efficient implementation of weighted
  ENO schemes. *J. Comput. Phys.*
- Gottlieb, S., Shu, C.-W. and Tadmor, E. (2001). Strong
  stability-preserving high-order time discretization methods.
  *SIAM Review.*
- Dedner, A. et al. (2002). Hyperbolic divergence cleaning for the MHD
  equations. *J. Comput. Phys.*
- Fedkiw, R. P. et al. (1999). A non-oscillatory Eulerian approach to
  interfaces in multimaterial flows (the ghost fluid method).
  *J. Comput. Phys.*
- Dumbser, M. et al. (2008). A unified framework for the construction of
  one-step finite volume and discontinuous Galerkin schemes.
  *J. Comput. Phys.*
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid
  Dynamics.* Springer.
- Solov'ev, L. S. (1968). The theory of hydromagnetic stability of
  toroidal plasma configurations. *Soviet Physics JETP*, 26:400-407.
- Cerfon, A. J. and Freidberg, J. P. (2010). "One size fits all" analytic
  solutions to the Grad-Shafranov equation. *Phys. Plasmas*, 17(3):032502.
  doi:10.1063/1.3328818
- Dudson, B. (2019). FreeGS: Free Boundary Grad-Shafranov Solver.
  https://github.com/freegs-plasma/freegs
- Farmakalides, A., Nikiforakis, N., Millmore, S. et al. (2025).
  CRATOS-GS: a free-boundary, hierarchical adaptive mesh refinement
  Grad-Shafranov solver. *AIP Advances*, 15.
- Freidberg, J. P. (1987). *Ideal Magnetohydrodynamics.* Cambridge
  University Press.

## Contact

_To submit questions or queries, or to receive the model analysis code, please feel free to reach out directly._
