// ============================================================================
// Rarefac1D.h — MHD Riemann Problem 3 (rarefaction + out-of-plane rotation)
// ----------------------------------------------------------------------------
// Initial conditions (Table 3.1, RP3):
//   L: ρ=1.7  u=0  v=0  w=0         Bx=1.1  By=1.0           Bz=0.0          p=1.7
//   R: ρ=0.2  u=0  v=0  w=-1.496891 Bx=1.1  By=0.78588731621 Bz=0.61836983763 p=0.2
//   x0=0.5,  t_out=0.15,  γ=5/3
//
// Solver: HLLD + WENO3 + SSP-RK3 (all physics self-contained in .cpp)
// Output: 1D/Rarefac/data/cut.dat
//         1D/Rarefac/plots/png/cut.png   (if gnuplot is available)
// ============================================================================
#pragma once

void run_rarefac();
