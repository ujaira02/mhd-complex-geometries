// ============================================================================
// ReverseB1D.h — MHD Riemann Problem 4 (By-reversal, asymmetric domain)
// ----------------------------------------------------------------------------
// Initial conditions (Table 3.1, RP4):
//   L: ρ=1.0  u=0  v=0  w=0   Bx=1.3  By=+1.0  Bz=0.0  p=1.0
//   R: ρ=0.4  u=0  v=0  w=0   Bx=1.3  By=-1.0  Bz=0.0  p=0.4
//   x0=0.4  (asymmetric — see Table 3.1 caption),  t_out=0.16,  γ=5/3
//
// Solver: HLLD + WENO3 + SSP-RK3 (all physics self-contained in .cpp)
// Output: 1D/ReverseB/data/cut.dat
//         1D/ReverseB/plots/png/cut.png   (if gnuplot is available)
// ============================================================================
#pragma once

void run_reverse_b();
