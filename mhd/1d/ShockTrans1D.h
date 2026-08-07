// ============================================================================
// ShockTrans1D.h — MHD Riemann Problem 2 (trans-Alfvénic shock tube)
// ----------------------------------------------------------------------------
// Initial conditions (Table 3.1, RP2):
//   L: ρ=1.08    u=1.2      v=0.01    w=0.5
//      Bx=0.56418958354  By=1.01554125039  Bz=0.56418958354  p=0.95
//   R: ρ=0.9891  u=-0.0131  v=0.0269  w=0.010037
//      Bx=0.56418958354  By=1.13526228001  Bz=0.56492303     p=0.97159
//   x0=0.5,  t_out=0.2,  γ=5/3
//
// Solver: HLLD + WENO3 + SSP-RK3 (all physics self-contained in .cpp)
// Output: 1D/ShockTrans/data/cut.dat
//         1D/ShockTrans/plots/png/cut.png   (if gnuplot is available)
// ============================================================================
#pragma once

void run_shock_trans();
