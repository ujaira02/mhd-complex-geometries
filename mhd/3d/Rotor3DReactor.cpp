// ============================================================================
// Rotor3DReactor.cpp — MHD rotor (full 3D) with revolved-SDF reflective BC
// ----------------------------------------------------------------------------
//   - HLLD fluxes in x, y, z
//   - WENO3 reconstruction
//   - SSP-RK3 time stepping
//   - Optional Dedner-style GLM hyperbolic divergence cleaning
//   - Reflective boundary defined by an external SDF .dat file
// Initial conditions:
//   (0.0, 0.0, 0.0), r0 = 0.1, r1 = 0.115, with r = √(x² + y² + z²)
//      Core (r ≤ 0.1):           ρ = 10,      (u,v,w) = (-10y, 10x, 0)
//      Taper (0.1 < r ≤ 0.115):  ρ = 1+9f_r,  (u,v,w) = (-10y·f_r, 10x·f_r, 0)
//      Ambient (r > 0.115):      ρ = 1,       (u,v,w) = (0, 0, 0)
//   where f_r = (23 - 200r)/3, B = (1/√(4π))(0, 2.5, 0)
// ============================================================================
#include "Rotor3DReactor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
#include <chrono>
#include <thread>

// --------------------------------- helpers ----------------------------------
static inline int syscmd(const std::string& cmd) {
    return std::system(cmd.c_str());
}

static inline void mkdir_p(const std::string& path) {
    syscmd("mkdir -p \"" + path + "\"");
}

static inline bool cmd_exists(const std::string& name) {
    return syscmd("command -v " + name + " >/dev/null 2>&1") == 0;
}

static inline double clamp_pos(double x, double eps = 1e-12) {
    return (x < eps) ? eps : x;
}

static inline bool is_finite(double x) {
    return std::isfinite(x);
}

// -------------------------------- parameters --------------------------------
static constexpr double gamma_gas = 5.0 / 3.0;
static constexpr double CFL = 0.60;

static constexpr double x_min = -1.0, x_max = 1.0;
static constexpr double y_min = -1.0, y_max = 1.0;
static constexpr double z_min = -1.0, z_max = 1.0;

static constexpr double SDF_SCALE = 1.05;
static double g_major_radius = 0.5;
static double g_minor_scale = 1.0;

static constexpr double r0_rotor = 0.10;
static constexpr double r1_rotor = 0.115;
static constexpr double rho_core = 10.0;
static constexpr double rho_ambient = 1.0;
static constexpr double p_rotor = 1.0;
static double rotor_center_x = 0.40; 
static double rotor_center_y = 0.00;
static constexpr double rotor_center_z = 0.00;

static const double B_mag = 1.0 / std::sqrt(4.0 * M_PI);
static const double Bx0 = 0.0;
static const double By0 = B_mag * 2.5;
static const double Bz0 = 0.0;

struct SdfTableRotor3D {
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0, dx = 0, dy = 0;
    int nx = 0, ny = 0;
    std::vector<double> sdf;
    bool loaded = false;
    double x_mid = 0.0;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[rotor] Cannot open: " << path << "\n";
            return false;
        }
        std::vector<double> xs, ys, vs;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            double x, y, v;
            if (!(ss >> x >> y >> v)) continue;
            xs.push_back(x); ys.push_back(y); vs.push_back(v);
        }
        if (xs.empty()) { std::cerr << "[rotor] No data in: " << path << "\n"; return false; }

        xmin = *std::min_element(xs.begin(), xs.end());
        xmax = *std::max_element(xs.begin(), xs.end());
        ymin = *std::min_element(ys.begin(), ys.end());
        ymax = *std::max_element(ys.begin(), ys.end());

        nx = 0;
        const double y0r = ys[0];
        for (double yv : ys) {
            if (std::abs(yv - y0r) < 1e-12) {
                ++nx; 
            } else {
                break;
            }
        }

        if (nx < 2) {
            std::cerr << "[rotor] Cannot determine nx.\n";
            return false;
        }

        ny = static_cast<int>(xs.size()) / nx;
        if (nx * ny != static_cast<int>(xs.size())) {
            std::cerr << "[rotor] Size mismatch.\n"; return false;
        }

        dx = (xmax - xmin) / (nx - 1);
        dy = (ymax - ymin) / (ny - 1);
        sdf = vs;
        loaded = true;

        std::cerr << "[rotor] Loaded " << nx << "x" << ny << " from " << path << "\n";

        const double cy = 0.5 * (ymin + ymax);
        const double hy = 0.5 * (ymax - ymin);
        const double scale = SDF_SCALE / std::max(xmax, hy > 0.0 ? hy : xmax);
        xmin *= scale;
        xmax *= scale;
        ymin = (ymin - cy) * scale;
        ymax = (ymax - cy) * scale;
        dx *= scale;
        dy *= scale;
        for (double& v : sdf) {
            v *= scale;
        }
        std::cerr << "[rotor] Scale factor: " << scale << "\n";

        if (xmin < -1e-10)
            std::cerr << "[rotor] WARNING: xmin=" << xmin << " < 0. "
                      << "The SDF x-axis should be the cylindrical radius "
                      << "(always >= 0). Check the .dat file format.\n";

        std::cerr << "[rotor] Approximate major radius R ~ "
                  << 0.5 * (xmin + xmax) << "  (mid-radial point of table)\n";
        x_mid = 0.5 * (xmin + xmax);
        return true;
    }

    double interp(double x, double y) const {
        if (!loaded) {
            return 1.0;
        }

        if (x < xmin - 0.5 * dx || x > xmax + 0.5 * dx ||
            y < ymin - 0.5 * dy || y > ymax + 0.5 * dy) {
            return 1.0;
        }

        double fi = (x - xmin) / dx, fj = (y - ymin) / dy;
        int i0 = std::max(0, std::min((int)std::floor(fi), nx - 2));
        int j0 = std::max(0, std::min((int)std::floor(fj), ny - 2));
        const double tx = fi - i0;
        const double ty = fj - j0;

        return (1 - tx) * (1 - ty) * sdf[j0 * nx + i0] 
             + tx * (1 - ty) * sdf[j0 * nx + (i0 + 1)]
             + (1 - tx) * ty * sdf[(j0 + 1) * nx + i0] 
             + tx * ty * sdf[(j0 + 1) * nx + (i0 + 1)];
    }

    void gradient(double r, double y, double& gr, double& gy) const {
        gr = (interp(r + dx, y) - interp(r - dx, y)) / (2.0 * dx);
        gy = (interp(r, y + dy) - interp(r, y - dy)) / (2.0 * dy);
        const double mag = std::sqrt(gr * gr + gy * gy);
        
        if (mag > 1e-12) {
            gr /= mag;
            gy /= mag;
        } else {
            gr = 1.0;
            gy = 0.0;
        }
    }
};

static SdfTableRotor3D g_sdf;

static double g_revolve_dirn[2] = {0.0, 1.0};
static double g_revolve_point[2] = {0.0, 0.0};
static int g_revolve_segments = 0;
static std::string g_run_label;

static inline void revolve_project(double x, double y, double z,
                                    double& xyCoord0, double& xyCoord1) {
    const double* dirn = g_revolve_dirn;
    const double* point = g_revolve_point;

    const double nSq = dirn[0] * dirn[0] + dirn[1] * dirn[1];

    const double w = (dirn[0] * (x - point[0]) + dirn[1] * (y - point[1])) / nSq;

    const double q0 = point[0] + w * dirn[0];
    const double q1 = point[1] + w * dirn[1];

    double dist = std::sqrt((x - q0) * (x - q0) + (y - q1) * (y - q1) + z * z);

    const double sqrtNSq = std::sqrt(nSq);
    const double perp0 = dirn[1] / sqrtNSq;
    const double perp1 = -dirn[0] / sqrtNSq;

    if (g_revolve_segments > 0) {
        const double radial_dot = (x - q0) * perp0 + (y - q1) * perp1;
        const double theta = std::atan2(z, radial_dot) + M_PI;
        const double segSize = 2.0 * M_PI / g_revolve_segments;
        const double seg = std::fmod(theta, segSize);
        const double omega = 0.5 * (M_PI - segSize);
        dist *= std::sin(M_PI - seg - omega) / std::sin(omega);
    }

    const double r_cs = (dist - g_major_radius) / g_minor_scale + g_sdf.x_mid;
    const double y_axial = w * std::sqrt(nSq);
    const double y_cs = y_axial / g_minor_scale;

    xyCoord0 = r_cs;
    xyCoord1 = y_cs;
}

static inline double phi_sdf(double x, double y, double z) {
    double c0, c1;
    revolve_project(x, y, z, c0, c1);
    return g_sdf.interp(c0, c1);
}

static inline void sdf_normal(double x, double y, double z,
                                  double& nx_out, double& ny_out, double& nz_out)
{
    const double h = std::min(g_sdf.dx, g_sdf.dy) * 0.5;
    nx_out = (phi_sdf(x + h, y, z) - phi_sdf(x - h, y, z)) / (2.0 * h);
    ny_out = (phi_sdf(x, y + h, z) - phi_sdf(x, y - h, z)) / (2.0 * h);
    nz_out = (phi_sdf(x, y, z + h) - phi_sdf(x, y, z - h)) / (2.0 * h);
    const double mag = std::sqrt(nx_out * nx_out + ny_out * ny_out + nz_out * nz_out);

    if (mag > 1e-12) {
        nx_out /= mag;
        ny_out /= mag;
        nz_out /= mag;
    } else {
        nx_out = 1.0;
        ny_out = 0.0;
        nz_out = 0.0;
    }
}

// ------------------------------ GLM-MHD state -------------------------------
struct Prim {
    double rho, vx, vy, vz, p, Bx, By, Bz, psi;
};

struct Cons {
    double rho, mx, my, mz, Bx, By, Bz, E, psi; 
};

// --------------------------------- algebra ----------------------------------
static inline Cons c_add(const Cons& a, const Cons& b) {
    return {
        a.rho + b.rho, a.mx + b.mx, a.my + b.my, a.mz + b.mz,
        a.Bx + b.Bx, a.By + b.By, a.Bz + b.Bz, a.E + b.E, a.psi + b.psi
    };
}
static inline Cons c_sub(const Cons& a, const Cons& b) {
    return {
        a.rho - b.rho, a.mx - b.mx, a.my - b.my, a.mz - b.mz,
        a.Bx - b.Bx, a.By - b.By, a.Bz - b.Bz, a.E - b.E, a.psi - b.psi
    };
}

static inline Cons c_mul(double s, const Cons& a) {
    return {
        s * a.rho, s * a.mx, s * a.my, s * a.mz,
        s * a.Bx, s * a.By, s * a.Bz, s * a.E, s * a.psi
    };
}

// --------------------------------- physics ----------------------------------
static inline Prim prim_with_floors(Prim W) {
    W.rho = std::max(W.rho, 1e-10);
    W.p = std::max(W.p, 1e-10);
    return W;
}

static inline double pressure_from_cons(const Cons& U) {
    const double rho = clamp_pos(U.rho);
    const double vx = U.mx / rho;
    const double vy = U.my / rho;
    const double vz = U.mz / rho;
    const double v2 = vx * vx + vy * vy + vz * vz;
    const double B2 = U.Bx * U.Bx + U.By * U.By + U.Bz * U.Bz;
    const double eint = U.E - 0.5 * rho * v2 - 0.5 * B2;
    return std::max((gamma_gas - 1.0) * eint, 1e-12);
}

static inline bool state_is_admissible(const Cons& U) {
    if (!is_finite(U.rho) || !is_finite(U.mx) || !is_finite(U.my) ||
        !is_finite(U.mz) || !is_finite(U.Bx) || !is_finite(U.By) ||
        !is_finite(U.Bz) || !is_finite(U.E) || !is_finite(U.psi)) {
        return false;
    }
    if (U.rho <= 1e-10) return false;
    const double p = pressure_from_cons(U);
    return is_finite(p) && p > 1e-10;
}

static inline Prim cons_to_prim(const Cons& U) {
    Prim W{};
    W.rho = clamp_pos(U.rho);
    W.vx = U.mx / W.rho;
    W.vy = U.my / W.rho;
    W.vz = U.mz / W.rho;
    W.Bx = U.Bx;
    W.By = U.By;
    W.Bz = U.Bz;
    W.psi = U.psi;
    W.p = pressure_from_cons(U);
    return W;
}

static inline Cons prim_to_cons(const Prim& W) {
    Cons U{};
    U.rho = clamp_pos(W.rho);
    U.mx = U.rho * W.vx;
    U.my = U.rho * W.vy;
    U.mz = U.rho * W.vz;
    U.Bx = W.Bx;
    U.By = W.By;
    U.Bz = W.Bz;
    U.psi = W.psi;

    const double v2 = W.vx * W.vx + W.vy * W.vy + W.vz * W.vz;
    const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
    const double eint = W.p / (gamma_gas - 1.0);
    U.E = eint + 0.5 * U.rho * v2 + 0.5 * B2;
    return U;
}

static inline double fast_speed_dir(const Prim& W, double Bn) {
    const double rho = clamp_pos(W.rho);
    const double a2 = gamma_gas * W.p / rho;
    const double B2 = (W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz) / rho;
    const double Bn2 = (Bn * Bn) / rho;
    const double term = a2 + B2;
    const double disc = std::max(0.0, term * term - 4.0 * a2 * Bn2);
    const double cf2 = 0.5 * (term + std::sqrt(disc));
    return std::sqrt(std::max(0.0, cf2));
}

// ------------------------------ GLM correction ------------------------------
static inline void glm_correct_x(Prim& WL, Prim& WR, double ch, bool use_glm) {
    if (!use_glm) return;
    const double chs = std::max(ch, 1e-14);
    const double Bx_hat = 0.5 * (WL.Bx + WR.Bx) - 0.5 * (WR.psi - WL.psi) / chs;
    const double psi_hat = 0.5 * (WL.psi + WR.psi) - 0.5 * chs * (WR.Bx - WL.Bx);
    WL.Bx = WR.Bx = Bx_hat;
    WL.psi = WR.psi = psi_hat;
}

static inline void glm_correct_y(Prim& WL, Prim& WR, double ch, bool use_glm) {
    if (!use_glm) return;
    const double chs = std::max(ch, 1e-14);
    const double By_hat = 0.5 * (WL.By + WR.By) - 0.5 * (WR.psi - WL.psi) / chs;
    const double psi_hat = 0.5 * (WL.psi + WR.psi) - 0.5 * chs * (WR.By - WL.By);
    WL.By = WR.By = By_hat;
    WL.psi = WR.psi = psi_hat;
}

static inline void glm_correct_z(Prim& WL, Prim& WR, double ch, bool use_glm) {
    if (!use_glm) return;
    const double chs = std::max(ch, 1e-14);
    const double Bz_hat = 0.5 * (WL.Bz + WR.Bz) - 0.5 * (WR.psi - WL.psi) / chs;
    const double psi_hat = 0.5 * (WL.psi + WR.psi) - 0.5 * chs * (WR.Bz - WL.Bz);
    WL.Bz = WR.Bz = Bz_hat;
    WL.psi = WR.psi = psi_hat;
}

static bool prompt_use_glm() {
    std::cout << "\n[rotor] Enable hyperbolic divergence cleaning (GLM)? [y/n, default y]: ";
    std::string answer;
    std::getline(std::cin, answer);
    if (answer.empty()) return true;
    const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(answer[0])));
    return c != 'n';
}

// -------------------------------- GLM fluxes --------------------------------
static inline Cons flux_x(const Cons& U, double ch, bool use_glm) {
    const Prim W = cons_to_prim(U);
    const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
    const double pt  = W.p + 0.5 * B2;
    const double vdotB = W.vx * W.Bx + W.vy * W.By + W.vz * W.Bz;
    
    Cons F{};
    F.rho = U.mx;
    F.mx = U.mx * W.vx + pt - W.Bx * W.Bx;
    F.my = U.my * W.vx - W.Bx * W.By;
    F.mz = U.mz * W.vx - W.Bx * W.Bz;

    if (use_glm) {
        F.Bx = W.psi;
        F.psi = ch * ch * W.Bx;
    } else {
        F.Bx = 0.0;
        F.psi = 0.0;
    }

    F.By = W.By * W.vx - W.Bx * W.vy;
    F.Bz = W.Bz * W.vx - W.Bx * W.vz;
    F.E = (U.E + pt) * W.vx - W.Bx * vdotB;
    return F;
}

static inline Cons flux_y(const Cons& U, double ch, bool use_glm) {
    const Prim W = cons_to_prim(U);
    const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
    const double pt = W.p + 0.5 * B2;
    const double vdotB = W.vx * W.Bx + W.vy * W.By + W.vz * W.Bz;

    Cons G{};
    G.rho = U.my;
    G.mx = U.mx * W.vy - W.By * W.Bx;
    G.my = U.my * W.vy + pt - W.By * W.By;
    G.mz = U.mz * W.vy - W.By * W.Bz;

    if (use_glm) {
        G.By = W.psi;
        G.psi = ch * ch * W.By;
    } else {
        G.By = 0.0;
        G.psi = 0.0;
    }

    G.Bz = W.Bz * W.vy - W.By * W.vz;
    G.E = (U.E + pt) * W.vy - W.By * vdotB;
    return G;
}

static inline Cons flux_z(const Cons& U, double ch, bool use_glm) {
    const Prim W = cons_to_prim(U);
    const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
    const double pt = W.p + 0.5 * B2;
    const double vdotB = W.vx * W.Bx + W.vy * W.By + W.vz * W.Bz;

    Cons H{};
    H.rho = U.mz;
    H.mx = U.mx * W.vz - W.Bz * W.Bx;
    H.my = U.my * W.vz - W.Bz * W.By;
    H.mz = U.mz * W.vz + pt - W.Bz * W.Bz;

    if (use_glm) {
        H.Bz = W.psi;
        H.psi = ch * ch * W.Bz;
    } else {
        H.Bz = 0.0;   
        H.psi = 0.0;
    }

    H.Bx = W.Bx * W.vz - W.Bz * W.vx;
    H.By = W.By * W.vz - W.Bz * W.vy;
    H.E = (U.E + pt) * W.vz - W.Bz * vdotB;
    return H;
}

// ----------------------------------- HLLD -----------------------------------
static inline Cons hll_x(const Cons& UL_in, const Cons& UR_in, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL_in));
    Prim WR = prim_with_floors(cons_to_prim(UR_in));
    glm_correct_x(WL, WR, ch, use_glm);
    const Cons UL = prim_to_cons(WL);
    const Cons UR = prim_to_cons(WR);
    const double cfL = fast_speed_dir(WL, WL.Bx);
    const double cfR = fast_speed_dir(WR, WR.Bx);
    const double SL = std::min(WL.vx - cfL, WR.vx - cfR);
    const double SR = std::max(WL.vx + cfL, WR.vx + cfR);
    const Cons FL = flux_x(UL,ch,use_glm);
    const Cons FR = flux_x(UR,ch,use_glm);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double denom = SR - SL;
    const Cons num = c_add(c_sub(c_mul(SR, FL), c_mul(SL, FR)), c_mul(SL * SR, c_sub(UR, UL)));
    return c_mul(1.0 / denom, num);
}

static inline Cons hll_y(const Cons& UL_in, const Cons& UR_in, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL_in));
    Prim WR = prim_with_floors(cons_to_prim(UR_in));
    glm_correct_y(WL, WR, ch, use_glm);
    const Cons UL = prim_to_cons(WL);
    const Cons UR = prim_to_cons(WR);
    const double cfL = fast_speed_dir(WL, WL.By);
    const double cfR = fast_speed_dir(WR, WR.By);
    const double SL = std::min(WL.vy - cfL, WR.vy - cfR);
    const double SR = std::max(WL.vy + cfL, WR.vy + cfR);
    const Cons GL = flux_y(UL, ch, use_glm);
    const Cons GR = flux_y(UR, ch, use_glm);

    if (SL >= 0.0) return GL;
    if (SR <= 0.0) return GR;

    const double denom = SR - SL;
    const Cons num = c_add(c_sub(c_mul(SR, GL), c_mul(SL, GR)), c_mul(SL * SR, c_sub(UR, UL)));
    return c_mul(1.0 / denom, num);
}

static inline Cons hll_z(const Cons& UL_in, const Cons& UR_in, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL_in));
    Prim WR = prim_with_floors(cons_to_prim(UR_in));
    glm_correct_z(WL, WR, ch, use_glm);
    const Cons UL = prim_to_cons(WL);
    const Cons UR = prim_to_cons(WR);
    const double cfL = fast_speed_dir(WL, WL.Bz);
    const double cfR = fast_speed_dir(WR, WR.Bz);
    const double SL = std::min(WL.vz - cfL, WR.vz - cfR);
    const double SR = std::max(WL.vz + cfL, WR.vz + cfR);
    const Cons HL = flux_z(UL, ch, use_glm);
    const Cons HR = flux_z(UR, ch, use_glm);

    if (SL >= 0.0) return HL;
    if (SR <= 0.0) return HR;

    const double denom = SR - SL;
    const Cons num = c_add(c_sub(c_mul(SR, HL), c_mul(SL, HR)), c_mul(SL * SR, c_sub(UR, UL)));
    return c_mul(1.0 / denom, num);
}

static inline Cons rotate_xy_cons(const Cons& U) {
    return {U.rho, U.my, U.mx, U.mz, U.By, U.Bx, U.Bz, U.E, U.psi};
}
static inline Cons rotate_xy_flux_back(const Cons& F) {
    return {F.rho, F.my, F.mx, F.mz, F.By, F.Bx, F.Bz, F.E, F.psi};
}
static inline Cons rotate_xz_cons(const Cons& U) {
    return {U.rho, U.mz, U.my, U.mx, U.Bz, U.By, U.Bx, U.E, U.psi};
}
static inline Cons rotate_xz_flux_back(const Cons& H) {
    return {H.rho, H.mz, H.my, H.mx, H.Bz, H.By, H.Bx, H.E, H.psi};
}

static inline Cons hlld_x(const Cons& UL_in, const Cons& UR_in, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL_in));
    Prim WR = prim_with_floors(cons_to_prim(UR_in));
    glm_correct_x(WL, WR, ch, use_glm);
    const Cons UL = prim_to_cons(WL);
    const Cons UR = prim_to_cons(WR);
    const Cons FL = flux_x(UL, ch, use_glm);
    const Cons FR = flux_x(UR, ch, use_glm);

    const double Bx_face = 0.5 * (WL.Bx + WR.Bx);
    const double psi_face = 0.5 * (WL.psi + WR.psi);
    const double ptL = WL.p + 0.5 * (WL.Bx * WL.Bx + WL.By * WL.By + WL.Bz * WL.Bz);
    const double ptR = WR.p + 0.5 * (WR.Bx * WR.Bx + WR.By * WR.By + WR.Bz * WR.Bz);
    const double cfL = fast_speed_dir(WL, Bx_face);
    const double cfR = fast_speed_dir(WR, Bx_face);

    const double SL = std::min(WL.vx - cfL, WR.vx - cfR);
    const double SR = std::max(WL.vx + cfL, WR.vx + cfR);
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double rhoL = clamp_pos(WL.rho);
    const double rhoR = clamp_pos(WR.rho);
    const double denomM = (SR - WR.vx) * rhoR - (SL - WL.vx) * rhoL;
    if (std::abs(denomM) < 1e-12) return hll_x(UL, UR, ch, use_glm);

    const double SM =
        ((SR - WR.vx) * UR.mx - (SL - WL.vx) * UL.mx + ptL - ptR) / denomM;
    const double pt_star =
        ((SR - WR.vx) * rhoR * ptL - (SL - WL.vx) * rhoL * ptR +
         rhoL * rhoR * (SR - WR.vx) * (SL - WL.vx) * (WR.vx - WL.vx)) / denomM;

    const auto star_state = [&](const Prim& W, const Cons& U, double S, double pt_side) -> std::pair<bool, Cons> {
        Cons Us{};
        const double rho = clamp_pos(W.rho);
        const double rho_star = rho * (S - W.vx) / (S - SM);
        const double D = rho * (S - W.vx) * (S - SM) - Bx_face * Bx_face;
        if (!is_finite(rho_star) || rho_star <= 0.0 || std::abs(D) < 1e-12) {
            return {false, {}};
        }

        const double factor = (rho * (S - W.vx) * (S - W.vx) - Bx_face * Bx_face) / D;
        const double vy_star = W.vy - (Bx_face * W.By * (SM - W.vx)) / D;
        const double vz_star = W.vz - (Bx_face * W.Bz * (SM - W.vx)) / D;
        const double By_star = W.By * factor;
        const double Bz_star = W.Bz * factor;

        const double vdotB = W.vx * Bx_face + W.vy * W.By + W.vz * W.Bz;
        const double vdotBs = SM * Bx_face + vy_star * By_star + vz_star * Bz_star;
        const double E_star =
            ((S - W.vx) * U.E - pt_side * W.vx + pt_star * SM +
             Bx_face * (vdotB - vdotBs)) / (S - SM);

        Us = {rho_star, rho_star * SM, rho_star * vy_star, rho_star * vz_star,
              Bx_face, By_star, Bz_star, E_star, psi_face};
        if (!state_is_admissible(Us)) {
            return {false, {}};
        }
        return {true, Us};
    };

    const auto [okL, UstL] = star_state(WL, UL, SL, ptL);
    const auto [okR, UstR] = star_state(WR, UR, SR, ptR);
    if (!(okL && okR)) return hll_x(UL, UR, ch, use_glm);

    const double rhoSL = clamp_pos(UstL.rho);
    const double rhoSR = clamp_pos(UstR.rho);
    const double sqL = std::sqrt(rhoSL);
    const double sqR = std::sqrt(rhoSR);
    const double denomA = sqL + sqR;
    if (denomA < 1e-12) return hll_x(UL, UR, ch, use_glm);

    const double signBx = (Bx_face >= 0.0) ? 1.0 : -1.0;
    const double SstL = SM - std::abs(Bx_face) / std::sqrt(rhoSL);
    const double SstR = SM + std::abs(Bx_face) / std::sqrt(rhoSR);

    const double vy_ss =
        (sqL * (UstL.my / rhoSL) + sqR * (UstR.my / rhoSR) +
         signBx * (UstR.By - UstL.By)) / denomA;
    const double vz_ss =
        (sqL * (UstL.mz / rhoSL) + sqR * (UstR.mz / rhoSR) +
         signBx * (UstR.Bz - UstL.Bz)) / denomA;
    const double By_ss =
        (sqL * UstL.By + sqR * UstR.By +
         signBx * std::sqrt(rhoSL * rhoSR) * ((UstR.my / rhoSR) - (UstL.my / rhoSL))) / denomA;
    const double Bz_ss =
        (sqL * UstL.Bz + sqR * UstR.Bz +
         signBx * std::sqrt(rhoSL * rhoSR) * ((UstR.mz / rhoSR) - (UstL.mz / rhoSL))) / denomA;

    Cons UssL = UstL, UssR = UstR;
    UssL.my = rhoSL * vy_ss;
    UssL.mz = rhoSL * vz_ss;
    UssL.By = By_ss;
    UssL.Bz = Bz_ss;
    UssR.my = rhoSR * vy_ss;
    UssR.mz = rhoSR * vz_ss;
    UssR.By = By_ss;
    UssR.Bz = Bz_ss;

    const double vtBtL = (UstL.my / rhoSL) * UstL.By + (UstL.mz / rhoSL) * UstL.Bz;
    const double vtBtR = (UstR.my / rhoSR) * UstR.By + (UstR.mz / rhoSR) * UstR.Bz;
    const double vtBtSS = vy_ss * By_ss + vz_ss * Bz_ss;
    UssL.E = UstL.E - sqL * (vtBtL - vtBtSS) * signBx;
    UssR.E = UstR.E + sqR * (vtBtR - vtBtSS) * signBx;
    if (!state_is_admissible(UssL) || !state_is_admissible(UssR)) {
        return hll_x(UL, UR, ch, use_glm);
    }

    const Cons FstL = c_add(FL, c_mul(SL, c_sub(UstL, UL)));
    const Cons FstR = c_add(FR, c_mul(SR, c_sub(UstR, UR)));
    const Cons FssL = c_add(FstL, c_mul(SstL, c_sub(UssL, UstL)));
    const Cons FssR = c_add(FstR, c_mul(SstR, c_sub(UssR, UstR)));

    Cons F = (SL <= 0.0 && 0.0 <= SstL) ? FstL :
             (SstL <= 0.0 && 0.0 <= SM) ? FssL :
             (SM <= 0.0 && 0.0 <= SstR) ? FssR : FstR;
    F.Bx = psi_face;
    F.psi = ch * ch * Bx_face;
    return F;
}

static inline Cons hlld_y(const Cons& UL, const Cons& UR, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL));
    Prim WR = prim_with_floors(cons_to_prim(UR));
    glm_correct_y(WL, WR, ch, use_glm);
    return rotate_xy_flux_back(
        hlld_x(rotate_xy_cons(prim_to_cons(WL)), rotate_xy_cons(prim_to_cons(WR)), ch, use_glm)
    );
}

static inline Cons hlld_z(const Cons& UL, const Cons& UR, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL));
    Prim WR = prim_with_floors(cons_to_prim(UR));
    glm_correct_z(WL, WR, ch, use_glm);
    return rotate_xz_flux_back(
        hlld_x(rotate_xz_cons(prim_to_cons(WL)), rotate_xz_cons(prim_to_cons(WR)), ch, use_glm)
    );
}

// ---------------------------------- WENO3 -----------------------------------
static inline double weno3_left_scalar(double qm1, double q0, double qp1) {
    const double eps = 1e-10;
    const double beta0 = (q0 - qm1) * (q0 - qm1);
    const double beta1 = (qp1 - q0) * (qp1 - q0);
    const double d0 = 1.0 / 3.0;
    const double d1 = 2.0 / 3.0;
    const double a0 = d0 / ((eps + beta0) * (eps + beta0));
    const double a1 = d1 / ((eps + beta1) * (eps + beta1));
    const double w0 = a0 / (a0 + a1);
    const double w1 = a1 / (a0 + a1);
    const double p0 = -0.5 * qm1 + 1.5 * q0;
    const double p1 = 0.5 * q0 + 0.5 * qp1;
    return w0 * p0 + w1 * p1;
}

static inline double weno3_right_scalar(double q0, double qp1, double qp2) {
    const double eps = 1e-10;
    const double beta0 = (qp1 - q0) * (qp1 - q0);
    const double beta1 = (qp2 - qp1) * (qp2 - qp1);
    const double d0 = 2.0 / 3.0;
    const double d1 = 1.0 / 3.0;
    const double a0 = d0 / ((eps + beta0) * (eps + beta0));
    const double a1 = d1 / ((eps + beta1) * (eps + beta1));
    const double w0 = a0 / (a0 + a1);
    const double w1 = a1 / (a0 + a1);
    const double p0 = 0.5 * q0 + 0.5 * qp1;
    const double p1 = 1.5 * qp1 - 0.5 * qp2;
    return w0 * p0 + w1 * p1;
}

static Prim reconstruct_left_weno3(const Prim& Wm1, const Prim& W0, const Prim& Wp1) {
    Prim W{};
    W.rho = weno3_left_scalar(Wm1.rho, W0.rho, Wp1.rho);
    W.vx = weno3_left_scalar(Wm1.vx, W0.vx, Wp1.vx);
    W.vy = weno3_left_scalar(Wm1.vy, W0.vy, Wp1.vy);
    W.vz = weno3_left_scalar(Wm1.vz, W0.vz, Wp1.vz);
    W.p = weno3_left_scalar(Wm1.p, W0.p, Wp1.p);
    W.Bx = weno3_left_scalar(Wm1.Bx, W0.Bx, Wp1.Bx);
    W.By = weno3_left_scalar(Wm1.By, W0.By, Wp1.By);
    W.Bz = weno3_left_scalar(Wm1.Bz, W0.Bz, Wp1.Bz);
    W.psi = W0.psi;
    return prim_with_floors(W);
}

static Prim reconstruct_right_weno3(const Prim& W0, const Prim& Wp1, const Prim& Wp2) {
    Prim W{};
    W.rho = weno3_right_scalar(W0.rho, Wp1.rho, Wp2.rho);
    W.vx = weno3_right_scalar(W0.vx, Wp1.vx, Wp2.vx);
    W.vy = weno3_right_scalar(W0.vy, Wp1.vy, Wp2.vy);
    W.vz = weno3_right_scalar(W0.vz, Wp1.vz, Wp2.vz);
    W.p = weno3_right_scalar(W0.p, Wp1.p, Wp2.p);
    W.Bx = weno3_right_scalar(W0.Bx, Wp1.Bx, Wp2.Bx);
    W.By = weno3_right_scalar(W0.By, Wp1.By, Wp2.By);
    W.Bz = weno3_right_scalar(W0.Bz, Wp1.Bz, Wp2.Bz);
    W.psi = Wp1.psi;
    return prim_with_floors(W);
}

// -------------------------------- grid + BC ---------------------------------
struct Grid {
    int Nx = 0, Ny = 0, Nz = 0, ng = 2;
    double x0 = 0, y0 = 0, z0 = 0;
    double Lx = 1, Ly = 1, Lz = 1;
    double dx = 0, dy = 0, dz = 0;
    std::vector<Cons> U;

    int nx() const { return Nx + 2 * ng; }
    int ny() const { return Ny + 2 * ng; }
    int nz() const { return Nz + 2 * ng; }
    inline int id(int i, int j, int k) const { return k * ny() * nx() + j * nx() + i; }
};

static Prim sample_prim_bilinear(const Grid& G, double x, double y, double z) {
    const double x_lo = G.x0 + G.dx * 0.501;
    const double x_hi = G.x0 + G.Lx - G.dx * 0.501;
    const double y_lo = G.y0 + G.dy * 0.501;
    const double y_hi = G.y0 + G.Ly - G.dy * 0.501;
    const double z_lo = G.z0 + G.dz * 0.501;
    const double z_hi = G.z0 + G.Lz - G.dz * 0.501;
    x = std::max(x_lo, std::min(x, x_hi));
    y = std::max(y_lo, std::min(y, y_hi));
    z = std::max(z_lo, std::min(z, z_hi));

    const double i_float = (x - G.x0) / G.dx + G.ng - 0.5;
    const double j_float = (y - G.y0) / G.dy + G.ng - 0.5;
    const double k_float = (z - G.z0) / G.dz + G.ng - 0.5;

    const int i0 = std::max(G.ng, std::min((int)std::floor(i_float), G.Nx + G.ng - 2));
    const int j0 = std::max(G.ng, std::min((int)std::floor(j_float), G.Ny + G.ng - 2));
    const int k0 = std::max(G.ng, std::min((int)std::floor(k_float), G.Nz + G.ng - 2));
    const int i1 = std::min(i0 + 1, G.Nx + G.ng - 1);
    const int j1 = std::min(j0 + 1, G.Ny + G.ng - 1);
    const int k1 = std::min(k0 + 1, G.Nz + G.ng - 1);
    const double tx = i_float - i0;
    const double ty = j_float - j0;
    const double tz = k_float - k0;

    const Prim P[2][2][2] = {
        {{cons_to_prim(G.U[G.id(i0, j0, k0)]), cons_to_prim(G.U[G.id(i0, j0, k1)])},
         {cons_to_prim(G.U[G.id(i0, j1, k0)]), cons_to_prim(G.U[G.id(i0, j1, k1)])}},
        {{cons_to_prim(G.U[G.id(i1, j0, k0)]), cons_to_prim(G.U[G.id(i1, j0, k1)])},
         {cons_to_prim(G.U[G.id(i1, j1, k0)]), cons_to_prim(G.U[G.id(i1, j1, k1)])}}
    };

    auto tri = [&](auto f) {
        double v = 0;
        for (int di = 0; di < 2; di++) {
            for (int dj = 0; dj < 2; dj++) {
                for (int dk = 0; dk < 2; dk++) {
                    const double w = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty) * (dk ? tz : 1 - tz);
                    v += w * f(P[di][dj][dk]);
                }
            }
        }
        return std::isfinite(v) ? v : 0.0;
    };

    Prim W{};
    W.rho = tri([](const Prim& p){ return p.rho;});
    W.vx = tri([](const Prim& p){ return p.vx; });
    W.vy = tri([](const Prim& p){ return p.vy; });
    W.vz = tri([](const Prim& p){ return p.vz;});
    W.p = tri([](const Prim& p){ return p.p; });
    W.Bx = tri([](const Prim& p){ return p.Bx; });
    W.By = tri([](const Prim& p){ return p.By; });
    W.Bz = tri([](const Prim& p){ return p.Bz; });
    W.psi = tri([](const Prim& p){ return p.psi;});
    return prim_with_floors(W);
}

static inline bool is_interior_cell(int i, int j, int k, const Grid& G) {
    const double x = G.x0 + (i - G.ng + 0.5) * G.dx;
    const double y = G.y0 + (j - G.ng + 0.5) * G.dy;
    const double z = G.z0 + (k - G.ng + 0.5) * G.dz;
    return phi_sdf(x,y,z) <= 0.0;
}

static void apply_reflective_bc(Grid& G) {
    const int nx = G.nx();
    const int ny = G.ny();
    const int nz = G.nz();
    const double h = std::max({G.dx, G.dy, G.dz});

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double x = G.x0 + (i - G.ng + 0.5) * G.dx;
                const double y = G.y0 + (j - G.ng + 0.5) * G.dy;
                const double z = G.z0 + (k - G.ng + 0.5) * G.dz;
                if (phi_sdf(x,y,z) <= 0.0) continue;

                const double phi = phi_sdf(x,y,z);
                double nxw;
                double nyw;
                double nzw;
                sdf_normal(x, y, z, nxw, nyw, nzw);

                const double xs_mirror = x - 2.0 * phi * nxw;
                const double ys_mirror = y - 2.0 * phi * nyw;
                const double zs_mirror = z - 2.0 * phi * nzw;

                Prim W1{};
                if (phi_sdf(xs_mirror, ys_mirror, zs_mirror) <= 0.0) {
                    W1 = sample_prim_bilinear(G, xs_mirror, ys_mirror, zs_mirror);
                } else {
                    const double xs = x - (phi + h) * nxw;
                    const double ys = y - (phi + h) * nyw;
                    const double zs = z - (phi + h) * nzw;
                    if (phi_sdf(xs, ys, zs) <= 0.0) {
                        W1 = sample_prim_bilinear(G, xs, ys, zs);
                    } else {
                        W1.rho = rho_ambient;
                        W1.p = p_rotor;
                        W1.vx = W1.vy = W1.vz = 0.0;
                        W1.Bx = Bx0;
                        W1.By = By0;
                        W1.Bz = Bz0;
                        W1.psi = 0.0;
                    }
                }

                const double Bn0 = Bx0 * nxw + By0 * nyw + Bz0 * nzw;
                const bool no_slip = std::abs(Bn0) > 1e-12;

                const double vn = W1.vx * nxw + W1.vy * nyw + W1.vz * nzw;

                double vx_g, vy_g, vz_g;
                if (no_slip) {
                    vx_g = -W1.vx;
                    vy_g = -W1.vy;
                    vz_g = -W1.vz;
                } else {
                    vx_g = W1.vx - 2.0 * vn * nxw;
                    vy_g = W1.vy - 2.0 * vn * nyw;
                    vz_g = W1.vz - 2.0 * vn * nzw;
                }

                const double Bn_lab = W1.Bx * nxw + W1.By * nyw + W1.Bz * nzw;
                const double Btx = W1.Bx - Bn_lab * nxw;
                const double Bty = W1.By - Bn_lab * nyw;
                const double Btz = W1.Bz - Bn_lab * nzw;

                const double Bn_ghost = -Bn_lab + 2.0 * Bn0;

                const double Bx_g = Bn_ghost * nxw + Btx;
                const double By_g = Bn_ghost * nyw + Bty;
                const double Bz_g = Bn_ghost * nzw + Btz;

                Prim Wg{};
                Wg.rho = W1.rho;
                Wg.p = W1.p;
                Wg.vx = vx_g;
                Wg.vy = vy_g;
                Wg.vz = vz_g;
                Wg.Bx = Bx_g;
                Wg.By = By_g;
                Wg.Bz = Bz_g;
                Wg.psi = -W1.psi;

                G.U[G.id(i, j, k)] = prim_to_cons(prim_with_floors(Wg));
            }
        }
    }
}

static void enforce_floors(Grid& G) {
    const int ng = G.ng, nx = G.nx(), ny = G.ny(), nz = G.nz();
    constexpr double hard_limit = 1e8;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            for (int i = ng; i < nx - ng; ++i) {
                if (!is_interior_cell(i, j, k, G)) continue;
                Prim W = cons_to_prim(G.U[G.id(i, j, k)]);
                if (!is_finite(W.rho) || !is_finite(W.p) || !is_finite(W.vx) ||
                    !is_finite(W.vy) || !is_finite(W.vz) || !is_finite(W.Bx) ||
                    !is_finite(W.By) || !is_finite(W.Bz) || !is_finite(W.psi) ||
                    std::abs(W.vx) > hard_limit || std::abs(W.vy) > hard_limit || 
                    std::abs(W.vz) > hard_limit || std::abs(W.Bx) > hard_limit ||
                    std::abs(W.By) > hard_limit || std::abs(W.Bz) > hard_limit ||
                    std::abs(W.psi) > hard_limit) {
                    W = {rho_ambient, 0, 0, 0, p_rotor, Bx0, By0, Bz0, 0.0};
                }
                G.U[G.id(i, j, k)] = prim_to_cons(prim_with_floors(W));
            }
        }
    }
}

static void damp_glm_psi(Grid& G, double dt, double ch, bool use_glm) {
    if (!use_glm || ch <= 0.0 || dt <= 0.0) return;
    const int ng = G.ng, nx = G.nx(), ny = G.ny(), nz = G.nz();
    const double h = std::min({G.dx, G.dy, G.dz});
    const double alpha = 0.18;
    const double rate = ch * alpha / std::max(h, 1e-14);
    const double factor = std::exp(-dt * rate);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            for (int i = ng; i < nx - ng; ++i) {
                if (!is_interior_cell(i, j, k, G)) continue;
                G.U[G.id(i, j, k)].psi *= factor;
            }
        }
    }
}

// -------------------------------- time step ---------------------------------
static double compute_ch_and_dt(const Grid& G, double& ch_out, bool use_glm) {
    double amax = 1e-14;
    const int ng = G.ng, nx = G.nx(), ny = G.ny(), nz = G.nz();
    #pragma omp parallel for collapse(2) reduction(max:amax) schedule(static)
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            for (int i = ng; i < nx - ng; ++i) {
                if (!is_interior_cell(i, j, k, G)) continue;
                const Prim W = cons_to_prim(G.U[G.id(i, j, k)]);
                if (!is_finite(W.rho) || !is_finite(W.p) ||
                    !is_finite(W.vx) || !is_finite(W.vy) ||
                    !is_finite(W.vz)) continue;
                const double ax = std::abs(W.vx) + fast_speed_dir(W, W.Bx);
                const double ay = std::abs(W.vy) + fast_speed_dir(W, W.By);
                const double az = std::abs(W.vz) + fast_speed_dir(W, W.Bz);
                if (is_finite(ax)) amax = std::max(amax,ax);
                if (is_finite(ay)) amax = std::max(amax,ay);
                if (is_finite(az)) amax = std::max(amax,az);
            }
        }
    }
    amax = std::max(amax, 1e-8);
    ch_out = use_glm ? amax : 0.0;
    const double inv_sum = 1.0 / G.dx + 1.0 / G.dy + 1.0 / G.dz;
    return std::max(CFL / (amax * inv_sum), 1e-8);
}

// ---------------------------- semi-discrete RHS -----------------------------
static std::vector<Cons> rhs(const Grid& G, double ch, bool use_glm) {
    const int ng = G.ng, nx = G.nx(), ny = G.ny(), nz = G.nz();
    const int total = nx * ny * nz;
    std::vector<Cons> LU(total, Cons{0,0,0,0,0,0,0,0,0});

    std::vector<Prim> W(total);
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; ++idx) {
        W[idx] = cons_to_prim(G.U[idx]);
    }

    std::vector<Cons> Fx((nx - 1) * ny * nz, Cons{0,0,0,0,0,0,0,0,0});
    auto fx_id = [&](int i, int j, int k){ return k * ny * (nx - 1) + j * (nx - 1) + i; };
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            for (int i = ng - 1; i <= nx - ng - 1; ++i) {
                const Prim WL = reconstruct_left_weno3(W[G.id(i - 1, j, k)], W[G.id(i, j, k)], W[G.id(i + 1, j, k)]);
                const Prim WR = reconstruct_right_weno3(W[G.id(i, j, k)], W[G.id(i + 1, j, k)], W[G.id(i + 2, j, k)]);
                const Cons UL = prim_to_cons(prim_with_floors(WL));
                const Cons UR = prim_to_cons(prim_with_floors(WR));
                Fx[fx_id(i, j, k)] = hlld_x(UL, UR, ch, use_glm);
            }
        }
    }

    std::vector<Cons> Gy(nx * (ny - 1) * nz, Cons{0,0,0,0,0,0,0,0,0});
    auto gy_id = [&](int i, int j, int k){ return k * (ny - 1) * nx + j * nx + i; };
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng - 1; j <= ny - ng - 1; ++j) {
            for (int i = ng; i < nx - ng; ++i) {
                const Prim WL = reconstruct_left_weno3(W[G.id(i, j - 1, k)], W[G.id(i, j, k)], W[G.id(i, j + 1, k)]);
                const Prim WR = reconstruct_right_weno3(W[G.id(i, j, k)], W[G.id(i, j + 1, k)], W[G.id(i, j + 2, k)]);
                const Cons UL = prim_to_cons(prim_with_floors(WL));
                const Cons UR = prim_to_cons(prim_with_floors(WR));
                Gy[gy_id(i, j, k)] = hlld_y(UL, UR, ch, use_glm);
            }
        }
    }
 
    std::vector<Cons> Hz(nx * ny * (nz - 1), Cons{0,0,0,0,0,0,0,0,0});
    auto hz_id = [&](int i, int j, int k){ return k * ny * nx + j * nx + i; };
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = ng - 1; k <= nz - ng - 1; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            for (int i = ng; i < nx - ng; ++i) {
                const Prim WL = reconstruct_left_weno3(W[G.id(i, j, k - 1)], W[G.id(i, j, k)], W[G.id(i, j, k + 1)]);
                const Prim WR = reconstruct_right_weno3(W[G.id(i, j, k)], W[G.id(i, j, k + 1)], W[G.id(i, j, k + 2)]);
                const Cons UL = prim_to_cons(prim_with_floors(WL));
                const Cons UR = prim_to_cons(prim_with_floors(WR));                
                Hz[hz_id(i, j, k)] = hlld_z(UL, UR, ch, use_glm);
            }
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            for (int i = ng; i < nx - ng; ++i) {
                if (!is_interior_cell(i, j, k, G)) continue;
                const Cons dF = c_sub(Fx[fx_id(i, j, k)], Fx[fx_id(i - 1, j, k)]);
                const Cons dG = c_sub(Gy[gy_id(i, j, k)], Gy[gy_id(i, j - 1, k)]);
                const Cons dH = c_sub(Hz[hz_id(i, j, k)], Hz[hz_id(i, j, k - 1)]);
                LU[G.id(i, j, k)] = c_add(c_mul(-1.0 / G.dx, dF), 
                                    c_add(c_mul(-1.0 / G.dy, dG),
                                    c_mul(-1.0 / G.dz, dH)));
            }
        }
    }
    return LU;
}

// --------------------------------- RK3 step ---------------------------------
static void advance_rk3(Grid& G, double dt, double ch, bool use_glm) {
    const int nx = G.nx(), ny = G.ny(), nz = G.nz();
    const int ng = G.ng;

    auto update = [&](Grid& dst, const Grid& base, const Grid& src,
                      double wBase, double wSrc, double wL,
                      const std::vector<Cons>& L)  {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    if (i >= ng && i < nx - ng && j >= ng && j < ny - ng && k >= ng 
                        && k < nz - ng && !is_interior_cell(i, j, k, base)) continue;
                    const int id = base.id(i, j, k);
                    dst.U[id] = c_add(c_mul(wBase, base.U[id]), c_add(c_mul(wSrc, src.U[id]), c_mul(wL, L[id])));
                }
            }
        }
    };

    apply_reflective_bc(G);
    enforce_floors(G);

    const auto L1 = rhs(G, ch, use_glm);
    Grid G1 = G;
    update(G1, G, G, 1.0, 0.0, dt, L1);
    apply_reflective_bc(G1);
    damp_glm_psi(G1, dt, ch, use_glm);
    enforce_floors(G1);

    const auto L2 = rhs(G1, ch, use_glm);
    Grid G2 = G;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (i >= ng && i < nx - ng && j >= ng && j < ny - ng && k >= ng 
                    && k < nz - ng && !is_interior_cell(i, j, k, G)) continue;
                const int id = G.id(i, j, k);
                G2.U[id] = c_add(c_mul(0.75, G.U[id]), c_add(c_mul(0.25, G1.U[id]), c_mul(0.25 * dt, L2[id])));
            }
        }
    }
    apply_reflective_bc(G2);
    damp_glm_psi(G2, dt, ch, use_glm);
    enforce_floors(G2);

    const auto L3 = rhs(G2, ch, use_glm);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (i >= ng && i < nx - ng && j >= ng && j < ny - ng && k >= ng 
                    && k < nz - ng && !is_interior_cell(i, j, k, G)) continue;
                const int id = G.id(i, j, k);
                G.U[id] = c_add(c_mul(1.0 / 3.0, G.U[id]), c_add(c_mul(2.0 / 3.0, G2.U[id]), c_mul(2.0 / 3.0 * dt, L3[id])));
            }
        }
    }
    apply_reflective_bc(G);
    damp_glm_psi(G, dt, ch, use_glm);
    enforce_floors(G);
}

// --------------------------------- plotting ---------------------------------
static void write_slice_2d(const std::string& path,
                            const Grid& G,
                            const std::vector<double>& field,
                            char plane) {
    std::ofstream f(path);
    f << std::setprecision(16);
    const int ng = G.ng;

    if (plane == 'z') {
        const int k = ng + G.Nz / 2;
        const double z = G.z0 + (k - ng + 0.5) * G.dz;
        f << "# z=0 midplane z=" << z << "\n";
        for (int j = ng; j < G.ny() - ng; ++j) {
            const double y = G.y0 + (j - ng + 0.5) * G.dy;
            for (int i = ng; i < G.nx() - ng; ++i) {
                const double x = G.x0 + (i - ng + 0.5) * G.dx;
                if (phi_sdf(x, y, z) <= 0.0) {
                    f << x << " " << y << " " << field[G.id(i, j, k)] << "\n";
                } else {
                    f << x << " " << y << " NaN\n";
                }
            }
            f << "\n";
        }
    } else if (plane == 'y') {
        const int j = ng + G.Ny / 2;
        const double y = G.y0 + (j - ng + 0.5) * G.dy;
        f << "# y=0 midplane y=" << y << "\n";
        for (int k = ng; k < G.nz() - ng; ++k) {
            const double z = G.z0 + (k - ng + 0.5) * G.dz;
            for (int i = ng; i < G.nx() - ng; ++i) {
                const double x = G.x0 + (i - ng + 0.5) * G.dx;
                if (phi_sdf(x, y, z) <= 0.0) {
                    f << x << " " << z << " " << field[G.id(i, j, k)] << "\n";
                } else {
                    f << x << " " << z << " NaN\n";
                }
            }
            f << "\n";
        }
    } else {
        const int i = ng + G.Nx / 2;
        const double x = G.x0 + (i - ng + 0.5) * G.dx;
        f << "# x=0 midplane x=" << x << "\n";
        for (int k = ng; k < G.nz() - ng; ++k) {
            const double z = G.z0 + (k - ng + 0.5) * G.dz;
            for (int j = ng; j < G.ny() - ng; ++j) {
                const double y = G.y0 + (j - ng + 0.5) * G.dy;
                if (phi_sdf(x, y, z) <= 0.0) {
                    f << z << " " << y << " " << field[G.id(i, j, k)] << "\n";
                } else {
                    f << z << " " << y << " NaN\n";
                }
            }
            f << "\n";
        }
    }
}

static void write_slice_line(const std::string& path, const Grid& G, double y_fixed) {
    constexpr int n_samples = 512;
    const double span = g_sdf.loaded ? (g_sdf.xmax - g_sdf.xmin) : 1.0;
    const double cx = g_sdf.loaded ? 0.5 * (g_sdf.xmin + g_sdf.xmax) : 0.0;

    std::ofstream f(path);
    f << "# s x y p_B rho p |v| |B|\n" << std::setprecision(16);

    for (int k = 0; k < n_samples; ++k) {
        const double s = (k + 0.5) / n_samples;
        double x = cx - 0.5 * span + s * span;
        double y = y_fixed;
        double z = 0.0;

        if (g_run_label == "ITER") y += 0.25;

        const Prim W = sample_prim_bilinear(G, x, y, z);
        const double Bmag = std::sqrt(W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz);
        const double vmag = std::sqrt(W.vx * W.vx + W.vy * W.vy + W.vz * W.vz);
        const double pmag = 0.5 * (W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz);

        f << s << " " << x << " " << y << " " << pmag
          << " " << W.rho << " " << W.p << " " << vmag
          << " " << Bmag << "\n";
    }
}

static void make_slice_plot(const std::string& base, const std::string& tag, double tout) {
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");

    const std::string dat = base + "/data/slice_" + tag + ".dat";
    const std::string plt = base + "/plots/plt/slice_" + tag + ".plt";
    const std::string png = base + "/plots/png/slice_" + tag + ".png";

    std::ofstream gp(plt);
    gp << "set term pngcairo size 1200,1000 enhanced\n";
    gp << "set output '" << png << "'\n";
    gp << "set multiplot layout 2,2 rowsfirst "
       << "title 'Reactor t = "
       << std::fixed << std::setprecision(3) << tout
       << "s' font ',16'\n";
    gp << "unset key\n";
    gp << "set border linewidth 1.2\n";
    gp << "set tics scale 0.6\n";
    gp << "set xrange [0:1]\n";
    gp << "set xlabel 'Position'\n";
    gp << "set pointsize 0.35\n";
    gp << "set grid\n";

    for (auto[t, c] : std::initializer_list<std::pair<const char*,int>> {
            {"Magnetic Pressure", 4},
            {"Density", 5},
            {"Gas Pressure", 6},
            {"Velocity Magnitude", 7}}
    ) {
        gp << "set title '" << t << "'\n";
        gp << "set autoscale y\n";
        gp << "plot '" << dat << "' u 1:" << c << " w p pt 7 ps 0.25 lc rgb '#000000'\n";
    }
    gp << "unset multiplot\n";
    gp.close();

    if (cmd_exists("gnuplot")) {
        syscmd("gnuplot \"" + plt + "\" >/dev/null 2>&1");
    }
}

static void make_colour_plot(const std::string& base,
                             const std::string& dat,
                             const std::string& png,
                             const std::string& title,
                             const std::string& cblabel,
                             const std::string& xlabel = "x",
                             const std::string& ylabel = "y") {
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");

    const std::string plt = base + "/plots/plt/" + png.substr(0, png.size() - 4) + ".plt";
    std::ofstream gp(plt);
    gp << "set term pngcairo size 900,800\n";
    gp << "set output '" << base << "/plots/png/" << png << "'\n";
    gp << "unset key\n";
    gp << "set size ratio -1\n";
    gp << "set xlabel '" << xlabel << "'\nset ylabel '" << ylabel << "'\n";
    gp << "set title '" << title << "'\n";
    gp << "set pm3d map\nset colorbox\n";
    gp << "set contour base\nset cntrparam levels 20\n";
    gp << "set cblabel '" << cblabel << "'\n";
    gp << "set palette viridis\n";
    gp << "set border linewidth 1.2\nset tics scale 0.6\n";
    gp << "set table $contours\n";
    gp << "splot '" << base << "/data/" << dat << "' u 1:2:3\n";
    gp << "unset table\n";
    gp << "splot '" << base << "/data/" << dat << "' u 1:2:3\n";
    gp << "unset table\n";
    gp << "splot '" << base << "/data/" << dat << "' u 1:2:3 w pm3d, "
       << "$contours u 1:2:3 w l lc rgb '#202020' lw 0.45\n";
    gp.close();

    if (cmd_exists("gnuplot")) {
        syscmd("gnuplot \"" + plt + "\" >/dev/null 2>&1");
    }
}

static void dump_fields(const Grid& G, const std::string& base,
                        double tout, const std::string& tag) {
    const int ng = G.ng, total = G.nx() * G.ny() * G.nz();
    std::vector<double> rho(total);
    std::vector<double> p(total);
    std::vector<double> pmag(total);
    std::vector<double> Bmag(total);
    std::vector<double> vmag(total);

    for (int k = ng; k < G.nz() - ng; ++k) {
        for (int j = ng; j < G.ny() - ng; ++j) {
            for (int i = ng; i < G.nx() - ng; ++i) {
                const Prim W = cons_to_prim(G.U[G.id(i, j, k)]);
                const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
                const int id = G.id(i, j, k);
                rho[id] = W.rho;
                p[id] = W.p;
                pmag[id] = 0.5 * B2;
                Bmag[id] = std::sqrt(B2);
                vmag[id] = std::sqrt(W.vx * W.vx + W.vy * W.vy + W.vz * W.vz);
            }
        }
    }

    using Row = std::tuple<std::string,std::vector<double>*,std::string>;
    for (auto& [name, vec, label] : std::vector<Row> {
        {"rho", &rho, "{/Symbol r}"},
        {"p", &p, "p"},
        {"pmag", &pmag, "p_B"},
        {"vmag", &vmag, "|v|"},
        {"Bmag", &Bmag, "|B|"}}) {

        const std::string fz0 = name + "_z_" + tag + ".dat";
        const std::string fy0 = name + "_y_" + tag + ".dat";
        const std::string fx0 = name + "_x_" + tag + ".dat";
        write_slice_2d(base + "/data/" + fz0, G, *vec, 'z');
        write_slice_2d(base + "/data/" + fy0, G, *vec, 'y');
        write_slice_2d(base + "/data/" + fx0, G, *vec, 'x');

        const std::string tstr = std::to_string(tout);
        make_colour_plot(base, fz0, name + "_z_" + tag + ".png",
            "Rotor " + label + " at t = " + tstr + "s", label, "x","y");
        make_colour_plot(base , fy0 , name + "_y_" + tag + ".png",
            "Rotor " + label + " at t = " + tstr + "s", label, "x","z");
        make_colour_plot(base, fx0, name + "_x_" + tag + ".png",
            "Rotor " + label + " at t = " + tstr + "s", label, "z","y");
    }

    const double y_line = 0.0;
    write_slice_line(base + "/data/slice_" + tag + ".dat", G, y_line);
    make_slice_plot(base, tag, tout);
}

// --------------------------------- timing -----------------------------------
class TimerRotor {
public:
    explicit TimerRotor(std::string label)
        : label_(std::move(label)),
          start_(std::chrono::steady_clock::now()) {}

    ~TimerRotor() {
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::cerr << "[rotor] " << label_ << " took " << ms << " ms\n";
    }

    TimerRotor(const TimerRotor&) = delete;
    TimerRotor& operator=(const TimerRotor&) = delete;

private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
};

// ------------------------------ thread selection -----------------------------
static int prompt_num_threads() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        std::cerr << "[rotor] Could not detect hardware concurrency; assuming 8.\n";
        hw = 8;
    }

    std::cout << "\n[rotor] Detected " << hw << " hardware thread(s) available.\n";
    std::cout << "[rotor] Number of OpenMP threads to use [default " << hw << "]: ";

    std::string s;
    std::getline(std::cin, s);

    int n_threads = static_cast<int>(hw);
    if (!s.empty()) {
        try {
            int requested = std::stoi(s);
            if (requested <= 0) {
                std::cerr << "[rotor] Ignoring non-positive value; using " << hw << ".\n";
            } else {
                n_threads = requested;
                if (static_cast<unsigned int>(n_threads) > hw) {
                    std::cerr << "[rotor] Warning: requested " << n_threads
                              << " threads, but only " << hw
                              << " hardware threads detected. Proceeding anyway.\n";
                }
            }
        } catch (...) {
            std::cerr << "[rotor] Invalid input, using " << hw << ".\n";
        }
    }

    std::cerr << "[rotor] Using " << n_threads << " threads\n";
    return n_threads;
}

// ---------------------------------- driver ----------------------------------
void run_rotor_3d() {

    const int n_threads = prompt_num_threads();
    omp_set_num_threads(n_threads);

    struct Opt{
        std::string label, path, name;
    };

    const std::vector<Opt> opts = {
        {"MAST  — sdf/MAST/mast.dat",  "sdf/MAST/mast.dat",  "MAST" },
        {"ITER  — sdf/ITER/iter.dat",  "sdf/ITER/iter.dat",  "ITER"},
        {"ST40  — sdf/ST40/st40.dat",  "sdf/ST40/st40.dat",  "ST40" },
        {"Other — enter path manually", "",                     ""    }
    };

    std::cout << "\n[rotor] Select SDF geometry:\n";
    for (std::size_t k = 0; k < opts.size(); ++k) {
        std::cout << "  " << (k + 1) << ")  " << opts[k].label << "\n";
    }
    std::cout << "[rotor] Choice [1-" << opts.size() << ", default 1]: ";

    std::string cs; std::getline(std::cin, cs);
    int ch = 1;
    if(!cs.empty()) {
        try {
            ch = std::stoi(cs);
        } catch(...) {
            ch = 1;
        }
        ch = std::max(1, std::min(ch, (int)opts.size()));
    }

    std::string sdf_path,run_label;
    if (ch < (int)opts.size()) {
        sdf_path = opts[ch - 1].path;
        run_label = opts[ch - 1].name;
    } else {
        std::cout << "[rotor] Enter path to SDF .dat file: ";
        std::getline(std::cin, sdf_path);
        const std::size_t sl = sdf_path.find_last_of("/\\");
        const std::string fn = (sl == std::string::npos) ? sdf_path : sdf_path.substr(sl + 1);
        const std::size_t dt = fn.find_last_of('.');
        run_label = (dt == std::string::npos) ? fn : fn.substr(0, dt);
    }

    g_run_label = run_label;

    if (run_label == "ITER") {
        rotor_center_y = 0.25;
    }
    
    if (sdf_path.empty()) {
        std::cerr << "[rotor] No path given.\n";
        return;
    }

    if (!g_sdf.load(sdf_path)) {
        std::cerr << "[rotor] Failed to load: " << sdf_path << "\n"
                  << "        Check sdf/ directory exists next to the binary.\n";
                  return;
                }
    std::cerr << "[rotor] Geometry: " << run_label << "\n";

    g_revolve_dirn[0] = 0.0;
    g_revolve_dirn[1] = 1.0;
    g_revolve_point[0] = 0.0;
    g_revolve_point[1] = 0.0;
    g_revolve_segments = 0;

    std::cout << "\n[rotor] Toroidal major radius R [default 0.5]: ";
    {
        std::string s; std::getline(std::cin, s);
        if (!s.empty()) {
            try {
                const double R = std::stod(s);
                if (R > 0.0) {
                    g_major_radius = R;
                } else if (R < 0.0) {
                    std::cerr << "[rotor] Ignoring non-positive R.\n";
                    g_major_radius = -R;
                } else if (R == 0.0) {
                    std::cerr << "[rotor] Ignoring zero R.\n";
                }
            } catch(...) {
                std::cerr << "[rotor] Invalid major radius.\n";
            }
        }
    }
    std::cerr << "[rotor] Toroidal major radius R = " << g_major_radius << "\n";
    rotor_center_x = g_major_radius - 0.10;
    std::cerr << "[rotor] Rotor centre x = " << rotor_center_x << "\n";

    std::cout << "\n[rotor] Cross-section scale s (1.0 = natural SDF size) [default 1.0]: ";
    {
        std::string s; std::getline(std::cin, s);
        if (!s.empty()) {
            try {
                const double sc = std::stod(s);
                if (sc > 0.0) {
                    g_minor_scale = sc;
                } else if (sc < 0.0) {
                    std::cerr << "[rotor] Ignoring non-positive scale.\n";
                    g_minor_scale = -sc;
                } else {
                    std::cerr << "[rotor] Ignoring zero scale.\n";
                }
            } catch(...) {
                std::cerr << "[rotor] Invalid scale.\n";
            }
        }
    }
    std::cerr << "[rotor] Toroidal minor scale s = " << g_minor_scale << "\n";

    const bool use_glm = prompt_use_glm();

    std::cout << "\n[rotor] Resolution N (cells per dimension, N³ total) [default 128]: ";
    std::string ns;
    std::getline(std::cin, ns);
    
    int N = 128;
    if (!ns.empty()) {
        try {
            N = std::max(1, std::stoi(ns)); 
        } catch(...) { 
            std::cerr << "[rotor] Invalid resolution.\n";
        }
    }
    std::cerr << "[rotor] N³ = " << N << "³ (" << (long long) N * N * N << " cells)\n";
    
    TimerRotor _timer_guard("run_rotor_3d");

    const std::vector<double> t_outs = {0.0, 0.1, 0.25, 0.5, 0.65};
    const std::vector<std::string> t_tags = {"t000", "t010", "t025", "t050", "t065"};
    const double tfinal = t_outs.back();

    Grid G;
    G.Nx = G.Ny = G.Nz = N;
    G.ng = 2;
    G.x0 = x_min;
    G.y0 = y_min;
    G.z0 = z_min;
    G.Lx = x_max - x_min;
    G.Ly = y_max - y_min;
    G.Lz = z_max - z_min;
    G.dx = G.Lx / G.Nx;
    G.dy = G.Ly / G.Ny;
    G.dz = G.Lz / G.Nz;
    G.U.assign(G.nx() * G.ny() * G.nz(), Cons{0,0,0,0,0,0,0,0,0});

    for (int k = 0; k < G.nz(); ++k)
        for (int j = 0; j < G.ny(); ++j)
            for (int i = 0; i < G.nx(); ++i) {
                const double x = x_min + (i - G.ng + 0.5) * G.dx;
                const double y = y_min + (j - G.ng + 0.5) * G.dy;
                const double z = z_min + (k - G.ng + 0.5) * G.dz;

                const double rho = std::sqrt(x * x + z * z);
                const double cphi = (rho > 1e-14) ? (x / rho) : 1.0;
                const double sphi = (rho > 1e-14) ? (z / rho) : 0.0;

                const double xi = rho - rotor_center_x;
                const double eta = y - rotor_center_y;
                const double r = std::sqrt(xi * xi + eta * eta);

                const double u_pol = -10.0 * eta;
                const double v_pol =  10.0 * xi;
                const double fr = (23.0 - 200.0 * r) / 3.0;

                Prim W{};
                W.p = p_rotor;
                W.Bx = Bx0;
                W.By = By0;
                W.Bz = Bz0;

                double rho_val, u_scale;
                if (r <= r0_rotor) {
                    rho_val = rho_core;
                    u_scale = 1.0;
                } else if (r <= r1_rotor) {
                    rho_val = rho_ambient + 9.0 * fr;
                    u_scale = fr;
                } else {
                    rho_val = rho_ambient;
                    u_scale = 0.0;
                }
                W.rho = rho_val;

                W.vx = u_scale * u_pol * cphi;
                W.vy = u_scale * v_pol;
                W.vz = u_scale * u_pol * sphi;

                G.U[G.id(i, j, k)] = prim_to_cons(prim_with_floors(W));
            }

    apply_reflective_bc(G);
    enforce_floors(G);

    const std::string base = "outputs/MHD/3D/Rotor/Reflective/" + run_label;
    mkdir_p(base+"/data");
    mkdir_p(base+"/plots/plt");
    mkdir_p(base+"/plots/png");
    dump_fields(G, base, t_outs[0], t_tags[0]);

    double t = 0.0;
    int step = 0;
    std::vector<bool> wrote(t_outs.size(), false);
    wrote[0] = true;

    while (t < tfinal - 1e-14) {
        double ch_glm = 0.0;
        double dt = compute_ch_and_dt(G, ch_glm, use_glm);

        for (std::size_t n = 1; n < t_outs.size(); ++n) {
            if (!wrote[n] && t < t_outs[n] && t + dt > t_outs[n]) {
                dt = t_outs[n] - t;
                break;
            }
        }

        if (t + dt > tfinal) {
            dt = tfinal - t;
        }

        if (!is_finite(dt) || dt <= 0.0) {
            std::cerr << "[rotor] invalid dt=" << dt
                      << " at t=" << t << " step=" << step << "\n";
            break;
        }

        advance_rk3(G, dt, ch_glm, use_glm);

        t += dt;
        ++step;

        for (std::size_t n = 1; n < t_outs.size(); ++n) {
            if (!wrote[n] && std::abs(t - t_outs[n]) < 1e-12) {
                dump_fields(G, base, t_outs[n], t_tags[n]);
                wrote[n] = true;
            }
        }
        
        if (step % 50 == 0||t >= tfinal) {
            std::cerr << "[rotor] step=" << step
                      <<" t=" << t << " dt=" << dt << " ch=" << ch_glm << "\n";
        }
    }

    std::cerr << "[rotor] Wrote data to " << base << "/data\n";
    std::cerr << "[rotor] Wrote plots to " << base << "/plots/png\n";
}