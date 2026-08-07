// ============================================================================
// Blast2DReactor.cpp — MHD blast-wave test (2D) with SDF-based boundary
// ----------------------------------------------------------------------------
//   - HLLD fluxes
//   - WENO3 reconstruction
//   - SSP-RK3 time stepping
//   - Optional Dedner-style GLM hyperbolic divergence cleaning
//   - Reflective boundary defined by an external SDF .dat file
// 
// Initial conditions (axisymmetric):
//   rho = 1, v = (0, 0, 0)
//   B   = (1/sqrt(4*pi)) * (0, 10/sqrt(eps), 0)
//   p   = 10/eps  if r <= R = 0.1
//         0.1/eps otherwise            with eps = 10
// ============================================================================
#include "Blast2DReactor.h"

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

static inline double clamp_pos(double x, double eps = 1e-10) {
    return (x < eps) ? eps : x;
}

static inline bool is_finite(double x) {
    return std::isfinite(x);
}

// -------------------------------- parameters --------------------------------
static constexpr double gamma_gas = 5.0 / 3.0;
static constexpr double CFL = 0.60;

static constexpr double x_min = -6;
static constexpr double x_max = 6;
static constexpr double y_min = -6;
static constexpr double y_max = 6;
static constexpr double SDF_SCALE = 6;
static double blast_center_x = -0.5;
static double blast_center_y = 0.0;
static std::string g_run_label;

static constexpr double blast_radius = 0.85;
static constexpr double rho_ambient = 1.0;
static constexpr double eps_weak = 10.0;
static constexpr double p_inner = 10.0 / eps_weak;
static constexpr double p_outer = 0.1 / eps_weak;

static const double B_mag = 1.0 / std::sqrt(4.0 * M_PI);
static const double Bx0 = 0.0;
static const double By0 = (B_mag * 10.0) / std::sqrt(eps_weak);

// ----------------------------- SDF construction -----------------------------
struct SdfTableBlast2D {
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0, dx = 0, dy = 0;
    int nx = 0, ny = 0;
    std::vector<double> sdf;
    bool loaded = false;
    mutable bool last_gradient_used_fallback = false;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) { std::cerr << "[blast wave] Cannot open: "<<path<<"\n"; return false; }

        std::vector<double> xs, ys, vs;
        std::string line;

        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            double x, y, v;
            if (!(ss >> x >> y >> v)) continue;
            xs.push_back(x); 
            ys.push_back(y); 
            vs.push_back(v);
        }
        if (xs.empty()) { std::cerr << "[blast wave] No data in: " << path << "\n"; return false; }

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
            std::cerr << "[blast wave] Cannot determine nx.\n";
            return false;
        }

        ny = static_cast<int> (xs.size()) / nx;
        if (nx * ny != static_cast<int> (xs.size())) {
            std::cerr << "[blast wave] Size " << xs.size() << " not divisible by nx = " << nx << "\n";
            return false;
        }

        dx = (xmax - xmin) / (nx - 1);
        dy = (ymax - ymin) / (ny - 1);
        sdf = vs;
        loaded = true;

        std::cerr << "[blast wave] Loaded " << nx << "x" << ny << " from " << path << "\n";

        const double cx = 0.5 * (xmin + xmax);
        const double cy = 0.5 * (ymin + ymax);
        const double hx = 0.5 * (xmax - xmin); 
        const double hy = 0.5 * (ymax - ymin);
        const double scale = SDF_SCALE / std::max(hx, hy);

        xmin = (xmin - cx) * scale;
        xmax = (xmax - cx) * scale;
        ymin = (ymin - cy) * scale;
        ymax = (ymax - cy) * scale;
        dx *= scale; dy *= scale;
        for (double& v : sdf) v *= scale;

        std::cerr << "[blast wave] Normalised: x[" << xmin << "," << xmax << "]  y[" << ymin << "," << ymax << "]\n";
        std::cerr << "[blast wave] Scale factor: " << scale << "\n";
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

        double fi = (x - xmin) / dx;
        double fj = (y - ymin) / dy;
        int i0 = std::max(0, std::min((int)std::floor(fi), nx - 2));
        int j0 = std::max(0, std::min((int)std::floor(fj), ny - 2));
        const double tx = fi - i0, ty = fj - j0;

        const double v00 = sdf[j0 * nx + i0];
        const double v10 = sdf[j0 * nx + (i0 + 1)];
        const double v01 = sdf[(j0 + 1) * nx + i0];
        const double v11 = sdf[(j0 + 1) * nx + (i0 + 1)];
        return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11;
    }

     void gradient(double r, double y, double& gr, double& gy) const {
        last_gradient_used_fallback = false;
        for (int mult = 1; mult <= 8; mult *= 2) {
            const double er = dx * mult;
            const double ey = dy * mult;
            gr = (interp(r + er, y) - interp(r - er, y)) / (2.0 * er);
            gy = (interp(r, y + ey) - interp(r, y - ey)) / (2.0 * ey);
            const double mag = std::sqrt(gr * gr + gy * gy);
            if (mag > 1e-9) {
                gr /= mag;
                gy /= mag;
                return;
            }
        }

        last_gradient_used_fallback = true;
        const double cx = 0.5 * (xmin + xmax);
        const double cy = 0.5 * (ymin + ymax);
        const double rx = r - cx, ry = y - cy;
        const double rmag = std::sqrt(rx * rx + ry * ry);
        if (rmag > 1e-12) {
            gr = rx / rmag;
            gy = ry / rmag;
        } else {
            gr = 1.0;
            gy = 0.0;
        }
    }
};

static SdfTableBlast2D g_sdf;

// ----------------------------- rotation support -----------------------------
static double g_box_theta_rad = 0.0;
static double g_box_cos_theta = 1.0;
static double g_box_sin_theta = 0.0;

static void set_box_rotation_angle(double theta_degrees) {
    g_box_theta_rad = theta_degrees * M_PI / 180.0;
    g_box_cos_theta = std::cos(g_box_theta_rad);
    g_box_sin_theta = std::sin(g_box_theta_rad);
}

static inline void transform_to_box_frame(double& x, double& y) {
    const double x_box = x * g_box_cos_theta + y * g_box_sin_theta;
    const double y_box = -x * g_box_sin_theta + y * g_box_cos_theta;
    x = x_box;
    y = y_box;
}

static inline void transform_from_box_frame(double& x, double& y) {
    const double x_lab = x * g_box_cos_theta - y * g_box_sin_theta;
    const double y_lab = x * g_box_sin_theta + y * g_box_cos_theta;
    x = x_lab;
    y = y_lab;
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
    W.rho = std::max(W.rho, 1e-6 * rho_ambient);
    W.p = std::max(W.p, 1e-6 * p_outer);
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

static bool prompt_use_glm() {
    std::cout << "[blast wave] Enable hyperbolic divergence cleaning (GLM)? [y/n, default y]: ";
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
    const double pt = W.p + 0.5 * B2;
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
    const Cons FL = flux_x(UL, ch, use_glm);
    const Cons FR = flux_x(UR, ch, use_glm);

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

static inline Cons rotate_xy_cons(const Cons& U) {
    return {U.rho, U.my, U.mx, U.mz, U.By, U.Bx, U.Bz, U.E, U.psi};
}

static inline Cons rotate_xy_flux_back(const Cons& F) {
    return {F.rho, F.my, F.mx, F.mz, F.By, F.Bx, F.Bz, F.E, F.psi};
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
    int Nx = 0, Ny = 0, ng = 2;
    double x0 = 0.0, y0 = 0.0, Lx = 1.0, Ly = 1.0, dx = 0.0, dy = 0.0;
    std::vector<Cons> U;

    int nx() const { return Nx + 2 * ng; }
    int ny() const { return Ny + 2 * ng; }
    inline int id(int i, int j) const { return j * nx() + i; }
};

static inline Prim bilinear_interp(const Prim& W00, const Prim& W10,
                                   const Prim& W01, const Prim& W11,
                                   double tx, double ty) {
    tx = std::max(0.0, std::min(1.0, tx));
    ty = std::max(0.0, std::min(1.0, ty));

    Prim W{};
    auto interp = [&](double f00, double f10, double f01, double f11) {
        const double val =
            (1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10 +
            (1.0 - tx) * ty * f01 + tx * ty * f11;
        return std::isfinite(val) ? val : 0.0;
    };

    W.rho = interp(W00.rho, W10.rho, W01.rho, W11.rho);
    W.vx = interp(W00.vx, W10.vx, W01.vx, W11.vx);
    W.vy = interp(W00.vy, W10.vy, W01.vy, W11.vy);
    W.vz = interp(W00.vz, W10.vz, W01.vz, W11.vz);
    W.p = interp(W00.p, W10.p, W01.p, W11.p);
    W.Bx = interp(W00.Bx, W10.Bx, W01.Bx, W11.Bx);
    W.By = interp(W00.By, W10.By, W01.By, W11.By);
    W.Bz = interp(W00.Bz, W10.Bz, W01.Bz, W11.Bz);
    W.psi = interp(W00.psi, W10.psi, W01.psi, W11.psi);
    return prim_with_floors(W);
}

static inline double phi_sdf(double x, double y) {
    transform_to_box_frame(x,y);
    return g_sdf.interp(x,y);
}

static inline void sdf_normal(double x,double y,double& nx_out,double& ny_out){
    double xb = x;
    double yb = y;
    transform_to_box_frame(xb,yb);

    double gx, gy; 
    g_sdf.gradient(xb, yb, gx, gy);

    nx_out = gx * g_box_cos_theta - gy * g_box_sin_theta;
    ny_out = gx * g_box_sin_theta + gy * g_box_cos_theta;
}

static Prim sample_prim_bilinear(const Grid& G, double x, double y) {
    const double x_lo = G.x0 + G.dx * 0.501;
    const double x_hi = G.x0 + G.Lx - G.dx * 0.501;
    const double y_lo = G.y0 + G.dy * 0.501;
    const double y_hi = G.y0 + G.Ly - G.dy * 0.501;
    x = std::max(x_lo, std::min(x, x_hi));
    y = std::max(y_lo, std::min(y, y_hi));

    const double i_float = (x - G.x0) / G.dx + G.ng - 0.5;
    const double j_float = (y - G.y0) / G.dy + G.ng - 0.5;

    const int i0 = static_cast<int>(std::floor(i_float));
    const int j0 = static_cast<int>(std::floor(j_float));
    const double tx = i_float - i0;
    const double ty = j_float - j0;

    const int i0c = std::max(G.ng, std::min(i0, G.Nx + G.ng - 2));
    const int i1c = std::min(i0c + 1, G.Nx + G.ng - 1);
    const int j0c = std::max(G.ng, std::min(j0, G.Ny + G.ng - 2));
    const int j1c = std::min(j0c + 1, G.Ny + G.ng - 1);

    const Prim W00 = cons_to_prim(G.U[G.id(i0c, j0c)]);
    const Prim W10 = cons_to_prim(G.U[G.id(i1c, j0c)]);
    const Prim W01 = cons_to_prim(G.U[G.id(i0c, j1c)]);
    const Prim W11 = cons_to_prim(G.U[G.id(i1c, j1c)]);
    return bilinear_interp(W00, W10, W01, W11, tx, ty);
}

static inline bool is_interior_cell(int i, int j, const Grid& G) {
    const double x = G.x0 + (i - G.ng + 0.5) * G.dx;
    const double y = G.y0 + (j - G.ng + 0.5) * G.dy;
    return phi_sdf(x, y) <= 0.0;
}

static void apply_reflective_bc(Grid& G) {
    const int nx = G.nx();
    const int ny = G.ny();
    const double h = std::max(G.dx, G.dy);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = G.x0 + (i - G.ng + 0.5) * G.dx;
            const double y = G.y0 + (j - G.ng + 0.5) * G.dy;
            if (phi_sdf(x, y) <= 0.0) continue;

            const double phi = phi_sdf(x, y);
            double nxw;
            double nyw;
            sdf_normal(x, y, nxw, nyw);

            const double xs_mirror = x - 2.0 * phi * nxw;
            const double ys_mirror = y - 2.0 * phi * nyw;

            Prim W1{};
            if (phi_sdf(xs_mirror, ys_mirror) <= 0.0) {
                W1 = sample_prim_bilinear(G, xs_mirror, ys_mirror);
            } else {
                const double xs = x - (phi + h) * nxw;
                const double ys = y - (phi + h) * nyw;
                if (phi_sdf(xs, ys) <= 0.0) {
                    W1 = sample_prim_bilinear(G, xs, ys);
                } else {
                    W1.rho = rho_ambient;
                    W1.p = p_outer;
                    W1.vx = W1.vy = W1.vz = 0.0;
                    W1.Bx = Bx0;
                    W1.By = By0;
                    W1.Bz = 0.0;
                    W1.psi = 0.0;
                }
            }

            const double Bn0 = Bx0 * nxw + By0 * nyw;
            const bool no_slip = std::abs(Bn0) > 1e-12;

            const double vn = W1.vx * nxw + W1.vy * nyw;

            double vx_g, vy_g, vz_g;
            if (no_slip) {
                vx_g = -W1.vx;
                vy_g = -W1.vy;
                vz_g = -W1.vz;
            } else {
                vx_g = W1.vx - 2.0 * vn * nxw;
                vy_g = W1.vy - 2.0 * vn * nyw;
                vz_g = W1.vz;
            }

            const double Bn_lab = W1.Bx * nxw + W1.By * nyw;
            const double Btx = W1.Bx - Bn_lab * nxw;
            const double Bty = W1.By - Bn_lab * nyw;

            const double Bn_ghost = -Bn_lab + 2.0 * Bn0;

            const double Bx_g = Bn_ghost * nxw + Btx;
            const double By_g = Bn_ghost * nyw + Bty;
            const double Bz_g = W1.Bz;

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

            G.U[G.id(i, j)] = prim_to_cons(prim_with_floors(Wg));
        }
    }
}

static void enforce_floors(Grid& G) {
    const int ng = G.ng, nx = G.nx(), ny = G.ny();
    constexpr double hard_limit = 1e8;
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            Prim W = cons_to_prim(G.U[G.id(i, j)]);
            if (!is_finite(W.rho) || !is_finite(W.p) || !is_finite(W.vx) ||
                !is_finite(W.vy) || !is_finite(W.vz) || !is_finite(W.Bx) ||
                !is_finite(W.By) || !is_finite(W.Bz) || !is_finite(W.psi) ||
                std::abs(W.vx) > hard_limit || std::abs(W.vy) > hard_limit ||
                std::abs(W.vz) > hard_limit || std::abs(W.Bx) > hard_limit ||
                std::abs(W.By) > hard_limit || std::abs(W.Bz) > hard_limit ||
                std::abs(W.psi) > hard_limit) {
                W.rho = rho_ambient;
                W.vx = W.vy = W.vz = 0.0;
                W.p = p_outer;
                W.Bx = Bx0;
                W.By = By0;
                W.Bz = W.psi = 0.0;
            }
            G.U[G.id(i, j)] = prim_to_cons(prim_with_floors(W));
        }
    }
}

static void damp_glm_psi(Grid& G, double dt, double ch, bool use_glm) {
    if (!use_glm || ch <= 0.0 || dt <= 0.0) return;
    const int ng = G.ng, nx = G.nx(), ny = G.ny();
    const double h = std::min(G.dx, G.dy);
    const double alpha = 0.18;
    const double rate = ch * alpha / std::max(h, 1e-14);
    const double factor = std::exp(-dt * rate);
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            if (!is_interior_cell(i, j, G)) {
                continue;
            }
            G.U[G.id(i, j)].psi *= factor;
        }
    }
}

// -------------------------------- time step ---------------------------------
static double compute_ch_and_dt(const Grid& G, double& ch_out, bool use_glm) {
    double amax = 1e-14;
    for (int j = G.ng; j < G.ny() - G.ng; ++j) {
        for (int i = G.ng; i < G.nx() - G.ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            const Prim W = cons_to_prim(G.U[G.id(i, j)]);
            if (!is_finite(W.rho) || !is_finite(W.p) ||
                !is_finite(W.vx) || !is_finite(W.vy) ||
                !is_finite(W.Bx) || !is_finite(W.By)) {
                    continue;
                }
            const double ax = std::abs(W.vx) + fast_speed_dir(W, W.Bx);
            const double ay = std::abs(W.vy) + fast_speed_dir(W, W.By);
            if (is_finite(ax)) amax = std::max(amax, ax);
            if (is_finite(ay)) amax = std::max(amax, ay);
        }
    }
    amax = std::max(amax, 1e-8);
    ch_out = use_glm ? amax : 0.0;
    return std::max(CFL * std::min(G.dx, G.dy) / amax, 1e-8);
}

// ---------------------------- semi-discrete RHS -----------------------------
static std::vector<Cons> rhs(const Grid& G, double ch, bool use_glm) {
    const int ng = G.ng, nx = G.nx(), ny = G.ny();
    std::vector<Cons> LU(nx * ny, Cons{0,0,0,0,0,0,0,0,0});

    std::vector<Prim> W(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            W[G.id(i, j)] = cons_to_prim(G.U[G.id(i, j)]);
        }
    }

    constexpr double min_wall_dist_cells = 2.0;
    const double h_min = std::min(G.dx, G.dy);
    auto near_wall = [&](int i, int j) {
        const double x = G.x0 + (i - G.ng + 0.5) * G.dx;
        const double y = G.y0 + (j - G.ng + 0.5) * G.dy;
        return std::abs(phi_sdf(x, y)) < min_wall_dist_cells * h_min;
    };

    std::vector<Cons> Fx((nx - 1) * ny);
    auto fx_id = [&](int i, int j) { return j * (nx - 1) + i; };
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng - 1; i <= nx - ng - 1; ++i) {
            if (near_wall(i - 1, j) || near_wall(i, j) ||
                near_wall(i + 1, j) || near_wall(i + 2, j)) {
                const Cons UL = prim_to_cons(prim_with_floors(W[G.id(i, j)]));
                const Cons UR = prim_to_cons(prim_with_floors(W[G.id(i + 1, j)]));
                Fx[fx_id(i, j)] = hlld_x(UL, UR, ch, use_glm);
            } else {
                Prim WL = reconstruct_left_weno3(W[G.id(i - 1, j)], W[G.id(i, j)], W[G.id(i + 1, j)]);
                Prim WR = reconstruct_right_weno3(W[G.id(i, j)], W[G.id(i + 1, j)], W[G.id(i + 2, j)]);
                const Cons UL = prim_to_cons(prim_with_floors(WL));
                const Cons UR = prim_to_cons(prim_with_floors(WR));
                Fx[fx_id(i, j)] = hlld_x(UL, UR, ch, use_glm);
            }
        }
    }

    std::vector<Cons> Gy(nx * (ny - 1));
    auto gy_id = [&](int i, int j) { return j * nx + i; };

    for (int j = ng - 1; j <= ny - ng - 1; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            if (near_wall(i, j - 1) || near_wall(i, j) ||
                near_wall(i, j + 1) || near_wall(i, j + 2)) {
                const Cons UL = prim_to_cons(prim_with_floors(W[G.id(i, j)]));
                const Cons UR = prim_to_cons(prim_with_floors(W[G.id(i, j + 1)]));
                Gy[gy_id(i, j)] = hlld_y(UL, UR, ch, use_glm);
            } else {
                Prim WL = reconstruct_left_weno3(W[G.id(i, j - 1)], W[G.id(i, j)], W[G.id(i, j + 1)]);
                Prim WR = reconstruct_right_weno3(W[G.id(i, j)], W[G.id(i, j + 1)], W[G.id(i, j + 2)]);
                WL = prim_with_floors(WL);
                WR = prim_with_floors(WR);
                Gy[gy_id(i, j)] = hlld_y(prim_to_cons(WL), prim_to_cons(WR), ch, use_glm);
            }
        }
    }

    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            const Cons dFx = c_sub(Fx[fx_id(i, j)], Fx[fx_id(i - 1, j)]);
            const Cons dGy = c_sub(Gy[gy_id(i, j)], Gy[gy_id(i, j - 1)]);
            Cons L = c_add(c_mul(-1.0 / G.dx, dFx), c_mul(-1.0 / G.dy, dGy));

            LU[G.id(i, j)] = L;
        }
    }
    return LU;
}

// --------------------------------- RK3 step ---------------------------------
static double peak_wave_speed(const Grid& G, int* out_i = nullptr, int* out_j = nullptr) {
    double amax = 1e-14;
    int amax_i = -1, amax_j = -1;
    for (int j = G.ng; j < G.ny() - G.ng; ++j) {
        for (int i = G.ng; i < G.nx() - G.ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            const Prim W = cons_to_prim(G.U[G.id(i, j)]);
            if (!is_finite(W.rho) || !is_finite(W.p) ||
                !is_finite(W.vx) || !is_finite(W.vy) ||
                !is_finite(W.Bx) || !is_finite(W.By)) continue;
            const double ax = std::abs(W.vx) + fast_speed_dir(W, W.Bx);
            const double ay = std::abs(W.vy) + fast_speed_dir(W, W.By);
            if (is_finite(ax) && ax > amax) { amax = ax; amax_i = i; amax_j = j; }
            if (is_finite(ay) && ay > amax) { amax = ay; amax_i = i; amax_j = j; }
        }
    }
    if (out_i) *out_i = amax_i;
    if (out_j) *out_j = amax_j;
    return amax;
}

static bool advance_rk3(Grid& G, double dt, double ch, bool use_glm,
                         bool verbose_on_reject = false) {
    const int nx = G.nx(), ny = G.ny();
    constexpr double growth_tolerance = 2.0;

    apply_reflective_bc(G);
    enforce_floors(G);

    const std::vector<Cons> L1 = rhs(G, ch, use_glm);
    Grid G1 = G;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (i >= G.ng && i < nx - G.ng && j >= G.ng && j < ny - G.ng &&
            !is_interior_cell(i, j, G)) {
                continue;
            }
            G1.U[G.id(i, j)] = c_add(G.U[G.id(i, j)], c_mul(dt, L1[G.id(i, j)]));
        }
    }

    apply_reflective_bc(G1);
    damp_glm_psi(G1, dt, ch, use_glm);
    enforce_floors(G1);

    const double ch1 = peak_wave_speed(G1);
    if (ch1 > growth_tolerance * std::max(ch, 1e-8)) {
        if (verbose_on_reject) {
            int ri = -1, rj = -1;
            peak_wave_speed(G1, &ri, &rj);
            if (ri >= 0) {
                const double x = G.x0 + (ri - G.ng + 0.5) * G.dx;
                const double y = G.y0 + (rj - G.ng + 0.5) * G.dy;
                const double phi = phi_sdf(x, y);
                const Prim W = cons_to_prim(G1.U[G1.id(ri, rj)]);
                std::cerr << "  [reject@substage1] ch1=" << ch1
                          << " at (i,j)=(" << ri << "," << rj << ") (x,y)=("
                          << x << "," << y << ") phi=" << phi
                          << " rho=" << W.rho << " p=" << W.p << "\n";
            }
        }
        return false;
    }

    const std::vector<Cons> L2 = rhs(G1, ch, use_glm);
    Grid G2 = G;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (i >= G.ng && i < nx - G.ng && j >= G.ng && j < ny - G.ng &&
            !is_interior_cell(i, j, G)) {
                continue;
            }
            G2.U[G.id(i, j)] = c_add(c_mul(0.75, G.U[G.id(i, j)]),
                c_mul(0.25, c_add(G1.U[G.id(i, j)], c_mul(dt, L2[G.id(i, j)]))));
        }
    }

    apply_reflective_bc(G2);
    damp_glm_psi(G2, dt, ch, use_glm);
    enforce_floors(G2);

    const double ch2 = peak_wave_speed(G2);
    if (ch2 > growth_tolerance * std::max(ch, 1e-8)) {
        if (verbose_on_reject) {
            int ri = -1, rj = -1;
            peak_wave_speed(G2, &ri, &rj);
            if (ri >= 0) {
                const double x = G.x0 + (ri - G.ng + 0.5) * G.dx;
                const double y = G.y0 + (rj - G.ng + 0.5) * G.dy;
                const double phi = phi_sdf(x, y);
                const Prim W = cons_to_prim(G2.U[G2.id(ri, rj)]);
                std::cerr << "  [reject@substage2] ch2=" << ch2
                          << " at (i,j)=(" << ri << "," << rj << ") (x,y)=("
                          << x << "," << y << ") phi=" << phi
                          << " rho=" << W.rho << " p=" << W.p << "\n";
            }
        }
        return false;
    }

    const std::vector<Cons> L3 = rhs(G2, ch, use_glm);
    Grid G_new = G;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (i >= G.ng && i < nx - G.ng && j >= G.ng && j < ny - G.ng &&
            !is_interior_cell(i, j, G)) {
                continue;
            }
            G_new.U[G.id(i, j)] = c_add(c_mul(1.0 / 3.0, G.U[G.id(i, j)]),
                c_mul(2.0 / 3.0, c_add(G2.U[G.id(i, j)], c_mul(dt, L3[G.id(i, j)]))));
        }
    }

    apply_reflective_bc(G_new);
    damp_glm_psi(G_new, dt, ch, use_glm);
    enforce_floors(G_new);

    const double ch3 = peak_wave_speed(G_new);
    if (ch3 > growth_tolerance * std::max(ch, 1e-8)) {
        if (verbose_on_reject) {
            int ri = -1, rj = -1;
            peak_wave_speed(G_new, &ri, &rj);
            if (ri >= 0) {
                const double x = G.x0 + (ri - G.ng + 0.5) * G.dx;
                const double y = G.y0 + (rj - G.ng + 0.5) * G.dy;
                const double phi = phi_sdf(x, y);
                const Prim W = cons_to_prim(G_new.U[G_new.id(ri, rj)]);
                std::cerr << "  [reject@substage3] ch3=" << ch3
                          << " at (i,j)=(" << ri << "," << rj << ") (x,y)=("
                          << x << "," << y << ") phi=" << phi
                          << " rho=" << W.rho << " p=" << W.p << "\n";
            }
        }
        return false;
    }

    G = G_new;
    return true;
}

// --------------------------------- plotting ---------------------------------
static void write_field_xy(const std::string& path,
                           const Grid& G,
                           const std::vector<double>& field,
                           const std::string& header) {
    std::ofstream f(path);
    f << "# " << header << "\n" << std::setprecision(16);
    const int ng = G.ng;
    for (int j = ng; j < G.ny() - ng; ++j) {
        const double y = G.y0 + (j - ng + 0.5) * G.dy;
        for (int i = ng; i < G.nx() - ng; ++i) {
            const double x = G.x0 + (i - ng + 0.5) * G.dx;
            if (phi_sdf(x, y) <= 0.0) {
                f << x << " " << y << " " << field[G.id(i, j)] << "\n";
            } else {
                f << x << " " << y << " NaN\n";
            }
        }
        f << "\n";
    }
}

static void make_colour_plot(const std::string& base,
                             const std::string& dat,
                             const std::string& png,
                             const std::string& title,
                             const std::string& cblabel) {
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");

    const std::string pltfile = base + "/plots/plt/" + png.substr(0, png.size() - 4) + ".plt";
    std::ofstream gp(pltfile);
    gp << "set term pngcairo size 900,800\n";
    gp << "set output '" << base << "/plots/png/" << png << "'\n";
    gp << "unset key\n";
    gp << "set size ratio -1\n";
    gp << "set xlabel 'x'\nset ylabel 'y'\n";
    gp << "set xrange [" << x_min << ":" << x_max << "]\n";
    gp << "set yrange [" << y_min << ":" << y_max << "]\n";
    gp << "set title '" << title << "'\n";
    gp << "set pm3d map\nset colorbox\n";
    gp << "set contour base\nset cntrparam levels 20\n";
    gp << "set cblabel '" << cblabel << "'\n";
    gp << "set palette viridis\n";
    gp << "set border linewidth 1.2\nset tics scale 0.6\n";
    gp << "set table $contours\n";
    gp << "splot '"<<base<<"/data/"<<dat<<"' u 1:2:3\n";
    gp << "unset table\n";
    gp << "splot '"<<base<<"/data/"<<dat<<"' u 1:2:3 w pm3d, "
       << "$contours u 1:2:3 w l lc rgb '#202020' lw 0.45\n";
    gp.close();

    if (cmd_exists("gnuplot")) {
        syscmd("gnuplot \"" + pltfile + "\" >/dev/null 2>&1");
    }
}

static void write_slice(const std::string& path, const Grid& G) {
    constexpr int n_samples = 512;
    const double span = g_sdf.loaded ? (g_sdf.xmax - g_sdf.xmin) : 1.0;
    const double cx = g_sdf.loaded ? 0.5 * (g_sdf.xmin + g_sdf.xmax) : 0.0;

    std::ofstream f(path);
    f << "# s x y p_B rho p |v| |B|\n" << std::setprecision(16);

    for (int k = 0; k < n_samples; ++k) {
        const double s = (k + 0.5) / n_samples;
        double x = cx - 0.5 * span + s * span, y = 0.5 * (g_sdf.loaded ? g_sdf.ymin + g_sdf.ymax : 0.0);
        if (g_run_label == "ITER") y += 1.375;        
        transform_from_box_frame(x, y);

        const Prim W = sample_prim_bilinear(G, x, y);
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

    for (auto[t, c]:std::initializer_list<std::pair<const char*,int>> {
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

static void dump_fields(const Grid& G, const std::string& base, double tout, const std::string& tag) {
    const int ng = G.ng;
    std::vector<double> rho(G.nx() * G.ny(), 0.0);
    std::vector<double> p(G.nx() * G.ny(), 0.0);
    std::vector<double> pmag(G.nx() * G.ny(), 0.0);
    std::vector<double> Bmag(G.nx() * G.ny(), 0.0);
    std::vector<double> vmag(G.nx() * G.ny(), 0.0);

    for (int j = ng; j < G.ny() - ng; ++j) {
        for (int i = ng; i < G.nx() - ng; ++i) {
            const Prim W = cons_to_prim(G.U[G.id(i, j)]);
            const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
            rho[G.id(i, j)] = W.rho;
            p[G.id(i, j)] = W.p;
            pmag[G.id(i, j)] = 0.5 * B2;
            Bmag[G.id(i, j)] = std::sqrt(B2);
            vmag[G.id(i, j)] = std::sqrt(W.vx * W.vx + W.vy * W.vy + W.vz * W.vz);
        }
    }

    write_field_xy(base + "/data/rho_" + tag + ".dat", G, rho, "x y rho");
    write_field_xy(base + "/data/p_" + tag + ".dat", G, p, "x y p");
    write_field_xy(base + "/data/pmag_" + tag + ".dat", G, pmag, "x y pmag");
    write_field_xy(base + "/data/vmag_" + tag + ".dat", G, vmag, "x y vmag");
    write_field_xy(base + "/data/Bmag_" + tag + ".dat", G, Bmag, "x y Bmag");
    write_slice(base + "/data/slice_" + tag + ".dat", G);

    make_colour_plot(base, "rho_" + tag + ".dat", "rho_" + tag + ".png",
                     "Blast Wave {/Symbol r} at t = " + std::to_string(tout) + "s", "{/Symbol r}");
    make_colour_plot(base, "p_" + tag + ".dat", "p_" + tag + ".png",
                     "Blast Wave p at t = " + std::to_string(tout) + "s", "p");
    make_colour_plot(base, "pmag_" + tag + ".dat", "pmag_" + tag + ".png",
                     "Blast Wave p_B at t = " + std::to_string(tout) + "s", "p_B");
    make_colour_plot(base, "vmag_" + tag + ".dat", "vmag_" + tag + ".png",
                     "Blast Wave |v| at t = " + std::to_string(tout) + "s", "|v|");
    make_colour_plot(base, "Bmag_" + tag + ".dat", "Bmag_" + tag + ".png",
                     "Blast Wave |B| at t = " + std::to_string(tout) + "s", "|B|");
    make_slice_plot(base, tag, tout);
}

// ---------------------------------- driver ----------------------------------
void run_blast_reactor() {

    struct Opt{
        std::string label, path, name;
    };

    const std::vector<Opt> opts = {
        {"MAST  — sdf/MAST/mast.dat",  "sdf/MAST/mast.dat",  "MAST" },
        {"ITER  — sdf/ITER/iter.dat",  "sdf/ITER/iter.dat",  "ITER"},
        {"ST40  — sdf/ST40/st40.dat",  "sdf/ST40/st40.dat",  "ST40" },
        {"Other — enter path manually", "",                     ""    }
    };

    std::cout << "\n[blast wave] Select SDF geometry:\n";
    for (std::size_t k = 0; k < opts.size(); ++k) {
        std::cout << "  " << (k + 1) << ")  " << opts[k].label << "\n";
    }
    std::cout << "[blast wave] Choice [1-" << opts.size() << ", default 1]: ";

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
        std::cout << "[blast wave] Enter path to SDF .dat file: ";
        std::getline(std::cin, sdf_path);
        const std::size_t sl = sdf_path.find_last_of("/\\");
        const std::string fn = (sl == std::string::npos) ? sdf_path : sdf_path.substr(sl + 1);
        const std::size_t dt = fn.find_last_of('.');
        run_label = (dt == std::string::npos) ? fn : fn.substr(0, dt);
    }

    g_run_label = run_label;
    if (run_label == "ITER") {
        blast_center_y = 1.375;
    }

    if (sdf_path.empty()) {
        std::cerr << "[blast wave] No path given.\n";
        return;
    }

    if (!g_sdf.load(sdf_path)) {
        std::cerr << "[blast wave] Failed to load: " << sdf_path << "\n"
                  << "        Check sdf/ directory exists next to the binary.\n";
                  return;
                }
    std::cerr << "[blast wave] Geometry: " << run_label << "\n";

    std::cout << "\n[blast wave] Rotation angle theta (degrees) [default 0]: ";
    std::string ts;
    std::getline(std::cin, ts);

    double theta = 0.0;
    if (!ts.empty()) {
        try {
            theta = std::stod(ts);
        } catch(...) {
            std::cerr << "[blast wave] Invalid rotation angle.\n";
        }
    }
    set_box_rotation_angle(theta);
    std::cerr << "[blast wave] Rotation theta = " << theta << " deg\n";

    const bool use_glm = prompt_use_glm();

    std::cout << "\n[blast wave] Enter resolution N (cells per dimension, N² total) [default 256]: ";
    std::string N_str;
    std::getline(std::cin, N_str);
    
    int N = 256;
    if (!N_str.empty()) {
        try {
            N = std::stoi(N_str);
            if (N < 1) {
                std::cerr << "[blast wave] Invalid resolution, using 256\n";
                N = 256; 
            }
        } catch (...) {
            std::cerr << "[blast wave] Invalid resolution, using 256\n";
            N = 256;
        }
    }
    std::cerr << "[blast wave] Resolution N² = " << N << "² (" << (long long) N * N << " cells)\n";

    const std::vector<double> t_outs = {0.0, 1.0, 2.5, 4.5, 6.5};
    const std::vector<std::string> t_tags = {"t000", "t100", "t250", "t450", "t650"};
    const double tfinal = t_outs.back();

    Grid G;
    G.Nx = N;
    G.Ny = N;
    G.ng = 2;
    G.x0 = x_min;
    G.y0 = y_min;
    G.Lx = x_max - x_min;
    G.Ly = y_max - y_min;
    G.dx = G.Lx / G.Nx;
    G.dy = G.Ly / G.Ny;
    G.U.assign(G.nx() * G.ny(), Cons{0,0,0,0,0,0,0,0,0});

    for (int j = 0; j < G.ny(); ++j) {
        for(int i = 0; i < G.nx(); ++i) {
            const double x = x_min + (i - G.ng + 0.5) * G.dx;
            const double y = y_min + (j - G.ng + 0.5) * G.dy;

            const double dx = x - blast_center_x;
            const double dy = y - blast_center_y;
            const double r = std::sqrt(dx * dx + dy * dy);

            Prim W{};
            W.rho = rho_ambient;
            W.vx = 0.0;
            W.vy = 0.0;
            W.vz = 0.0;
            W.p = (r <= blast_radius) ? p_inner : p_outer;
            W.Bx = Bx0;
            W.By = By0;
            W.Bz = 0.0;
            W.psi = 0.0;

            G.U[G.id(i, j)] = prim_to_cons(prim_with_floors(W));
        }
    }

    apply_reflective_bc(G);
    enforce_floors(G);
    
    const std::string base = "outputs/MHD/2D/Blast/Reflective/" + run_label;
    mkdir_p(base + "/data");
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");
    dump_fields(G, base, t_outs[0], t_tags[0]);

    double t = 0.0;
    int step = 0;
    std::vector<bool> wrote(t_outs.size(), false);
    wrote[0] = true;

    while (t < tfinal - 1e-14) {
        double ch = 0.0;
        double dt = compute_ch_and_dt(G, ch, use_glm);

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
            std::cerr << "[blast wave] invalid dt=" << dt
                      << " at t=" << t << " step=" << step << "\n";
            break;
        }

        bool step_ok = false;
        int retry = 0;
        constexpr int max_retries = 10;
        while (!step_ok && retry < max_retries) {
            step_ok = advance_rk3(G, dt, ch, use_glm, /*verbose_on_reject=*/retry >= 2);
            if (!step_ok) {
                dt *= 0.5;
                ++retry;
                if (!is_finite(dt) || dt <= 0.0) break;
            }
        }

        if (!step_ok) {
            std::cerr << "[blast wave] step=" << step
                      << " t=" << t << " FAILED after " << max_retries
                      << " dt-halving retries; aborting.\n";
            break;
        }

        if (retry > 0) {
            std::cerr << "[blast wave] step=" << step << " t=" << t
                      << " retried " << retry << "x, accepted dt=" << dt << "\n";
        }

        t += dt;
        ++step;

        for (std::size_t n = 1; n < t_outs.size(); ++n) {
            if (!wrote[n] && std::abs(t - t_outs[n]) < 1e-12) {
                dump_fields(G, base, t_outs[n], t_tags[n]);
                wrote[n] = true;
            }
        }

        if (step % 50 == 0 || t >= tfinal) {
            std::cerr << "[blast wave] step=" << step
                      << " t=" << t << " dt=" << dt << " ch=" << ch << "\n";
        }
    }

    std::cerr << "[blast wave] Wrote data to " << base << "/data\n";
    std::cerr << "[blast wave] Wrote plots to " << base << "/plots/png\n";
}