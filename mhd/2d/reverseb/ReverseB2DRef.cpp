// ============================================================================
// ReverseB2DRef.cpp - RP4 reverse-B MHD Riemann problem in a 2D rigid box.
// ----------------------------------------------------------------------------
//   - HLLD fluxes
//   - WENO3 reconstruction
//   - SSP-RK3 time stepping
//   - Optional Dedner-style GLM hyperbolic divergence cleaning
//
// Initial conditions:
//   L: ρ=1.0  u=0  v=0  w=0  Bx=1.3  By=1.0   Bz=0.0  p=1
//   R: ρ=0.4  u=0  v=0  w=0  Bx=1.3  By=-1.0  Bz=0.0  p=0
//   Discontinuity: x0=0.5  γ = 5/3  t_out=0.16
//   Box boundaries: x=[-0.75,0.75], y=[-0.5,0.5]
//   Rotation: theta degrees
// ============================================================================
#include "ReverseB2DRef.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

static inline double clamp_pos(double x, double eps = 1e-12) {
    return (x < eps) ? eps : x;
}

static inline bool is_finite(double x) {
    return std::isfinite(x);
}

// ----------------------------- rotation support -----------------------------
static double g_theta_rad = 0.0;
static double g_cos_theta = 1.0;
static double g_sin_theta = 0.0;

static void set_rotation_angle(double theta_degrees) {
    g_theta_rad = theta_degrees * M_PI / 180.0;
    g_cos_theta = std::cos(g_theta_rad);
    g_sin_theta = std::sin(g_theta_rad);
}

static inline void transform_to_computational_frame(double& x, double& y) {
    double x_comp = x * g_cos_theta - y * g_sin_theta;
    double y_comp = x * g_sin_theta + y * g_cos_theta;
    x = x_comp;
    y = y_comp;
}

// -------------------------------- parameters --------------------------------
static constexpr double gamma_gas = 5.0 / 3.0;
static constexpr double CFL = 0.60;
static constexpr double RP4_X0 = 0.01; 

static constexpr double BOX_X0 = -0.75;
static constexpr double BOX_X1 = 0.75;
static constexpr double BOX_Y0 = -0.50;
static constexpr double BOX_Y1 = 0.50;

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
    const double p = (gamma_gas - 1.0) * eint;
    return std::max(p, 1e-12);
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
    std::cout << "[reverse b-field] Enable hyperbolic divergence cleaning (GLM)? [y/n, default y]: ";
    std::string answer;
    std::getline(std::cin, answer);
    if (answer.empty()) return true;
    const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(answer[0])));
    return c != 'n';
}

// -------------------------------- GLM fluxes --------------------------------
static inline Cons flux_x(const Cons& U, double ch) {
    const Prim W = cons_to_prim(U);
    const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
    const double pt = W.p + 0.5 * B2;
    const double vdotB = W.vx * W.Bx + W.vy * W.By + W.vz * W.Bz;

    Cons F{};
    F.rho = U.mx;
    F.mx = U.mx * W.vx + pt - W.Bx * W.Bx;
    F.my = U.my * W.vx - W.Bx * W.By;
    F.mz = U.mz * W.vx - W.Bx * W.Bz;
    F.Bx = W.psi;
    F.By = W.By * W.vx - W.Bx * W.vy;
    F.Bz = W.Bz * W.vx - W.Bx * W.vz;
    F.E = (U.E + pt) * W.vx - W.Bx * vdotB;
    F.psi = ch * ch * W.Bx;
    return F;
}

static inline Cons flux_y(const Cons& U, double ch) {
    const Prim W = cons_to_prim(U);
    const double B2 = W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz;
    const double pt = W.p + 0.5 * B2;
    const double vdotB = W.vx * W.Bx + W.vy * W.By + W.vz * W.Bz;

    Cons G{};
    G.rho = U.my;
    G.mx = U.mx * W.vy - W.By * W.Bx;
    G.my = U.my * W.vy + pt - W.By * W.By;
    G.mz = U.mz * W.vy - W.By * W.Bz;
    G.Bx = W.Bx * W.vy - W.By * W.vx;
    G.By = W.psi;
    G.Bz = W.Bz * W.vy - W.By * W.vz;
    G.E = (U.E + pt) * W.vy - W.By * vdotB;
    G.psi = ch * ch * W.By;
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
    const Cons FL = flux_x(UL, ch);
    const Cons FR = flux_x(UR, ch);

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
    const Cons GL = flux_y(UL, ch);
    const Cons GR = flux_y(UR, ch);

    if (SL >= 0.0) return GL;
    if (SR <= 0.0) return GR;

    const double denom = SR - SL;
    const Cons num = c_add(c_sub(c_mul(SR, GL), c_mul(SL, GR)),
                           c_mul(SL * SR, c_sub(UR, UL)));
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
    const Cons FL = flux_x(UL, ch);
    const Cons FR = flux_x(UR, ch);

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

    const auto star_state = [&](const Prim& W, const Cons& U, double S, double pt_side) {
        Cons Us{};
        const double rho = clamp_pos(W.rho);
        const double rho_star = rho * (S - W.vx) / (S - SM);
        const double D = rho * (S - W.vx) * (S - SM) - Bx_face * Bx_face;
        if (!std::isfinite(rho_star) || rho_star <= 0.0 || std::abs(D) < 1e-12) {
            return std::pair<bool, Cons>{false, {}};
        }

        const double factor = (rho * (S - W.vx) * (S - W.vx) - Bx_face * Bx_face) / D;
        const double vy_star = W.vy - (Bx_face * W.By * (SM - W.vx)) / D;
        const double vz_star = W.vz - (Bx_face * W.Bz * (SM - W.vx)) / D;
        const double By_star = W.By * factor;
        const double Bz_star = W.Bz * factor;

        const double vdotB = W.vx * Bx_face + W.vy * W.By + W.vz * W.Bz;
        const double vdotB_star = SM * Bx_face + vy_star * By_star + vz_star * Bz_star;
        const double E_star = ((S - W.vx) * U.E - pt_side * W.vx + pt_star * SM +
                               Bx_face * (vdotB - vdotB_star)) / (S - SM);

        Us.rho = rho_star;
        Us.mx = rho_star * SM;
        Us.my = rho_star * vy_star;
        Us.mz = rho_star * vz_star;
        Us.Bx = Bx_face;
        Us.By = By_star;
        Us.Bz = Bz_star;
        Us.E = E_star;
        Us.psi = psi_face;
        return std::pair<bool, Cons>{true, Us};
    };

    const auto [okL, UstL] = star_state(WL, UL, SL, ptL);
    const auto [okR, UstR] = star_state(WR, UR, SR, ptR);
    if (!(okL && okR)) return hll_x(UL, UR, ch, use_glm);

    const double rho_star_L = clamp_pos(UstL.rho);
    const double rho_star_R = clamp_pos(UstR.rho);
    const double sqrt_rho_L = std::sqrt(rho_star_L);
    const double sqrt_rho_R = std::sqrt(rho_star_R);
    const double denomA = sqrt_rho_L + sqrt_rho_R;
    if (denomA < 1e-12) return hll_x(UL, UR, ch, use_glm);

    const double signBx = (Bx_face >= 0.0) ? 1.0 : -1.0;
    const double SstL = SM - std::abs(Bx_face) / std::sqrt(rho_star_L);
    const double SstR = SM + std::abs(Bx_face) / std::sqrt(rho_star_R);

    const double vy_ss =
        (sqrt_rho_L * (UstL.my / rho_star_L) + sqrt_rho_R * (UstR.my / rho_star_R) + signBx
         * (UstR.By - UstL.By)) / denomA;
    const double vz_ss =
        (sqrt_rho_L * (UstL.mz / rho_star_L) + sqrt_rho_R * (UstR.mz / rho_star_R) + signBx
         * (UstR.Bz - UstL.Bz)) / denomA;
    const double By_ss =
        (sqrt_rho_L * UstL.By + sqrt_rho_R * UstR.By + signBx * std::sqrt(rho_star_L * rho_star_R)
         * ((UstR.my / rho_star_R) - (UstL.my / rho_star_L))) / denomA;
    const double Bz_ss =
        (sqrt_rho_L * UstL.Bz + sqrt_rho_R * UstR.Bz + signBx * std::sqrt(rho_star_L * rho_star_R)
         * ((UstR.mz / rho_star_R) - (UstL.mz / rho_star_L))) / denomA;
         
    Cons UssL = UstL;
    Cons UssR = UstR;
    UssL.my = rho_star_L * vy_ss;
    UssL.mz = rho_star_L * vz_ss;
    UssL.By = By_ss;
    UssL.Bz = Bz_ss;
    UssR.my = rho_star_R * vy_ss;
    UssR.mz = rho_star_R * vz_ss;
    UssR.By = By_ss;
    UssR.Bz = Bz_ss;

    const double vtBtL = (UstL.my / rho_star_L) * UstL.By + (UstL.mz / rho_star_L) * UstL.Bz;
    const double vtBtR = (UstR.my / rho_star_R) * UstR.By + (UstR.mz / rho_star_R) * UstR.Bz;
    const double vtBtSS = vy_ss * By_ss + vz_ss * Bz_ss;
    UssL.E = UstL.E - sqrt_rho_L * (vtBtL - vtBtSS) * signBx;
    UssR.E = UstR.E + sqrt_rho_R * (vtBtR - vtBtSS) * signBx;

    Cons FstL = c_add(FL, c_mul(SL, c_sub(UstL, UL)));
    Cons FstR = c_add(FR, c_mul(SR, c_sub(UstR, UR)));
    Cons FssL = c_add(FstL, c_mul(SstL, c_sub(UssL, UstL)));
    Cons FssR = c_add(FstR, c_mul(SstR, c_sub(UssR, UstR)));

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
    const double eps = 1e-6;
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
    const double eps = 1e-6;
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
    W.psi = weno3_left_scalar(Wm1.psi, W0.psi, Wp1.psi);
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
    W.psi = weno3_right_scalar(W0.psi, Wp1.psi, Wp2.psi);
    return prim_with_floors(W);
}

// -------------------------------- grid + BC ---------------------------------
struct Grid {
    int Nx = 0, Ny = 0, ng = 2;
    double x0 = 0.0, y0 = 0.0, Lx = 0.0, Ly = 0.0, dx = 0.0, dy = 0.0;
    std::vector<Cons> U;

    int ibl = 0, ibh = 0, jbl = 0, jbh = 0;
    int nx() const { return Nx + 2 * ng; }
    int ny() const { return Ny + 2 * ng; }
    inline int id(int i, int j) const { return j * nx() + i; }
    inline bool inside(int i, int j) const { return i >= ibl && i <= ibh && j >= jbl && j <= jbh; }
};

static inline Prim bilinear_interp(const Prim& W00, const Prim& W10,
                                    const Prim& W01, const Prim& W11, 
                                    double tx, double ty) {

    tx = std::max(0.0, std::min(1.0, tx));
    ty = std::max(0.0, std::min(1.0, ty));

    Prim W{};

    auto interp = [&](double f00, double f10, double f01, double f11) {
        double val = (1.0 - tx) * (1.0 - ty) * f00 + tx * (1.0 - ty) * f10
                     + (1.0 - tx) * ty * f01 + tx * ty * f11;
        if (!std::isfinite(val)) return 0.0;
        return val;
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

    W = prim_with_floors(W);

    const double LIM = 1e3;
    W.vx = std::max(-LIM, std::min(W.vx, LIM));
    W.vy = std::max(-LIM, std::min(W.vy, LIM));
    W.vz = std::max(-LIM, std::min(W.vz, LIM));
    W.Bx = std::max(-LIM, std::min(W.Bx, LIM));
    W.By = std::max(-LIM, std::min(W.By, LIM));
    W.Bz = std::max(-LIM, std::min(W.Bz, LIM));
    W.psi = std::max(-LIM, std::min(W.psi, LIM));

    if (!std::isfinite(W.rho) || !std::isfinite(W.p)) {
        return prim_with_floors(W00);
    }

    return W;
}

static inline double phi_box(double x_ref, double y_ref) {
    const double dx = std::max({BOX_X0 - x_ref, 0.0, x_ref - BOX_X1});
    const double dy = std::max({BOX_Y0 - y_ref, 0.0, y_ref - BOX_Y1});
    const double outside = std::sqrt(dx*dx + dy*dy);

    const double inside = -std::min({
        x_ref - BOX_X0,
        BOX_X1 - x_ref,
        y_ref - BOX_Y0,
        BOX_Y1 - y_ref
    });

    return (outside > 0.0) ? outside : inside;
}

static Prim sample_prim_bilinear(const Grid& G, double x_comp, double y_comp) {
    x_comp = std::max(G.x0, std::min(x_comp, G.x0 + G.Lx));
    y_comp = std::max(G.y0, std::min(y_comp, G.y0 + G.Ly));

    const double i_float = (x_comp - G.x0) / G.dx + G.ng - 0.5;
    const double j_float = (y_comp - G.y0) / G.dy + G.ng - 0.5;

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
    double x = G.x0 + (i - G.ng + 0.5) * G.dx;
    double y = G.y0 + (j - G.ng + 0.5) * G.dy;
    const double phi = phi_box(x, y);
    return phi <= 0.0;
}

static void apply_reflective_bc(Grid& G) {
    const int NX = G.nx(), NY = G.ny();

    for (int j = 0; j < NY; ++j) {
        for (int i = 0; i < NX; ++i) {
            double x = G.x0 + (i - G.ng + 0.5) * G.dx;
            double y = G.y0 + (j - G.ng + 0.5) * G.dy;

            const double phi = phi_box(x, y);
            if (phi <= 0.0) continue;

            double x_reflected = x;
            double y_reflected = y;
            bool reflect_vx = false, reflect_vy = false;

            if (x < BOX_X0) {
                x_reflected = 2.0 * BOX_X0 - x;
                reflect_vx = true;
            } else if (x > BOX_X1) {
                x_reflected = 2.0 * BOX_X1 - x;
                reflect_vx = true;
            }

            if (y < BOX_Y0) {
                y_reflected = 2.0 * BOX_Y0 - y;
                reflect_vy = true;
            } else if (y > BOX_Y1) {
                y_reflected = 2.0 * BOX_Y1 - y;
                reflect_vy = true;
            }

            bool use_extrapolation = false;
            if (x_reflected < G.x0 || x_reflected > G.x0 + G.Lx ||
                y_reflected < G.y0 || y_reflected > G.y0 + G.Ly) {
                use_extrapolation = true;
            }

            Prim W_interp;

            if (!use_extrapolation) {
                W_interp = sample_prim_bilinear(G, x_reflected, y_reflected);
            } else {
                double x_boundary = x_reflected;
                double y_boundary = y_reflected;

                x_boundary = std::max(G.x0, std::min(x_boundary, G.x0 + G.Lx));
                y_boundary = std::max(G.y0, std::min(y_boundary, G.y0 + G.Ly));

                const double i_float_b = (x_boundary - G.x0) / G.dx + G.ng - 0.5;
                const double j_float_b = (y_boundary - G.y0) / G.dy + G.ng - 0.5;

                int i_b = (int)std::round(i_float_b);
                int j_b = (int)std::round(j_float_b);

                i_b = std::max(G.ibl, std::min(i_b, G.ibh));
                j_b = std::max(G.jbl, std::min(j_b, G.jbh));

                const Prim W_boundary = cons_to_prim(G.U[G.id(i_b, j_b)]);
                W_interp = W_boundary;
                W_interp.vx = 0.0;
                W_interp.vy = 0.0;
                W_interp.vz = 0.0;
                W_interp.psi = 0.0;
            }

            if (reflect_vx) W_interp.vx = -W_interp.vx;
            if (reflect_vy) W_interp.vy = -W_interp.vy;

            double nx = 0.0, ny = 0.0;

            if (x < BOX_X0) { nx = -1.0; ny = 0.0; }
            else if (x > BOX_X1) { nx = 1.0; ny = 0.0; }
            else if (y < BOX_Y0) { nx = 0.0; ny = -1.0; }
            else if (y > BOX_Y1) { nx = 0.0; ny = 1.0; }

            const double dx_in = x_reflected - x;
            const double dy_in = y_reflected - y;
            Prim W1 = sample_prim_bilinear(G, x_reflected, y_reflected);
            Prim W2 = sample_prim_bilinear(G, x_reflected + dx_in, y_reflected + dy_in);

            double Bx1, By1, Bz1;
            double Bx2, By2, Bz2;

            Bx1 = W1.Bx; By1 = W1.By; Bz1 = W1.Bz;
            Bx2 = W2.Bx; By2 = W2.By; Bz2 = W2.Bz;

            double Bn1 = Bx1 * nx + By1 * ny;
            double Bx_t = Bx1 - Bn1 * nx;
            double By_t = By1 - Bn1 * ny;
            double Bx_t2 = Bx2 - (Bx2 * nx + By2 * ny) * nx;
            double By_t2 = By2 - (Bx2 * nx + By2 * ny) * ny;
            double Bx_tg = 2.0 * Bx_t - Bx_t2;
            double By_tg = 2.0 * By_t - By_t2;
            double Bx_ref = Bn1 * nx + Bx_tg;
            double By_ref = Bn1 * ny + By_tg;
            double Bz_ref = 2.0 * Bz1 - Bz2;

            W_interp.Bx = Bx_ref;
            W_interp.By = By_ref;
            W_interp.Bz = Bz_ref;
            W_interp.vz = -W1.vz;
            W_interp.psi = -W_interp.psi;

            G.U[G.id(i, j)] = prim_to_cons(W_interp);
        }
    }
}

static void enforce_floors(Grid& G) {
    constexpr double HL = 1e6;
    for(int j = 0; j < G.ny(); ++j) {
        for(int i = 0; i < G.nx(); ++i) {
            Prim W = cons_to_prim(G.U[G.id(i, j)]);
            if(!is_finite(W.rho) || !is_finite(W.p) || !is_finite(W.vx) ||
               !is_finite(W.vy) || !is_finite(W.vz)|| !is_finite(W.Bx) ||
               !is_finite(W.By) || !is_finite(W.Bz)|| !is_finite(W.psi) ||
               std::abs(W.vx) > HL || std::abs(W.vy) > HL || std::abs(W.vz) > HL ||
               std::abs(W.Bx) > HL || std::abs(W.By) > HL || std::abs(W.Bz) > HL ||
               std::abs(W.psi) > HL) {
                if(!is_finite(W.rho) || W.rho < 0.0) W.rho = 1e-6;
                if(!is_finite(W.p) || W.p < 0.0) W.p = 1e-6;
                if(!is_finite(W.vx) || std::abs(W.vx) > HL) W.vx = 0.0;
                if(!is_finite(W.vy) || std::abs(W.vy) > HL) W.vy = 0.0;
                if(!is_finite(W.vz) || std::abs(W.vz) > HL) W.vz = 0.0;
                if(!is_finite(W.Bx) || std::abs(W.Bx) > HL) W.Bx = 0.0;
                if(!is_finite(W.By) || std::abs(W.By) > HL) W.By = 0.0;
                if(!is_finite(W.Bz) || std::abs(W.Bz) > HL) W.Bz = 0.0;
                if(!is_finite(W.psi) || std::abs(W.psi) > HL) W.psi = 0.0;
            }
            W = prim_with_floors(W);
            G.U[G.id(i, j)] = prim_to_cons(W);
        }
    }
}

// -------------------------------- time step ---------------------------------
static double compute_ch_and_dt(const Grid& G, double& ch_out, bool use_glm) {
    static double prev_dt = 1e-4;
    static double prev_ch = 1.0;

    double amax = 1e-12;
    for (int j = G.ng; j < G.ny() - G.ng; ++j) {
        for (int i = G.ng; i < G.nx() - G.ng; ++i) {
            const Prim W = cons_to_prim(G.U[G.id(i, j)]);

            if (!is_finite(W.rho) || !is_finite(W.p)) {
                continue;
            }

            const double ax = std::abs(W.vx) + fast_speed_dir(W, W.Bx);
            const double ay = std::abs(W.vy) + fast_speed_dir(W, W.By);
            const double local = std::max(ax, ay);
            
            if (is_finite(local)) {
                amax = std::max(amax, local);
            }
        }
    }

    amax = std::max(1e-6, std::min(amax, 10.0));
    ch_out = use_glm ? std::max(0.2, std::min(amax, 3.0)) : 0.0;
    const double dt_raw = CFL * std::min(G.dx, G.dy) / amax;

    double dt = dt_raw;
    if (!is_finite(dt) || dt < 1e-9) {
        dt = prev_dt * 0.8;
    } else if (dt > 2.0 * prev_dt) {
        dt = 1.2 * prev_dt;
    }

    prev_dt = dt;
    prev_ch = ch_out;
    return dt;
}

// ----------------------------------- RHS ------------------------------------
static std::vector<Cons> rhs(const Grid& G, double ch, bool use_glm) {
    const int NX = G.nx(), NY = G.ny();

    std::vector<Prim> W(NX * NY);
    for(int j = 0; j < NY; ++j) {
        for(int i = 0; i < NX; ++i) {
            W[G.id(i, j)] = cons_to_prim(G.U[G.id(i, j)]);
        }
    }

    std::vector<Cons> Fx(NX * NY, Cons{0,0,0,0,0,0,0,0,0});
    for(int j = G.ng; j < NY - G.ng; ++j) {
        for(int i = G.ng - 1; i < NX - G.ng; ++i) {
            bool cell_i_interior = is_interior_cell(i, j, G);
            bool cell_ip1_interior = is_interior_cell(i + 1, j, G);
            if (!cell_i_interior && !cell_ip1_interior) continue;
            bool safe_stencil =
                is_interior_cell(i - 1, j, G) &&
                is_interior_cell(i, j, G) &&
                is_interior_cell(i + 1, j, G) &&
                is_interior_cell(i + 2, j, G);
            Prim WL, WR;
            if (safe_stencil) {
                WL = reconstruct_left_weno3(W[G.id(i - 1, j)], W[G.id(i, j)], W[G.id(i + 1, j)]);
                WR = reconstruct_right_weno3(W[G.id(i, j)], W[G.id(i + 1, j)], W[G.id(i + 2, j)]);
            } else {
                WL = W[G.id(i, j)];
                WR = W[G.id(i + 1, j)];
            }
            WL = prim_with_floors(WL);
            WR = prim_with_floors(WR);
            Fx[G.id(i, j)] = hlld_x(prim_to_cons(WL), prim_to_cons(WR), ch, use_glm);
        }
    }

    std::vector<Cons> Gy(NX * NY, Cons{0,0,0,0,0,0,0,0,0});
    for(int j = G.ng - 1; j < NY - G.ng; ++j) {
        for(int i = G.ng; i < NX - G.ng; ++i) {
            bool cell_j_interior   = is_interior_cell(i, j, G);
            bool cell_jp1_interior = is_interior_cell(i, j + 1, G);
            if (!cell_j_interior && !cell_jp1_interior) continue;
            bool safe_stencil =
                is_interior_cell(i, j - 1, G) &&
                is_interior_cell(i, j, G) &&
                is_interior_cell(i, j + 1, G) &&
                is_interior_cell(i, j + 2, G);
            Prim WL, WR;
            if (safe_stencil) {
                WL = reconstruct_left_weno3(W[G.id(i, j - 1)], W[G.id(i, j)], W[G.id(i, j + 1)]);
                WR = reconstruct_right_weno3( W[G.id(i, j)], W[G.id(i, j + 1)], W[G.id(i, j + 2)]);
            } else {
                WL = W[G.id(i, j)];
                WR = W[G.id(i, j + 1)];
            }
            WL = prim_with_floors(WL);
            WR = prim_with_floors(WR);
            Gy[G.id(i, j)] = hlld_y( prim_to_cons(WL), prim_to_cons(WR), ch, use_glm );
        }
    }
    
    std::vector<Cons> LU(NX * NY, Cons{0,0,0,0,0,0,0,0,0});
    for(int j = G.ng; j < NY - G.ng; ++j) {
        for(int i = G.ng; i < NX - G.ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            const Cons dFx = c_sub(Fx[G.id(i, j)], Fx[G.id(i - 1, j)]);
            const Cons dGy = c_sub(Gy[G.id(i, j)], Gy[G.id(i, j - 1)]);
            Cons L = c_add(c_mul(-1.0 / G.dx, dFx), c_mul(-1.0 / G.dy, dGy));
            if(use_glm) {
                const double alpha = 0.1 * ch;
                L.psi += -alpha * G.U[G.id(i, j)].psi;
            }
            LU[G.id(i, j)] = L;
        }
    }
    return LU;
}

// ----------------------------------- RK3 ------------------------------------
static void advance_rk3(Grid& G, double dt, double ch, bool use_glm) {
    const int NX = G.nx(), NY = G.ny();
    apply_reflective_bc(G);
    enforce_floors(G);
    const auto L1 = rhs(G, ch, use_glm);
    Grid G1 = G;
    for(int j = G.ng; j < NY - G.ng; ++j) {
        for(int i = G.ng; i < NX - G.ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            G1.U[G.id(i, j)] = c_add(G.U[G.id(i, j)], c_mul(dt, L1[G.id(i, j)]));
        }
    }
    apply_reflective_bc(G1); 
    enforce_floors(G1);
    const auto L2 = rhs(G1, ch, use_glm);
    Grid G2 = G;
    for(int j = G.ng; j < NY - G.ng; ++j) {
        for(int i = G.ng; i < NX - G.ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            G2.U[G.id(i, j)] = c_add(c_mul(0.75, G.U[G.id(i, j)]),
                                      c_mul(0.25, c_add(G1.U[G.id(i, j)], c_mul(dt, L2[G.id(i, j)]))));
        }
    }
    apply_reflective_bc(G2);
    enforce_floors(G2);
    const auto L3 = rhs(G2, ch, use_glm);
    for(int j = G.ng; j < NY - G.ng; ++j) {
        for(int i = G.ng; i < NX - G.ng; ++i) {
            if (!is_interior_cell(i, j, G)) continue;
            G.U[G.id(i, j)] = c_add(c_mul(1.0/3.0, G.U[G.id(i, j)]),
                                     c_mul(2.0/3.0, c_add(G2.U[G.id(i, j)], c_mul(dt, L3[G.id(i, j)]))));
        }
    }
    apply_reflective_bc(G);
    enforce_floors(G);
}

// --------------------------------- plotting ---------------------------------
static void write_field_xy(const std::string& path, const Grid& G,
                           const std::vector<double>& val, const std::string& hdr) {
    std::ofstream f(path); f<<"# "<<hdr<<"\n"<<std::setprecision(14);
    const int ng = G.ng;
    for(int j = ng; j < G.ny() - ng; ++j) {
        const double y_ref = G.y0 + (j - ng + 0.5) * G.dy;
        for(int i = ng; i < G.nx() - ng; ++i) {
            const double x_ref = G.x0 + (i - ng + 0.5) * G.dx;
            double x = x_ref;
            double y = y_ref;
            transform_to_computational_frame(x, y);
            f<<x<<" "<<y<<" "<<val[G.id(i, j)]<<"\n";
        }
        f<<"\n";
    }
}

static void write_cut_plot(const std::string& path, const Grid& G) {
    const int n_samples = 256;
    std::ofstream f(path);
    f << "# s E rho p Bz dBx/dn v_z (shock-aligned cut at y_ref=0)\n"
      << std::setprecision(14);
    
    std::vector<double> Bx_vals(n_samples);
    std::vector<double> x_vals(n_samples);
    std::vector<double> E_vals(n_samples);
    std::vector<double> rho_vals(n_samples);
    std::vector<double> p_vals(n_samples);
    std::vector<double> Bz_vals(n_samples);
    std::vector<double> vz_vals(n_samples);
    
    for (int k = 0; k < n_samples; ++k) {
        double x_ref = BOX_X0 + (k + 0.5) * (BOX_X1 - BOX_X0) / n_samples;
        double y_ref = 0.0;
        
        const Prim W = sample_prim_bilinear(G, x_ref, y_ref);
        const Cons U = prim_to_cons(W);
        
        x_vals[k] = x_ref;
        Bx_vals[k] = W.Bx;
        E_vals[k] = U.E;
        rho_vals[k] = W.rho;
        p_vals[k] = W.p;
        Bz_vals[k] = W.Bz;
        vz_vals[k] = W.vz;
    }
    
    const double ds = (BOX_X1 - BOX_X0) / n_samples;
    for (int k = 0; k < n_samples; ++k) {
        double dBx_dn = 0.0;
        if (k == 0) {
            dBx_dn = (Bx_vals[1] - Bx_vals[0]) / ds;
        } else if (k == n_samples - 1) {
            dBx_dn = (Bx_vals[n_samples-1] - Bx_vals[n_samples-2]) / ds;
        } else {
            dBx_dn = (Bx_vals[k+1] - Bx_vals[k-1]) / (2.0 * ds);
        }
        
        f << x_vals[k] << " "
          << E_vals[k] << " "
          << rho_vals[k] << " "
          << p_vals[k] << " "
          << Bz_vals[k] << " "
          << dBx_dn << " "
          << vz_vals[k] << "\n";
    }
    f.close();
}

static void make_colour_plot(const std::string& base,
                             const std::string& datname,
                             const std::string& pngname,
                             const std::string& title,
                             const std::string& cblabel,
                             const Grid& G, double tout,
                             double theta_deg = 0.0,
                             bool use_full_domain = true) {
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");
    const std::string plt =
        base + "/plots/plt/" + pngname.substr(0, pngname.size() - 4) + ".plt";
    std::ofstream gp(plt);
    const double xlo = use_full_domain ? G.x0 : BOX_X0;
    const double xhi = use_full_domain ? (G.x0 + G.Lx) : BOX_X1;
    const double ylo = use_full_domain ? G.y0 : BOX_Y0;
    const double yhi = use_full_domain ? (G.y0 + G.Ly) : BOX_Y1;
    gp << "set term pngcairo size 700,700\n";
    gp << "set output '" << base << "/plots/png/" << pngname << "'\n";
    gp << "set size ratio -1\n";
    gp << "unset key\n";
    gp << "set xlabel 'x'\n";
    gp << "set ylabel 'y'\n";
    gp << "set title '" << title << " (θ = " << std::fixed << std::setprecision(1) 
       << theta_deg << "°)'\n";
    gp << "set cblabel '" << cblabel << "'\n";
    gp << "set pm3d map\n";
    gp << "set colorbox\n";
    gp << "set palette viridis\n";
    gp << "set border linewidth 1.4\n";
    gp << "set tics scale 0.7\n";
    gp << "set xrange [" << xlo << ":" << xhi << "]\n";
    gp << "set yrange [" << ylo << ":" << yhi << "]\n";
    gp << std::setprecision(14);
    gp << "cos_theta = " << g_cos_theta << "\n";
    gp << "sin_theta = " << g_sin_theta << "\n";
    gp << "box_x0 = " << BOX_X0 << "\n";
    gp << "box_x1 = " << BOX_X1 << "\n";
    gp << "box_y0 = " << BOX_Y0 << "\n";
    gp << "box_y1 = " << BOX_Y1 << "\n";
    gp << "splot '" << base << "/data/" << datname << "' u 1:2:"
        << "(x_ref=$1*cos_theta+$2*sin_theta, y_ref=-$1*sin_theta+$2*cos_theta, "
        << "(x_ref>=box_x0&&x_ref<=box_x1&&y_ref>=box_y0&&y_ref<=box_y1?$3:1/0)) w pm3d\n";
    gp.close();
    if (cmd_exists("gnuplot")) {
        syscmd("gnuplot \"" + plt + "\" >/dev/null 2>&1");
    }
}

static void make_cut_and_plot(const std::string& base,
                              const std::string& tag,
                              double tout, const Grid& G,
                              double theta_deg = 0.0) {
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");

    const std::string dat = base + "/data/ReverseBRef_Multiplot_" + tag + ".dat";
    const std::string plt = base + "/plots/plt/ReverseBRef_Multiplot_" + tag + ".plt";
    const std::string png = base + "/plots/png/ReverseBRef_Multiplot_" + tag + ".png";

    std::ofstream gp(plt);
    gp << "set term pngcairo size 1200,1400 enhanced\n";
    gp << "set output '" << png << "'\n";
    gp << "set multiplot layout 3,2 rowsfirst "
       << "title 'Reverse Magnetic Field Shock Tube at t = " << std::fixed << std::setprecision(6) << tout
       << "s (θ = " << std::fixed << std::setprecision(1) << theta_deg << "°)' font ',16'\n";
    gp << "unset key\n";
    gp << "set border linewidth 1.2\n";
    gp << "set tics scale 0.6\n";
    gp << "set format y '%.3g'\n";
    gp << "set xrange [" << BOX_X0 << ":" << BOX_X1 << "]\n";
    gp << "set xlabel 's (shock-normal direction)'\n";
    gp << "set pointsize 0.35\n";
    gp << "set grid\n";
    gp << "set style line 1 lc rgb '#000000' pt 7 ps 0.25 lw 1\n";
    gp << "set title 'Energy'\n";
    gp << "set ylabel 'E'\n";
    gp << "set autoscale y\n";
    gp << "plot '" << dat << "' u 1:2 w p ls 1 notitle\n";
    gp << "set title 'Density'\n";
    gp << "set ylabel '{/Symbol r}'\n";
    gp << "set autoscale y\n";
    gp << "plot '" << dat << "' u 1:3 w p ls 1 notitle\n";
    gp << "set title 'Pressure'\n";
    gp << "set ylabel 'p'\n";
    gp << "set autoscale y\n";
    gp << "plot '" << dat << "' u 1:4 w p ls 1 notitle\n";
    gp << "set title 'Magnetic Field in z-direction'\n";
    gp << "set ylabel 'B_z'\n";
    gp << "set autoscale y\n";
    gp << "plot '" << dat << "' u 1:5 w p ls 1 notitle\n";
    gp << "set title 'Magnetic Field Divergence'\n";
    gp << "set ylabel 'div(B_n)'\n";
    gp << "set autoscale y\n";
    gp << "plot '" << dat << "' u 1:6 w p ls 1 notitle\n";
    gp << "set title 'Velocity in z-direction'\n";
    gp << "set ylabel 'v_z'\n";
    gp << "set autoscale y\n";
    gp << "plot '" << dat << "' u 1:7 w p ls 1 notitle\n";
    gp << "unset multiplot\n";
    gp.close();

    if (cmd_exists("gnuplot")) {
        syscmd("gnuplot \"" + plt + "\" >/dev/null 2>&1");
    }
}

static void dump_snapshot(const Grid& G, const std::string& base,
                          const std::string& tag, double tout, double theta_deg = 0.0) {
    const int NX = G.nx(), NY = G.ny();
    std::vector<double> rho(NX * NY), p(NX * NY), Bz(NX * NY), E(NX * NY), divB(NX * NY);
    for(int j = G.ng; j < G.ny() - G.ng; ++j) {
        for(int i = G.ng; i < G.nx() - G.ng; ++i) {
            const Cons& U = G.U[G.id(i, j)];
            const Prim W = cons_to_prim(U);
            rho[G.id(i, j)] = W.rho;
            p[G.id(i, j)] = W.p;
            Bz[G.id(i, j)] = W.Bz;
            E[G.id(i, j)] = U.E;
        }
    }
    for(int j = G.ng; j < G.ny() - G.ng; ++j) {
        for(int i = G.ng; i < G.nx() - G.ng; ++i) {
            const Prim Wxp = cons_to_prim(G.U[G.id(i + 1, j)]);
            const Prim Wxm = cons_to_prim(G.U[G.id(i - 1, j)]);
            const Prim Wyp = cons_to_prim(G.U[G.id(i, j + 1)]);
            const Prim Wym = cons_to_prim(G.U[G.id(i, j - 1)]);
            divB[G.id(i, j)] = (Wxp.Bx - Wxm.Bx) / (2.0 * G.dx)
                             + (Wyp.By - Wym.By) / (2.0 * G.dy);
        }
    }
    mkdir_p(base + "/data");
    const std::string ts = std::to_string(tout);
    write_field_xy(base + "/data/rho_" + tag + ".dat", G, rho, "x y rho");
    write_field_xy(base + "/data/p_" + tag + ".dat", G, p, "x y p");
    write_field_xy(base + "/data/Bz_" + tag + ".dat", G, Bz, "x y Bz");
    write_field_xy(base + "/data/E_" + tag + ".dat", G, E, "x y E");
    write_field_xy(base + "/data/divB_" + tag + ".dat", G, divB, "x y divB");
    write_cut_plot(base + "/data/ReverseBRef_Multiplot_" + tag + ".dat", G);
    make_colour_plot(base, "E_" + tag + ".dat", "E_" + tag + ".png",
                    "Reverse-B RP4 E at t = " + ts + "s", "E", G, tout, theta_deg, true);
    make_colour_plot(base, "rho_" + tag + ".dat", "rho_" + tag + ".png",
                    "Reverse-B RP4 {/Symbol r} at t = " + ts + "s","{/Symbol r}",G, tout, theta_deg, true);
    make_colour_plot(base, "Bz_" + tag + ".dat", "Bz_" + tag + ".png",
                    "Reverse-B RP4 B_z at t = " + ts + "s", "B_z", G, tout, theta_deg, true);
    make_colour_plot(base, "divB_" + tag + ".dat", "divB_" + tag + ".png",
                    "Reverse-B RP4 ∇·B at t = " + ts + "s", "∇·B", G, tout, theta_deg, true);
    make_cut_and_plot(base, tag, tout, G, theta_deg);
}

// ---------------------------------- driver ----------------------------------
void run_reverse_b_ref() {
    std::cout << "[reverse b-field] Enter rotation angle theta (degrees) [default 0]: ";
    std::string theta_str;
    std::getline(std::cin, theta_str);
    double theta_deg = 0.0;
    if (!theta_str.empty()) {
        try {
            theta_deg = std::stod(theta_str);
        } catch (...) {
            std::cerr << "[reverse b-field] Invalid angle, using 0\n";
            theta_deg = 0.0;
        }
    }
    set_rotation_angle(theta_deg);
    std::cerr << "[reverse b-field] Rotation angle theta = " << theta_deg << " degrees\n";

    const bool use_glm = prompt_use_glm();

    const int Nx = 256;
    const int Ny = 256;
    const double DomX0 = -1.0;
    const double DomY0 = -1.0;
    const double DomLx = 2.0;
    const double DomLy = 2.0;
    const double dx = DomLx / Nx;
    const double dy = DomLy / Ny;

    Grid G;
    G.Nx = Nx;
    G.Ny = Ny;
    G.ng = 2;
    G.x0 = DomX0;
    G.y0 = DomY0;
    G.Lx = DomLx;
    G.Ly = DomLy;
    G.dx = dx;
    G.dy = dy;
    G.U.assign(G.nx() * G.ny(), Cons{0,0,0,0,0,0,0,0,0});

    G.ibl = G.ng + (int)std::round((BOX_X0 - DomX0) / dx);
    G.ibh = G.ng + (int)std::round((BOX_X1 - DomX0) / dx) - 1;
    G.jbl = G.ng + (int)std::round((BOX_Y0 - DomY0) / dy);
    G.jbh = G.ng + (int)std::round((BOX_Y1 - DomY0) / dy) - 1;

    std::cerr<< "[reverse b-field] box cells x: [" << G.ibl << "," << G.ibh << "] ("
             << G.ibh - G.ibl + 1 << " cells)\n";
    std::cerr<< "[reverse b-field] box cells y: [" << G.jbl << "," << G.jbh << "] ("
             << G.jbh - G.jbl + 1 << " cells)\n";

    Prim WL{};
    WL.rho = 1.0;
    WL.p = 1.0;
    WL.Bx = 1.3;
    WL.By = 0.0;
    WL.Bz = 1.0;

    Prim WR{};
    WR.rho = 0.4;
    WR.p = 0.4;
    WR.Bx = 1.3;
    WR.By = 0.0;
    WR.Bz = -1.0;

    for(int j = 0; j < G.ny(); ++j) {
        for(int i = 0; i < G.nx(); ++i) {
            double x = G.x0 + (i - G.ng + 0.5) * G.dx;
            const double eps = 1e-12, dx_ref = x - RP4_X0;
            if (dx_ref < -eps) {
                G.U[G.id(i, j)] = prim_to_cons(WL);
            }
            else if (dx_ref > eps) {
                G.U[G.id(i, j)] = prim_to_cons(WR);
            }
            else {
                Prim Wmid{};
                Wmid.rho = 0.5 * (WL.rho + WR.rho);
                Wmid.p = 0.5 * (WL.p + WR.p);
                Wmid.Bx = 0.5 * (WL.Bx + WR.Bx);
                Wmid.Bz = 0.5 * (WL.Bz + WR.Bz);
                G.U[G.id(i, j)] = prim_to_cons(Wmid);
            }
        }
    }

    apply_reflective_bc(G);
    enforce_floors(G);

    const std::string base = "outputs/MHD/2D/ReverseB/Reflective";
    mkdir_p(base + "/data");
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");
    dump_snapshot(G, base, "t0.000", 0.0, theta_deg);

    const std::vector<double> t_outs = {0.0, 0.16, 0.32, 0.48, 0.53};
    std::vector<std::string> t_tags = {"t0.000", "t0.160", "t0.320",
                                       "t0.480", "t0.530"};
    std::vector<bool> wrote(t_outs.size(), false);

    double t = 0.0;
    int step = 0;
    const double tfinal = t_outs.back();
    while(t < tfinal) {
        double ch = 0.0;
        double dt = compute_ch_and_dt(G, ch, use_glm);
        for(std::size_t k = 0; k < t_outs.size(); ++k) {
            if(!wrote[k] && t < t_outs[k] && t + dt > t_outs[k]) {
                dt = t_outs[k] - t;
            }
        }
        if(t + dt > tfinal) {
            dt = tfinal - t;
        }
        if(!is_finite(dt) || dt <= 0.0) {
            std::cerr<<"[reverse b-field] invalid dt=" << dt << " at t=" << t << "\n";
            break;
        }
        advance_rk3(G, dt, ch, use_glm);
        t += dt;
        ++step;
        for(std::size_t k=0; k < t_outs.size(); ++k) {
            if(!wrote[k] && std::abs(t - t_outs[k]) < 1e-12) {
                dump_snapshot(G, base, t_tags[k], t_outs[k], theta_deg);
                wrote[k] = true;
            }
        }
        if(step % 50 == 0 || t >= tfinal) {
            std::cerr<<"[reverse b-field] step=" << step << " t="
                     << t << " dt=" << dt << " ch=" << ch << "\n";
        }
    }
    
    std::cerr << "[reverse b-field] Wrote data to " << base << "/data\n";
    std::cerr << "[reverse b-field] Wrote plots to " << base << "/plots/png\n";
}