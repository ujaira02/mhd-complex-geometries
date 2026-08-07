// ============================================================================
// KelvinHelmholtz2D.cpp — Kelvin-Helmholtz Instability test (2D).
// ----------------------------------------------------------------------------
//   - HLLD fluxes
//   - WENO3 reconstruction
//   - SSP-RK3 time stepping
//   - Optional Dedner-style GLM hyperbolic divergence cleaning
// 
// Initial conditions (Vides et al. 2013, Table 3):
//   ρ = 1,  p = 1/γ,  vz = 0
//   vx = (M/2) tanh(y/y0),  vy = A sin(2πx) exp(−y²/σ²)
//   Bx = ca √ρ cos(θ),  By = 0,  Bz = ca √ρ sin(θ)
//   θ = π/3,  M = 1,  ca = 0.1,  y0 = 1/20,  σ = 0.1,  A = 0.01,  psi = 0
//   Output times: t = 5, 8, 12, 20
// ============================================================================
#include "KelvinHelmholtz2D.h"

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

// -------------------------------- parameters --------------------------------
static constexpr double gamma_gas = 5.0 / 3.0;
static constexpr double CFL = 0.22;

static constexpr double kh_ca = 0.1;
static constexpr double kh_theta = 3.141592653589793238462643383279502884197 / 3.0;
static constexpr double kh_bz0 = 0.08660254037844387;
static constexpr double kh_M = 1.0;
static constexpr double kh_y0 = 1.0 / 20.0;
static constexpr double kh_sigma = 0.1;
static constexpr double kh_amp = 0.01;

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
    std::cout << "[kelvin-helmholtz] Enable hyperbolic divergence cleaning (GLM)? [y/n, default y]: ";
    std::string answer;
    std::getline(std::cin, answer);
    if (answer.empty()) return true;
    const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(answer[0])));
    return c != 'n';
}

// --------------------------------- physics ----------------------------------
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
        const double E_star =
            ((S - W.vx) * U.E - pt_side * W.vx + pt_star * SM +
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
        (sqrt_rho_L * (UstL.my / rho_star_L) + sqrt_rho_R * (UstR.my / rho_star_R) +
         signBx * (UstR.By - UstL.By)) / denomA;
    const double vz_ss =
        (sqrt_rho_L * (UstL.mz / rho_star_L) + sqrt_rho_R * (UstR.mz / rho_star_R) +
         signBx * (UstR.Bz - UstL.Bz)) / denomA;
    const double By_ss =
        (sqrt_rho_L * UstL.By + sqrt_rho_R * UstR.By +
         signBx * std::sqrt(rho_star_L * rho_star_R) *
         ((UstR.my / rho_star_R) - (UstL.my / rho_star_L))) / denomA;
    const double Bz_ss =
        (sqrt_rho_L * UstL.Bz + sqrt_rho_R * UstR.Bz +
         signBx * std::sqrt(rho_star_L * rho_star_R) *
         ((UstR.mz / rho_star_R) - (UstL.mz / rho_star_L))) / denomA;

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
    double x0 = 0.0, y0 = -1.0, Lx = 1.0, Ly = 2.0, dx = 0.0, dy = 0.0;
    std::vector<Cons> U;

    int nx() const { return Nx + 2 * ng; }
    int ny() const { return Ny + 2 * ng; }
    inline int id(int i, int j) const { return j * nx() + i; }
};

static void apply_kh_bc(Grid& G) {
    const int nx = G.nx(), ny = G.ny(), ng = G.ng;
    const int Nx = G.Nx, Ny = G.Ny;

    for (int j = 0; j < ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            G.U[G.id(g, j)] = G.U[G.id(ng, j)];
            G.U[G.id(nx - 1 - g, j)] = G.U[G.id(nx - 1 - ng, j)];
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int g = 0; g < ng; ++g) {
            {
                Cons C = G.U[G.id(i, ng + g)];
                C.my = -C.my;
                C.By = -C.By;
                C.psi = -C.psi;
                G.U[G.id(i, ng - 1 - g)] = C;
            }
            {
                Cons C = G.U[G.id(i, ny - ng - 1 - g)];
                C.my = -C.my;
                C.By = -C.By;
                C.psi = -C.psi;
                G.U[G.id(i, ny - ng + g)] = C;
            }
        }
    }
}

static void enforce_floors(Grid& G) {
    const int ng = G.ng, nx = G.nx(), ny = G.ny();
    constexpr double hard_limit = 1e6;
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            Prim W = cons_to_prim(G.U[G.id(i, j)]);
            if (!is_finite(W.rho) || !is_finite(W.p) || !is_finite(W.vx) ||
                !is_finite(W.vy) || !is_finite(W.vz) || !is_finite(W.Bx) ||
                !is_finite(W.By) || !is_finite(W.Bz) || !is_finite(W.psi) ||
                std::abs(W.vx) > hard_limit || std::abs(W.vy) > hard_limit ||
                std::abs(W.vz) > hard_limit || std::abs(W.Bx) > hard_limit ||
                std::abs(W.By) > hard_limit || std::abs(W.Bz) > hard_limit ||
                std::abs(W.psi) > hard_limit) {
                W.rho = 1.0;
                W.vx = W.vy = W.vz = 0.0;
                W.p = 1.0;
                W.Bx = W.By = W.Bz = 0.0;
                W.psi = 0.0;
            }
            W = prim_with_floors(W);
            G.U[G.id(i, j)] = prim_to_cons(W);
        }
    }
}

// -------------------------------- time step ---------------------------------
static double compute_ch_and_dt(const Grid& G, double& ch_out, bool use_glm) {
    double amax = 1e-14;
    for (int j = G.ng; j < G.ny() - G.ng; ++j) {
        for (int i = G.ng; i < G.nx() - G.ng; ++i) {
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
    ch_out = use_glm ? std::min(amax, 3.0) : 0.0;
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

    std::vector<Cons> Fx((nx - 1) * ny);
    auto fx_id = [&](int i, int j) { return j * (nx - 1) + i; };

    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng - 1; i <= nx - ng - 1; ++i) {
            Prim WL = reconstruct_left_weno3(W[G.id(i - 1, j)], W[G.id(i, j)], W[G.id(i + 1, j)]);
            Prim WR = reconstruct_right_weno3(W[G.id(i, j)], W[G.id(i + 1, j)], W[G.id(i + 2, j)]);
            WL = prim_with_floors(WL);
            WR = prim_with_floors(WR);
            Fx[fx_id(i, j)] = hlld_x(prim_to_cons(WL), prim_to_cons(WR), ch, use_glm);
        }
    }

    std::vector<Cons> Gy(nx * (ny - 1));
    auto gy_id = [&](int i, int j) { return j * nx + i; };

    for (int j = ng - 1; j <= ny - ng - 1; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            Prim WL = reconstruct_left_weno3(W[G.id(i, j - 1)], W[G.id(i, j)], W[G.id(i, j + 1)]);
            Prim WR = reconstruct_right_weno3(W[G.id(i, j)], W[G.id(i, j + 1)], W[G.id(i, j + 2)]);
            WL = prim_with_floors(WL);
            WR = prim_with_floors(WR);
            Gy[gy_id(i, j)] = hlld_y(prim_to_cons(WL), prim_to_cons(WR), ch, use_glm);
        }
    }

    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            const Cons dFx = c_sub(Fx[fx_id(i, j)], Fx[fx_id(i - 1, j)]);
            const Cons dGy = c_sub(Gy[gy_id(i, j)], Gy[gy_id(i, j - 1)]);
            Cons L = c_add(c_mul(-1.0 / G.dx, dFx), c_mul(-1.0 / G.dy, dGy));

            const double h = std::min(G.dx, G.dy);
            const double alpha = 1.0 / 0.30;
            const double cp2 = std::max(ch * h / alpha, 1e-12);
            if (use_glm) {
                L.psi += -(ch * ch / cp2) * G.U[G.id(i, j)].psi;
            }

            LU[G.id(i, j)] = L;
        }
    }
    return LU;
}

// --------------------------------- RK3 step ---------------------------------
static void advance_rk3(Grid& G, double dt, double ch, bool use_glm) {
    const int nx = G.nx(), ny = G.ny();

    apply_kh_bc(G);
    enforce_floors(G);

    const std::vector<Cons> L1 = rhs(G, ch, use_glm);
    Grid G1 = G;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            G1.U[G.id(i, j)] = c_add(G.U[G.id(i, j)], c_mul(dt, L1[G.id(i, j)]));
        }
    }

    apply_kh_bc(G1);
    enforce_floors(G1);

    const std::vector<Cons> L2 = rhs(G1, ch, use_glm);
    Grid G2 = G;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            G2.U[G.id(i, j)] =
                c_add(c_mul(0.75, G.U[G.id(i, j)]),
                      c_mul(0.25, c_add(G1.U[G.id(i, j)], c_mul(dt, L2[G.id(i, j)]))));
        }
    }

    apply_kh_bc(G2);
    enforce_floors(G2);

    const std::vector<Cons> L3 = rhs(G2, ch, use_glm);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            G.U[G.id(i, j)] = c_add(c_mul(1.0 / 3.0, G.U[G.id(i, j)]),
                                    c_mul(2.0 / 3.0, c_add(G2.U[G.id(i, j)], c_mul(dt, L3[G.id(i, j)]))));
        }
    }

    apply_kh_bc(G);
    enforce_floors(G);
}

// --------------------------------- plotting ---------------------------------
static std::vector<double> compute_divB_cellcenter(const Grid& G) {
    const int ng = G.ng, nx = G.nx(), ny = G.ny();
    std::vector<double> divB(nx * ny, 0.0);
    for (int j = ng; j < ny - ng; ++j)
        for (int i = ng; i < nx - ng; ++i) {
            const Prim Wxp = cons_to_prim(G.U[G.id(i+1, j)]);
            const Prim Wxm = cons_to_prim(G.U[G.id(i-1, j)]);
            const Prim Wyp = cons_to_prim(G.U[G.id(i, j+1)]);
            const Prim Wym = cons_to_prim(G.U[G.id(i, j-1)]);
            divB[G.id(i, j)] = (Wxp.Bx - Wxm.Bx)/(2.0*G.dx)
                            + (Wyp.By - Wym.By)/(2.0*G.dy);
        }
    return divB;
}

static void write_field_xy(const std::string& path,
                           const Grid& G,
                           const std::vector<double>& field,
                           const std::string& header) {
    std::ofstream f(path);
    f << "# " << header << "\n" << std::setprecision(16);
    const int ng = G.ng;
    for (int j = ng; j < G.ny() - ng; ++j) {
        const double y = G.y0 + (j - ng + 0.5)*G.dy;
        for (int i = ng; i < G.nx() - ng; ++i) {
            const double x = G.x0 + (i - ng + 0.5)*G.dx;
            f << x << " " << y << " " << field[G.id(i, j)] << "\n";
        }
        f << "\n";
    }
}

static void make_colour_plot(const std::string& base,
                             const Grid& G,
                             const std::string& dat,
                             const std::string& png,
                             const std::string& title,
                             const std::string& cblabel) {
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");

    const std::string pltfile = base + "/plots/plt/" + png.substr(0, png.size()-4) + ".plt";
    std::ofstream gp(pltfile);
    gp << "set term pngcairo size 900,800\n";
    gp << "set output '" << base << "/plots/png/" << png << "'\n";
    gp << "unset key\n";
    gp << "set size ratio -1\n";
    gp << "set xlabel 'x'\nset ylabel 'y'\n";
    gp << "set xrange [" << G.x0 << ":" << (G.x0 + G.Lx) << "]\n";
    gp << "set yrange [" << G.y0 << ":" << (G.y0 + G.Ly) << "]\n";
    gp << "set title '" << title << "'\n";
    gp << "set pm3d map\nset colorbox\n";
    gp << "set cblabel '" << cblabel << "'\n";
    gp << "set palette viridis\n";
    gp << "set border linewidth 1.2\nset tics scale 0.6\n";
    gp << "splot '" << base << "/data/" << dat << "' u 1:2:3 w pm3d\n";
    gp.close();

    if (cmd_exists("gnuplot"))
        syscmd("gnuplot \"" + pltfile + "\" >/dev/null 2>&1");
}

static void dump_fields(const Grid& G,
                        const std::string& base,
                        double tout,
                        const std::string& tag) {
    const int ng = G.ng;
    std::vector<double> rho (G.nx() * G.ny(), 0.0);
    std::vector<double> vx (G.nx() * G.ny(), 0.0);
    std::vector<double> vy (G.nx() * G.ny(), 0.0);
    std::vector<double> Bratio(G.nx() * G.ny(), 0.0);
    const double Btor0 = std::max(std::abs(kh_bz0), 1e-14);

    for (int j = ng; j < G.ny() - ng; ++j)
        for (int i = ng; i < G.nx() - ng; ++i) {
            const Prim W = cons_to_prim(G.U[G.id(i, j)]);
            const double Bpol = std::sqrt(W.Bx * W.Bx + W.By * W.By);
            rho [G.id(i, j)] = W.rho;
            vx [G.id(i, j)] = W.vx;
            vy [G.id(i, j)] = W.vy;
            Bratio[G.id(i, j)] = Bpol / Btor0;
        }

    const std::vector<double> divB = compute_divB_cellcenter(G);

    write_field_xy(base + "/data/rho_" + tag + ".dat", G, rho, "x y rho");
    write_field_xy(base + "/data/vx_" + tag + ".dat", G, vx, "x y vx");
    write_field_xy(base + "/data/vy_" + tag + ".dat", G, vy, "x y vy");
    write_field_xy(base + "/data/Bratio_" + tag + ".dat", G, Bratio, "x y Bratio");
    write_field_xy(base + "/data/divB_" + tag + ".dat", G, divB, "x y divB");

    const std::string ts = std::to_string(tout);

    make_colour_plot(base, G, "rho_" + tag + ".dat", "rho_" + tag + ".png",
                     "Kelvin-Helmholtz Instability {/Symbol r} at t = " + ts + "s", "{/Symbol r}");
    make_colour_plot(base, G, "vx_" + tag + ".dat", "vx_" + tag + ".png",
                     "Kelvin-Helmholtz Instability v_x at t = " + ts + "s", "v_x");
    make_colour_plot(base, G, "vy_" + tag + ".dat", "vy_" + tag + ".png",
                     "Kelvin-Helmholtz Instability v_y at t = " + ts + "s", "v_y");
    make_colour_plot(base, G, "Bratio_" + tag + ".dat", "Bratio_" + tag + ".png",
                     "Kelvin-Helmholtz Instability {/Symbol \326}(B_x^2 + B_y^2)/B_z at t = " + ts + "s",
                     "{/Symbol \326}(B_x^2 + B_y^2)/B_z");
    make_colour_plot(base, G, "divB_" + tag + ".dat", "divB_" + tag + ".png",
                     "Kelvin-Helmholtz Instability ∇·B at t = " + ts + "s", "∇·B");
}

// ---------------------------------- driver ----------------------------------
void run_kelvin_helmholtz2d() {
    const bool use_glm = prompt_use_glm();

    const int Nx = 128;
    const int Ny = 2 * Nx;
    const double t_out1 = 5.0;
    const double t_out2 = 8.0;
    const double t_out3 = 12.0;
    const double t_out4 = 20.0;
    const double tfinal = t_out4;

    Grid G;
    G.Nx = Nx;
    G.Ny = Ny;
    G.ng = 2;
    G.x0 = 0.0;
    G.y0 = -1.0;
    G.Lx = 1.0;
    G.Ly = 2.0;
    G.dx = G.Lx / G.Nx;
    G.dy = G.Ly / G.Ny;
    G.U.assign(G.nx() * G.ny(), Cons{0,0,0,0,0,0,0,0,0});

    const double pi = std::acos(-1.0);

    for (int j = 0; j < G.ny(); ++j)
        for (int i = 0; i < G.nx(); ++i) {
            const double x = G.x0 + (i - G.ng + 0.5) * G.dx;
            const double y = G.y0 + (j - G.ng + 0.5) * G.dy;

            Prim W{};
            W.rho = 1.0;
            W.p = 1.0 / gamma_gas;
            W.vx = 0.5 * kh_M * std::tanh(y / kh_y0);
            W.vy = kh_amp * std::sin(2.0 * pi * x) * std::exp(-(y*y) / (kh_sigma * kh_sigma));
            W.vz = 0.0;
            W.Bx = kh_ca * std::sqrt(W.rho) * std::cos(kh_theta);
            W.By = 0.0;
            W.Bz = kh_ca * std::sqrt(W.rho) * std::sin(kh_theta);
            W.psi = 0.0;

            G.U[G.id(i, j)] = prim_to_cons(W);
        }

    apply_kh_bc(G);
    enforce_floors(G);

    if (!use_glm) {
        std::cerr << "[kelvin-helmholtz] GLM disabled; the run will no longer match the "
                     "Vides comparison setup.\n";
    }

    const std::string base = "outputs/MHD/2D/KelvinHelmholtz";
    mkdir_p(base + "/data");
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");

    double t = 0.0;
    int step = 0;
    bool wrote05 = false;
    bool wrote08 = false;
    bool wrote12 = false;
    bool wrote20 = false;

    while (t < tfinal - 1e-14) {
        double ch = 0.0;
        double dt = compute_ch_and_dt(G, ch, use_glm);

        if (!wrote05 && t < t_out1 && t + dt > t_out1) dt = t_out1 - t;
        if (!wrote08 && t < t_out2 && t + dt > t_out2) dt = t_out2 - t;
        if (!wrote12 && t < t_out3 && t + dt > t_out3) dt = t_out3 - t;
        if (!wrote20 && t < t_out4 && t + dt > t_out4) dt = t_out4 - t;
        if (t + dt > tfinal) dt = tfinal - t;

        if (!is_finite(dt) || dt <= 0.0) {
            std::cerr << "[kelvin-helmholtz] invalid dt=" << dt
                      << " at t=" << t << " step=" << step << "\n";
            break;
        }

        advance_rk3(G, dt, ch, use_glm);

        t += dt;
        ++step;

        if (!wrote05 && std::abs(t - t_out1) < 1e-12) {
            dump_fields(G, base, t_out1, "t05");
            wrote05 = true;
        }
        if (!wrote08 && std::abs(t - t_out2) < 1e-12) {
            dump_fields(G, base, t_out2, "t08");
            wrote08 = true;
        }
        if (!wrote12 && std::abs(t - t_out3) < 1e-12) {
            dump_fields(G, base, t_out3, "t12");
            wrote12 = true;
        }
        if (!wrote20 && std::abs(t - t_out4) < 1e-12) {
            dump_fields(G, base, t_out4, "t20");
            wrote20 = true;
        }

        if (step % 50 == 0 || t >= tfinal) {
            std::cerr << "[kelvin-helmholtz] step=" << step
                      << " t=" << t << " dt=" << dt << " ch=" << ch << "\n";
        }
    }

    std::cerr << "\n[kelvin-helmholtz] Wrote data to " << base << "/data\n";
    std::cerr << "[kelvin-helmholtz] Wrote plots to " << base << "/plots/png\n";
}