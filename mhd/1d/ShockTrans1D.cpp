// ============================================================================
// ShockTrans1D.cpp — 1D shock tube with a trans-Alfvénic shock wave
// ----------------------------------------------------------------------------
//   - HLLD fluxes
//   - WENO3 reconstruction
//   - SSP-RK3 time stepping
//   - Optional Dedner-style GLM hyperbolic divergence cleaning
//
// Initial conditions:
//   L: ρ=1.08   u=1.2      v=0.01    w=0.5       Bx=0.56418958354   
//      By=1.01554125039    Bz=0.56418958354      p=0.95
//   R: ρ=0.9891 u=-0.0131  v=0.0269  w=0.010037  Bx=0.56418958354
//      By=1.13526228001    Bz=0.0.56492303       p=0.97159
//   Discontinuity: x0=0.5  γ = 5/3  t_out=0.2
// ============================================================================
#include "ShockTrans1D.h"
 
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
static constexpr double CFL = 0.60;
 
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

static bool prompt_use_glm() {
    std::cout << "Enable hyperbolic divergence cleaning (GLM)? [y/n, default y]: ";
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

static inline Cons hlld_y(const Cons& UL_in, const Cons& UR_in, double ch, bool use_glm) {
    Prim WL = prim_with_floors(cons_to_prim(UL_in));
    Prim WR = prim_with_floors(cons_to_prim(UR_in));
    glm_correct_y(WL, WR, ch, use_glm);
    const Cons UL = prim_to_cons(WL);
    const Cons UR = prim_to_cons(WR);
    const Cons FL = flux_y(UL, ch, use_glm);
    const Cons FR = flux_y(UR, ch, use_glm);

    const double By_face = 0.5 * (WL.By + WR.By);
    const double psi_face = 0.5 * (WL.psi + WR.psi);
    const double ptL = WL.p + 0.5 * (WL.Bx * WL.Bx + WL.By * WL.By + WL.Bz * WL.Bz);
    const double ptR = WR.p + 0.5 * (WR.Bx * WR.Bx + WR.By * WR.By + WR.Bz * WR.Bz);
    const double cfL = fast_speed_dir(WL, By_face);
    const double cfR = fast_speed_dir(WR, By_face);

    const double SL = std::min(WL.vy - cfL, WR.vy - cfR);
    const double SR = std::max(WL.vy + cfL, WR.vy + cfR);
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double rhoL = clamp_pos(WL.rho);
    const double rhoR = clamp_pos(WR.rho);
    const double denomM = (SR - WR.vy) * rhoR - (SL - WL.vy) * rhoL;
    if (std::abs(denomM) < 1e-12) return hll_y(UL, UR, ch, use_glm);

    const double SM =
        ((SR - WR.vy) * UR.my - (SL - WL.vy) * UL.my + ptL - ptR) / denomM;
    const double pt_star =
        ((SR - WR.vy) * rhoR * ptL - (SL - WL.vy) * rhoL * ptR +
         rhoL * rhoR * (SR - WR.vy) * (SL - WL.vy) * (WR.vy - WL.vy)) / denomM;

    const auto star_state = [&](const Prim& W, const Cons& U, double S, double pt_side) -> std::pair<bool, Cons> {
        Cons Us{};
        const double rho = clamp_pos(W.rho);
        const double rho_star = rho * (S - W.vy) / (S - SM);
        const double D = rho * (S - W.vy) * (S - SM) - By_face * By_face;
        if (!is_finite(rho_star) || rho_star <= 0.0 || std::abs(D) < 1e-12) {
            return {false, {}};
        }

        const double factor = (rho * (S - W.vy) * (S - W.vy) - By_face * By_face) / D;
        const double vz_star = W.vz - (By_face * W.Bz * (SM - W.vy)) / D;
        const double vx_star = W.vx - (By_face * W.Bx * (SM - W.vy)) / D;
        const double Bz_star = W.Bz * factor;
        const double Bx_star = W.Bx * factor;

        const double vdotB = W.vy * By_face + W.vz * W.Bz + W.vx * W.Bx;
        const double vdotBs = SM * By_face + vz_star * Bz_star + vx_star * Bx_star;
        const double E_star =
            ((S - W.vy) * U.E - pt_side * W.vy + pt_star * SM +
             By_face * (vdotB - vdotBs)) / (S - SM);

        Us = {rho_star, rho_star * vx_star, rho_star * SM, rho_star * vz_star,
              Bx_star, By_face, Bz_star, E_star, psi_face};
        if (!state_is_admissible(Us)) {
            return {false, {}};
        }
        return {true, Us};
    };

    const auto [okL, UstL] = star_state(WL, UL, SL, ptL);
    const auto [okR, UstR] = star_state(WR, UR, SR, ptR);
    if (!(okL && okR)) return hll_y(UL, UR, ch, use_glm);

    const double rhoSL = clamp_pos(UstL.rho);
    const double rhoSR = clamp_pos(UstR.rho);
    const double sqL = std::sqrt(rhoSL);
    const double sqR = std::sqrt(rhoSR);
    const double denomA = sqL + sqR;
    if (denomA < 1e-12) return hll_y(UL, UR, ch, use_glm);

    const double signBy = (By_face >= 0.0) ? 1.0 : -1.0;
    const double SstL = SM - std::abs(By_face) / std::sqrt(rhoSL);
    const double SstR = SM + std::abs(By_face) / std::sqrt(rhoSR);

    const double vz_ss =
        (sqL * (UstL.mz / rhoSL) + sqR * (UstR.mz / rhoSR) +
         signBy * (UstR.Bz - UstL.Bz)) / denomA;
    const double vx_ss =
        (sqL * (UstL.mx / rhoSL) + sqR * (UstR.mx / rhoSR) +
         signBy * (UstR.Bx - UstL.Bx)) / denomA;
    const double Bz_ss =
        (sqL * UstL.Bz + sqR * UstR.Bz +
         signBy * std::sqrt(rhoSL * rhoSR) * ((UstR.mz / rhoSR) - (UstL.mz / rhoSL))) / denomA;
    const double Bx_ss =
        (sqL * UstL.Bx + sqR * UstR.Bx +
         signBy * std::sqrt(rhoSL * rhoSR) * ((UstR.mx / rhoSR) - (UstL.mx / rhoSL))) / denomA;

    Cons UssL = UstL, UssR = UstR;
    UssL.mz = rhoSL * vz_ss;
    UssL.mx = rhoSL * vx_ss;
    UssL.Bz = Bz_ss;
    UssL.Bx = Bx_ss;
    UssR.mz = rhoSR * vz_ss;
    UssR.mx = rhoSR * vx_ss;
    UssR.Bz = Bz_ss;
    UssR.Bx = Bx_ss;

    const double vtBtL = (UstL.mz / rhoSL) * UstL.Bz + (UstL.mx / rhoSL) * UstL.Bx;
    const double vtBtR = (UstR.mz / rhoSR) * UstR.Bz + (UstR.mx / rhoSR) * UstR.Bx;
    const double vtBtSS = vz_ss * Bz_ss + vx_ss * Bx_ss;
    UssL.E = UstL.E - sqL * (vtBtL - vtBtSS) * signBy;
    UssR.E = UstR.E + sqR * (vtBtR - vtBtSS) * signBy;
    if (!state_is_admissible(UssL) || !state_is_admissible(UssR)) {
        return hll_y(UL, UR, ch, use_glm);
    }

    const Cons FstL = c_add(FL, c_mul(SL, c_sub(UstL, UL)));
    const Cons FstR = c_add(FR, c_mul(SR, c_sub(UstR, UR)));
    const Cons FssL = c_add(FstL, c_mul(SstL, c_sub(UssL, UstL)));
    const Cons FssR = c_add(FstR, c_mul(SstR, c_sub(UssR, UstR)));

    Cons F = (SL <= 0.0 && 0.0 <= SstL) ? FstL :
             (SstL <= 0.0 && 0.0 <= SM) ? FssL :
             (SM <= 0.0 && 0.0 <= SstR) ? FssR : FstR;
    F.By = psi_face;
    F.psi = ch * ch * By_face;
    return F;
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
    const double p1 =  0.5 * q0  + 0.5 * qp1;
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
    double Lx = 1.0, Ly = 1.0, dx = 0.0, dy = 0.0;
    std::vector<Cons> U;

    int nx() const { return Nx + 2 * ng; }
    int ny() const { return Ny + 2 * ng; }
    inline int id(int i, int j) const { return j * nx() + i; }
};

static void apply_transmissive(Grid& G) {
    const int nx = G.nx(), ny = G.ny(), ng = G.ng;

    for (int j = 0; j < ny; ++j) {
        for (int g = 0; g < ng; ++g) {
            G.U[G.id(g, j)] = G.U[G.id(ng, j)];
            G.U[G.id(nx - 1 - g, j)] = G.U[G.id(nx - 1 - ng, j)];
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int g = 0; g < ng; ++g) {
            G.U[G.id(i, g)] = G.U[G.id(i, ng)];
            G.U[G.id(i, ny - 1 - g)] = G.U[G.id(i, ny - 1 - ng)];
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

// -------------------------------- time step --------------------------------
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

// ----------------------------- semi-discrete RHS ----------------------------
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

    apply_transmissive(G);
    enforce_floors(G);

    const std::vector<Cons> L1 = rhs(G, ch, use_glm);
    Grid G1 = G;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            G1.U[G.id(i, j)] = c_add(G.U[G.id(i, j)], c_mul(dt, L1[G.id(i, j)]));
        }
    }

    apply_transmissive(G1);
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
    
    apply_transmissive(G2);
    enforce_floors(G2);

    const std::vector<Cons> L3 = rhs(G2, ch, use_glm);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            G.U[G.id(i, j)] = c_add(c_mul(1.0 / 3.0, G.U[G.id(i, j)]),
                                    c_mul(2.0 / 3.0, c_add(G2.U[G.id(i, j)], c_mul(dt, L3[G.id(i, j)]))));
        }
    }

    apply_transmissive(G);
    enforce_floors(G);
}
 
// --------------------------------- plotting ---------------------------------
static void write_approx_dat_csv(const std::string& baseDir, int N,
                                const std::vector<Cons>& U, int ng, double dx) {
    const std::string datDir = baseDir + "/data/dat/approx/N" + std::to_string(N);
    const std::string csvDir = baseDir + "/data/csv/approx/N" + std::to_string(N);
    mkdir_p(datDir);
    mkdir_p(csvDir);

    const std::string datPath = datDir + "/approx.dat";
    const std::string csvPath = csvDir + "/approx.csv";

    std::ofstream dat(datPath);
    std::ofstream csv(csvPath);
    dat << "# x rho vx vy p By Bz \n";
    csv << "x,rho,vx,vy,p,By,Bz\n";
    dat << std::setprecision(16);
    csv << std::setprecision(16);

    for (int i = ng; i < static_cast<int>(U.size()) - ng; ++i) {
        const double x = (i - ng + 0.5) * dx;
        const Prim W = cons_to_prim(U[i]);
        dat << x << " " << W.rho << " " << W.vx << " " << W.vy << " "
            << W.p << " " << W.By << " " << W.Bz << "\n";
        csv << x << "," << W.rho << "," << W.vx << "," << W.vy << ","
            << W.p << "," << W.By << "," << W.Bz << "\n";
    }

    std::cerr << "[transverse] Wrote " << datPath << "\n";
    std::cerr << "[transverse] Wrote " << csvPath << "\n";
}

static void write_overlay_plots(const std::string& baseDir, double L) {
    const std::string pltDir = baseDir + "/plots/plt";
    const std::string pngDir = baseDir + "/plots/png";
    mkdir_p(pltDir);
    mkdir_p(pngDir);

    struct Var {
        std::string name;
        int col;
        std::string ylabel;
    };
    const std::vector<Var> vars = {
        {"rho", 2, "{/Symbol r}"},
        {"vx", 3, "v_x"},
        {"vy", 4, "v_y"},
        {"p", 5, "p"},
        {"By", 6, "B_y"},
        {"Bz", 7, "B_z"}
    };

    auto datpath = [&](int N) {
        return baseDir + "/data/dat/approx/N" + std::to_string(N) + "/approx.dat";
    };

    for (const auto& v : vars) {
        const std::string pltPath = pltDir + "/Transverse_" + v.name + ".plt";
        std::ofstream gp(pltPath);
        gp << "set term pngcairo size 900,650\n";
        gp << "set output '" << pngDir << "/Transverse_" << v.name << ".png'\n";
        gp << "unset grid\n";
        gp << "set border linewidth 1.5\n";
        gp << "set tics scale 0.5\n";
        gp << "set xlabel 'x'\n";
        gp << "set ylabel '" << v.ylabel << "'\n";
        gp << "set xrange [0:" << L << "]\n";
        gp << "set arrow from 0.5,graph 0 to 0.5,graph 1 nohead lc rgb '#aaaaaa' lw 1 dt 2\n";
        gp << "set key top right\n";
        gp << "plot \\\n";
        gp << "'" << datpath(100) << "' u 1:" << v.col
           << " w p pt 7 ps 0.6 lc rgb '#440154' title 'N = 100', \\\n";
        gp << "'" << datpath(200) << "' u 1:" << v.col
           << " w p pt 7 ps 0.6 lc rgb '#3B528B' title 'N = 200', \\\n";
        gp << "'" << datpath(400) << "' u 1:" << v.col
           << " w p pt 9 ps 1.2 lc rgb '#21918C' title 'N = 400'\n";
        gp.close();
        if (cmd_exists("gnuplot")) {
            syscmd("gnuplot \"" + pltPath + "\" >/dev/null 2>&1");
        }
    }
}

static void write_cut_and_plot(const std::string& base,
                               const std::vector<double>& s,
                               const std::vector<double>& rho,
                               const std::vector<double>& p,
                               const std::vector<double>& vx,
                               const std::vector<double>& vy,
                               const std::vector<double>& By,
                               const std::vector<double>& Bz) {
    mkdir_p(base + "/data");
    mkdir_p(base + "/plots/plt");
    mkdir_p(base + "/plots/png");
 
    {
        std::ofstream f(base + "/data/Transverse1D_Multiplot.dat");
        f << "# x  rho  p  vx  vy  By  Bz\n";
        f << std::setprecision(16);
        for (size_t k = 0; k < s.size(); ++k)
            f << s[k]  <<' '<< rho[k] <<' '<< p[k]  <<' '
              << vx[k] <<' '<< vy[k]  <<' '<< By[k] <<' '<< Bz[k] <<'\n';
    }
 
    {
        std::ofstream gp(base + "/plots/plt/Transverse1D_Multiplot.plt");
        gp << "set term pngcairo size 1000,900\n";
        gp << "set output '" << base << "/plots/png/Transverse1D_Multiplot.png'\n";
        gp << "unset grid\n";
        gp << "set border linewidth 1.2\n";
        gp << "set tics scale 0.6\n";
        gp << "set multiplot layout 3,2 rowsfirst title 'Transverse Shock Wave'\n";
        gp << "set xlabel 'x'\n";
        gp << "set xrange [0:1]\n";
        gp << "unset key\n";
        gp << "set arrow from 0.5,graph 0 to 0.5,graph 1 nohead lc rgb '#aaaaaa' lw 1 dt 2\n";
        gp << "set title '{/Symbol r}'\n";
        gp << "set yrange [*:*]\n";
        gp << "plot '" << base << "/data/Transverse1D_Multiplot.dat' u 1:2 w p pt 7 ps 0.25 lc rgb '#000000'\n";
        gp << "set title 'p'\n";
        gp << "set yrange [*:*]\n";
        gp << "plot '" << base << "/data/Transverse1D_Multiplot.dat' u 1:3 w p pt 7 ps 0.25 lc rgb '#000000'\n";
        gp << "set title 'v_x'\n";
        gp << "set yrange [*:*]\n";
        gp << "plot '" << base << "/data/Transverse1D_Multiplot.dat' u 1:4 w p pt 7 ps 0.25 lc rgb '#000000'\n";
        gp << "set title 'v_y'\n";
        gp << "set yrange [*:*]\n";
        gp << "plot '" << base << "/data/Transverse1D_Multiplot.dat' u 1:5 w p pt 7 ps 0.25 lc rgb '#000000'\n";
        gp << "set title 'B_y'\n";
        gp << "set yrange [*:*]\n";
        gp << "plot '" << base << "/data/Transverse1D_Multiplot.dat' u 1:6 w p pt 7 ps 0.25 lc rgb '#000000'\n";
        gp << "set title 'B_z'\n";
        gp << "set yrange [*:*]\n";
        gp << "plot '" << base << "/data/Transverse1D_Multiplot.dat' u 1:7 w p pt 7 ps 0.25 lc rgb '#000000'\n";
        gp << "unset multiplot\n";
        gp.close();
    }
 
    if (cmd_exists("gnuplot")) {
        syscmd("gnuplot \"" + base + "/plots/plt/Transverse1D_Multiplot.plt\" >/dev/null 2>&1");
    }
}
 
// ---------------------------------- driver ----------------------------------
void run_shock_trans() {
    const bool use_glm = prompt_use_glm();

    const int Ny = 4;
    const double tfinal = 0.2;
    const double x_discontinuity = 0.5;
    const std::vector<int> resolutions = {100, 200, 400};

    for (int Nx : resolutions) {
        Grid G;
        G.Nx = Nx;
        G.Ny = Ny;
        G.ng = 2;
        G.Lx = 1.0;
        G.Ly = 1.0;
        G.dx = G.Lx / G.Nx;
        G.dy = G.Ly / G.Ny;
        G.U.assign(G.nx() * G.ny(), Cons{0,0,0,0,0,0,0,0,0});
    
        Prim WL{};
        WL.rho = 1.08;
        WL.vx = 1.2;
        WL.vy = 0.01;
        WL.vz = 0.5;
        WL.p = 0.95;
        WL.Bx = 0.56418958354;
        WL.By = 1.01554125039;
        WL.Bz = 0.56418958354;
        WL.psi = 0.0;
    
        Prim WR{};
        WR.rho = 0.9891;
        WR.vx = -0.0131;
        WR.vy = 0.0269;
        WR.vz = 0.010037;
        WR.p = 0.97159;
        WR.Bx = 0.56418958354;
        WR.By = 1.13526228001;
        WR.Bz = 0.56492303;
        WR.psi = 0.0;
 
        for (int j = 0; j < G.ny(); ++j) {
            for (int i = 0; i < G.nx(); ++i) {
                const double x = (i - G.ng + 0.5) * G.dx;
                const bool left = (x <= x_discontinuity);
                G.U[G.id(i,j)] = prim_to_cons(left ? WL : WR);
            }
        }

        double t = 0.0;
        int step = 0;
        while (t < tfinal) {
            double ch = 0.0;
            double dt = compute_ch_and_dt(G, ch, use_glm);
            if (t + dt > tfinal) dt = tfinal - t;
            advance_rk3(G, dt, ch, use_glm);
            t += dt; ++step;
        }

        int jstar = G.ng; double best = 1e9;
        for (int j = G.ng; j < G.ny() - G.ng; ++j) {
            const double y = (j - G.ng + 0.5) * G.dy;
            const double d = std::abs(y - 0.5);
            if (d < best) { best = d; jstar = j; }
        }

        std::vector<double> s_out, rho_out, p_out, vx_out, vz_out, By_out, Bz_out;
        for (int i = G.ng; i < G.nx() - G.ng; ++i) {
            const double xi = (i - G.ng + 0.5) * G.dx;
            const Prim W = cons_to_prim(G.U[G.id(i, jstar)]);
            
            if (!is_finite(W.rho) || !is_finite(W.p)) continue;
            s_out.push_back(xi);
            rho_out.push_back(W.rho);
            p_out.push_back(W.p);
            vx_out.push_back(W.vx);
            vz_out.push_back(W.vz);
            By_out.push_back(W.By);
            Bz_out.push_back(W.Bz);
        }

        const std::string base = "outputs/MHD/1D/Transverse";
        write_approx_dat_csv(base, Nx, G.U, G.ng, G.dx);
        if (Nx == 400) {
            write_cut_and_plot(base, s_out, rho_out, p_out, vx_out, vz_out, By_out, Bz_out);
            std::cerr << "[transverse] Wrote data to " << base << "/data\n";
            std::cerr << "[transverse] Wrote plots to " << base << "/plots/png\n";
        }
        write_overlay_plots(base, G.Lx);
    }
}