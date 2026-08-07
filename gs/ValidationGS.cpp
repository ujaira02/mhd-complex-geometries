// =============================================================================
// ValidationGS.cpp - Validation of the Grad-Shafranov solver
// =============================================================================
// BUILD
//   g++ -std=c++17 -O2 -I /opt/homebrew/opt/eigen/include/eigen3 /
//   ValidationGS.cpp -o ValidationGS
// 
// RUN
//   ./ValidationGS
// =============================================================================
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "ValidationGS.h"

#include <vector>
#include <utility>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>

namespace gs {

// ---------------------------------- GS Grid ----------------------------------
class ValidationGSGrid {
public:
    ValidationGSGrid(double Rmin, double Rmax, double Zmin, double Zmax, int nR, int nZ)
           : Rmin_(Rmin), Rmax_(Rmax), Zmin_(Zmin), Zmax_(Zmax), nR_(nR), nZ_(nZ) {
        if (nR_ < 3 || nZ_ < 3) {
            throw std::invalid_argument("ValidationGSGrid: need at least 3 nodes per direction");
        }
        dR_ = (Rmax_ - Rmin_) / (nR_ - 1);
        dZ_ = (Zmax_ - Zmin_) / (nZ_ - 1);
    }

    int nR() const { return nR_; }
    int nZ() const { return nZ_; }
    double R(int i) const { return Rmin_ + i * dR_; }
    double Z(int j) const { return Zmin_ + j * dZ_; }
    int idx(int i, int j) const { return j * nR_ + i; }
    bool isBoundary(int i, int j) const {
        return (i == 0 || i == nR_ - 1 || j == 0 || j == nZ_ - 1);
    }

    Eigen::SparseMatrix<double> assembleDeltaStarOperator() const {
        const int N = nR_ * nZ_;
        Eigen::SparseMatrix<double> A(N, N);
        std::vector<Eigen::Triplet<double>> t;
        t.reserve(static_cast<size_t>(N) * 5);

        const double dR2 = dR_*dR_, dZ2 = dZ_*dZ_;

        for (int j = 0; j < nZ_; ++j)
            for (int i = 0; i < nR_; ++i) {
                int row = idx(i, j);
                if (isBoundary(i, j)) { t.emplace_back(row,row,1.0); continue; }
                double Ri = R(i);
                t.emplace_back(row, idx(i + 1, j), 1.0 / dR2 - 1.0 / (2.0 * Ri * dR_));
                t.emplace_back(row, idx(i - 1, j), 1.0 / dR2 + 1.0 / (2.0 * Ri * dR_));
                t.emplace_back(row, idx(i, j + 1), 1.0 / dZ2);
                t.emplace_back(row, idx(i, j - 1), 1.0 / dZ2);
                t.emplace_back(row, row, -2.0 / dR2 - 2.0 / dZ2);
            }
        A.setFromTriplets(t.begin(), t.end());
        A.makeCompressed();
        return A;
    }

    Eigen::VectorXd assembleRHS(
        const std::function<double(double, double)>& src,
        const std::function<double(double, double)>& bc) const {
        Eigen::VectorXd b(nR_ * nZ_);
        for (int j = 0; j < nZ_; ++j) {
            for (int i = 0; i < nR_; ++i) {
                b(idx(i, j)) = isBoundary(i, j) ? bc(R(i), Z(j)) : src(R(i), Z(j));
            }
        }
        return b;
    }

    void factorize() const {
        ensureFactorized();
    }

    Eigen::VectorXd solve(
        const std::function<double(double, double)>& src,
        const std::function<double(double, double)>& bc) const {
        ensureFactorized();
        Eigen::VectorXd psi = solver_.solve(assembleRHS(src,bc));
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("ValidationGSGrid: linear solve failed");
        }
        return psi;
    }

private:
    double Rmin_, Rmax_, Zmin_, Zmax_;
    int nR_, nZ_;
    double dR_, dZ_;

    mutable bool factorized_ = false;
    mutable Eigen::SparseMatrix<double> A_;
    mutable Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;

    void ensureFactorized() const {
        if (factorized_) return;
        A_ = assembleDeltaStarOperator();
        solver_.compute(A_);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("ValidationGSGrid: SparseLU factorization failed");
        }
        factorized_ = true;
    }
};

// ----------------------------------- Cases -----------------------------------
struct SolovevParams { double B0 = 0.5, q0 = 1.1, R0 = 0.95, kappa0 = 2.2; };

class SolovevCase {
public:
    explicit SolovevCase(const SolovevParams& p = SolovevParams{}) : p_(p) {
        c0_ = p_.B0 / (p_.R0 * p_.R0 * p_.kappa0 * p_.q0);
        c1_ = p_.B0 * (p_.kappa0 * p_.kappa0 + 1.0) / (p_.R0 * p_.R0 * p_.kappa0 * p_.q0);
    }
    double psiAnalytical(double R, double Z) const {
        double dR2 = R * R - p_.R0 * p_.R0;
        return 0.5   * c0_ * R * R * Z * Z + 0.125 * c0_ * p_.kappa0 * p_.kappa0 * dR2 * dR2;
    }
    double source(double R, double) const { return c1_ * R * R; }
    std::function<double(double, double)> sourceFunc() const {
        return [this](double R, double Z){ return source(R, Z); };
    }
    std::function<double(double, double)> bcFunc() const {
        return [this](double R, double Z){ return psiAnalytical(R, Z); };
    }
    double c0() const { return c0_; }
    double c1() const { return c1_; }
    double R0() const { return p_.R0; }
private:
    SolovevParams p_;
    double c0_, c1_;
};
double psiBasis(int n, double x, double y) {
    const double x2 = x * x, x4 = x2 * x2, lnx = std::log(x), y2 = y * y;
    switch(n) {
        case  1: return 1.0;
        case  2: return x2;
        case  3: return y2 - x2 * lnx;
        case  4: return x4 - 4.0 * x2 * y2;
        case  5: return 2.0 * y2 * y2 - 9.0 * y2 * x2 + 3.0 * x4 * lnx - 12.0 * x2 * y2 * lnx;
        case  6: return x4 * x2 - 12.0 * x4 * y2 + 8.0 * x2 * y2 * y2;
        case  7: return 8.0 * y2 * y2 * y2 - 140.0 * y2 * y2 * x2 + 75.0 * y2 * x4
                        - 15.0 * x4 * x2 * lnx + 180.0 * x4 * y2 * lnx - 120.0 * x2 * y2 * y2 * lnx;
        case  8: return y;
        case  9: return y * x2;
        case 10: return y2 * y - 3.0 * y * x2 * lnx;
        case 11: return 3.0 * y * x4 - 4.0 * y2 * y * x2;
        case 12: return 8.0 * y2 * y2 * y - 45.0 * y * x4 - 80.0 * y2 * y * x2 * lnx + 60.0 * y * x4 * lnx;
        default: throw std::invalid_argument("psiBasis: n in [1,12]");
    }
}

double psiBasis_dx(int n, double x, double y) {
    const double x2 = x * x, x3 = x2 * x, lnx = std::log(x), y2 = y * y;
    switch(n) {
        case  1: return 0.0;
        case  2: return 2.0 * x;
        case  3: return -2.0 * x * lnx - x;
        case  4: return 4.0 * x3 - 8.0 * x * y2;
        case  5: return -18.0 * y2 * x + 12.0 * x3 * lnx + 3.0 * x3 - 24.0 * x * y2 * lnx - 12.0 * x * y2;
        case  6: return 6.0 * x2 * x3 - 48.0 * x3 * y2 + 16.0 * x * y2 * y2;
        case  7: { const double y4 = y2 * y2;
                   return -280.0 * y4 * x + 300.0 * y2 * x3
                          - 15.0 * (6.0 * x2 * x3 * lnx + x2 * x3)
                          + 180.0 * (4.0 * x3 * y2 * lnx + x3 * y2)
                          - 120.0 * (2.0 * x * y4 * lnx + x * y4); }
        case  8: return 0.0;
        case  9: return 2.0 * x * y;
        case 10: return -3.0 * y * (2.0 * x * lnx + x);
        case 11: return 12.0 * y * x3 - 8.0 * y2 * y * x;
        case 12: return -180.0 * y * x3 - 80.0 * y2 * y * (2.0 * x * lnx + x) + 60.0 * y * (4.0 * x3 * lnx + x3);
        default: throw std::invalid_argument("psiBasis_dx: n in [1,12]");
    }
}

double psiBasis_dy(int n, double x, double y) {
    const double x2 = x * x, x4 = x2 * x2, lnx = std::log(x), y2 = y * y;
    switch(n) {
        case  1: return 0.0;
        case  2: return 0.0;
        case  3: return 2.0 * y;
        case  4: return -8.0 * x2 * y;
        case  5: return 8.0 * y * y2 - 18.0 * y * x2 - 24.0 * x2 * y * lnx;
        case  6: return -24.0 * x4 * y + 32.0 * x2 * y * y2;
        case  7: return 48.0 * y2 * y * y2 - 560.0 * y2 * y * x2 + 150.0 * y * x4
                        + 360.0 * x4 * y * lnx - 480.0 * x2 * y2 * y * lnx;
        case  8: return 1.0;
        case  9: return x2;
        case 10: return 3.0 * y2 - 3.0 * x2 * lnx;
        case 11: return 3.0 * x4 - 12.0 * y2 * x2;
        case 12: return 40.0 * y2 * y2 - 45.0 * x4 - 240.0 * y2 * x2 * lnx + 60.0 * x4 * lnx;
        default: throw std::invalid_argument("psiBasis_dy: n in [1,12]");
    }
}

double psiBasis_dxx(int n, double x, double y) {
    const double x2 = x * x, lnx = std::log(x), y2 = y * y;
    switch(n) {
        case  1: return 0.0;
        case  2: return 2.0;
        case  3: return -2.0 * lnx - 3.0;
        case  4: return 12.0 * x2 - 8.0 * y2;
        case  5: return -18.0 * y2 + 36.0 * x2 * lnx + 21.0 * x2 - 24.0 * y2 * lnx - 36.0 * y2;
        case  6: return 30.0 * x2 * x2 - 144.0 * x2 * y2 + 16.0 * y2 * y2;
        case  7: { const double y4 = y2 * y2;
                   return -280.0 * y4 + 900.0 * y2 * x2
                          - 15.0 * (30.0 * x2 * x2 * lnx + 11.0 * x2 * x2)
                          + 180.0 * (12.0 * x2 * y2 * lnx + 7.0 * x2 * y2)
                          - 120.0 * (2.0 * y4 * lnx + 3.0 * y4); }
        case  8: return 0.0;
        case  9: return 2.0 * y;
        case 10: return -3.0 * y * (2.0 * lnx + 3.0);
        case 11: return 36.0 * y * x2 - 8.0 * y2 * y;
        case 12: return -540.0 * y * x2 - 80.0 * y2 * y * (2.0 * lnx + 3.0) + 60.0 * y * (12.0 * x2 * lnx + 7.0 * x2);
        default: throw std::invalid_argument("psiBasis_dxx: n in [1,12]");
    }
}

double psiBasis_dyy(int n, double x, double y) {
    const double x2 = x * x, x4 = x2 * x2, lnx = std::log(x), y2 = y * y;
    switch(n) {
        case  1: return 0.0;
        case  2: return 0.0;
        case  3: return 2.0;
        case  4: return -8.0 * x2;
        case  5: return 24.0 * y2 - 18.0 * x2 - 24.0 * x2 * lnx;
        case  6: return -24.0 * x4 + 96.0 * x2 * y2;
        case  7: return 240.0 * y2 * y2 - 1680.0 * y2 * x2 + 150.0 * x4
                        + 360.0 * x4 * lnx - 1440.0 * x2 * y2 * lnx;
        case  8: return 0.0;
        case  9: return 0.0;
        case 10: return 6.0 * y;
        case 11: return -24.0 * y * x2;
        case 12: return 160.0 * y2 * y - 480.0 * y * x2 * lnx;
        default: throw std::invalid_argument("psiBasis_dyy: n in [1,12]");
    }
}

struct SingleNullParams {
    double A = -0.155, R0 = 1.0, epsilon = 0.32, kappa = 1.7, delta = 0.33;
};

class SingleNullCase {
public:
    explicit SingleNullCase(const SingleNullParams& p = SingleNullParams{}) : p_(p) {
        solveCoefficients();
    }

    double psiAnalytical(double R, double Z) const {
        double x = R / p_.R0, y = Z / p_.R0;
        double psi = psiParticular(x, y);
        for (int n = 1 ; n <= 12; ++n) psi += c_(n - 1)*psiBasis(n, x, y);
        return psi;
    }
    double source(double R, double) const {
        double x = R / p_.R0;
        return (1.0 - p_.A) * x * x + p_.A;
    }
    std::function<double(double, double)> sourceFunc() const {
        return [this](double R, double Z){ return source(R, Z); };
    }
    std::function<double(double, double)> bcFunc() const {
        return [this](double R, double Z){ return psiAnalytical(R, Z); };
    }
    const Eigen::Matrix<double,12,1>& coefficients() const { return c_; }

    double xPointR() const { return p_.R0 * x_s_; }
    double xPointZ() const { return p_.R0 * y_s_; }

private:
    SingleNullParams p_;
    Eigen::Matrix<double, 12, 1> c_;
    double x_s_ = 0.0, y_s_ = 0.0;

    double psiParticular(double x, double y) const {
        (void)y;
        double x2 = x * x, x4 = x2 * x2;
        return x4 / 8.0 + p_.A * (0.5 * x2 * std::log(x) - x4 / 8.0);
    }
    static double psiParticular_dx(double A, double x) {
        double x2 = x * x, x3 = x2 * x, lnx = std::log(x);
        return 0.5 * x3 + A * (x * lnx + 0.5 * x - 0.5 * x3);
    }
    static double psiParticular_dxx(double A, double x) {
        double x2 = x * x, lnx = std::log(x);
        return 1.5 * x2 + A * (lnx + 1.5 - 1.5 * x2);
    }

    void solveCoefficients() {
        const double eps = p_.epsilon, kap = p_.kappa, del = p_.delta, A = p_.A;

        const double x_o = 1.0 + eps, x_i = 1.0 - eps;
        const double x_t = 1.0 - del * eps, y_t = kap * eps;
        const double x_s = 1.0 - 1.1 * del * eps, y_s = -1.1 * kap * eps;
        x_s_ = x_s; y_s_ = y_s;

        const double alp = std::asin(del);
        const double N1 = -(1.0 + alp) * (1.0 + alp) / (eps * kap * kap);
        const double N2 = (1.0 - alp) * (1.0 - alp) / (eps * kap * kap);
        const double N3 = -kap / (eps * std::cos(alp) * std::cos(alp));

        Eigen::Matrix<double, 12, 12> M;
        Eigen::Matrix<double, 12, 1>  rhs;
        int r = 0;

        auto PP = [&](double x, double y) { return psiParticular(x, y); };
        auto dxP = [&](double x) { return psiParticular_dx(A, x); };
        auto dxxP= [&](double x) { return psiParticular_dxx(A, x); };

        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis(n, x_o, 0.0);
        rhs(r) = -PP(x_o, 0.0); ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dy(n, x_o, 0.0);
        rhs(r) = 0.0; ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dyy(n, x_o, 0.0) + N1 * psiBasis_dx(n, x_o, 0.0);
        rhs(r) = -N1 * dxP(x_o); ++r;

        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis(n, x_i, 0.0);
        rhs(r) = -PP(x_i, 0.0); ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dy(n, x_i, 0.0);
        rhs(r) = 0.0; ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dyy(n, x_i, 0.0) + N2 * psiBasis_dx(n, x_i, 0.0);
        rhs(r) = -N2 * dxP(x_i); ++r;

        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis(n, x_t, y_t);
        rhs(r) = -PP(x_t, y_t); ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dx(n, x_t, y_t);
        rhs(r) = -dxP(x_t); ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dxx(n, x_t, y_t) + N3 * psiBasis_dy(n, x_t, y_t);
        rhs(r) = -dxxP(x_t); ++r;

        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis(n, x_s, y_s);
        rhs(r) = -PP(x_s, y_s); ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dx(n, x_s, y_s);
        rhs(r) = -dxP(x_s); ++r;
        for (int n = 1; n <= 12; ++n) M(r, n - 1) = psiBasis_dy(n, x_s, y_s);
        rhs(r) = 0.0; ++r;

        if (r != 12) throw std::logic_error("SingleNullCase: constraint count mismatch");

        c_ = M.colPivHouseholderQr().solve(rhs);
        double res = (M * c_ - rhs).norm();
        if (res > 1e-6 * (rhs.norm() + 1.0)) {
            throw std::runtime_error("SingleNullCase: ill-conditioned system, residual=" + std::to_string(res));
        }
    }
};

// -------------------------------- Self-tests ---------------------------------
namespace selftest {
bool checkBasisFunctions() {
    std::vector<std::pair<double, double>> pts = {
        {1.0, 0.0}, {1.3, 0.4}, {0.8, -0.5}, {1.1, 0.2}, {0.9, 0.6}
    };
    std::cout << std::scientific << std::setprecision(3);
    bool ok = true;

    std::cout << "Delta* via FD (should be ~0):\n";
    for (int n = 1; n <= 12; ++n){
        std::cout <<"  psi_"<< n <<": ";
        for (auto&[x, y] : pts){
            double h = 1e-4;
            double pxx = (psiBasis(n, x + h, y) - 2 * psiBasis(n, x, y) + psiBasis(n, x - h, y)) / (h * h);
            double pyy = (psiBasis(n, x, y + h) - 2 * psiBasis(n, x, y) + psiBasis(n, x, y - h)) / (h * h);
            double px = (psiBasis(n, x + h, y) - psiBasis(n, x - h, y)) / (2 * h);
            double v = pxx - px / x + pyy;
            std::cout << v << " ";
            if (std::abs(v) / (std::abs(psiBasis(n, x, y)) + 1.0) > 1e-2) ok = false;
        }
        std::cout << "\n";
    }

    auto check1 = [&] (const char* tag, auto an, auto fd) {
        bool d = true;
        for (int n = 1; n <= 12; ++n) for (auto&[x, y] : pts) {
            double sc = std::abs(fd(n, x, y)) + 1.0;
            if (std::abs(an(n, x, y) - fd(n, x, y)) / sc > 1e-3) {
                std::cout << "  MISMATCH psi_" << n << " at (" << x << "," << y << ")\n"; d = false;
            }
        }
        std::cout << (d ? "  ALL OK\n" : "  MISMATCHES\n");
        return d;
    };

    double h1 = 1e-6, h2 = 1e-4;
    std::cout << "\nFirst-derivative cross-check (dx):\n";
    ok &= check1 ("dx",
        [](int n, double x, double y) {return psiBasis_dx(n, x, y);},
        [h1](int n, double x, double y) {return (psiBasis(n, x + h1, y) - psiBasis(n, x - h1, y)) / (2 * h1);});
    std::cout<<"First-derivative cross-check (dy):\n";
    ok &= check1("dy",
        [](int n, double x, double y) {return psiBasis_dy(n, x, y);},
        [h1](int n, double x, double y) {return (psiBasis(n, x, y + h1) - psiBasis(n, x, y - h1)) / (2 * h1);});
    std::cout<<"Second-derivative cross-check (dxx):\n";
    ok &= check1("dxx",
        [](int n, double x, double y) {return psiBasis_dxx(n, x, y);},
        [h2](int n, double x, double y) {
            return (psiBasis(n, x + h2, y) - 2 * psiBasis(n, x, y) + psiBasis(n, x - h2, y)) / (h2 * h2);});
    std::cout<<"Second-derivative cross-check (dyy):\n";
    ok &= check1("dyy",
        [](int n, double x, double y) {return psiBasis_dyy(n, x, y);},
        [h2](int n, double x, double y) {
            return (psiBasis(n, x, y + h2) - 2 * psiBasis(n, x, y) + psiBasis(n, x, y - h2)) / (h2 * h2);});
    std::cout<<"Exact analytical Delta* (dxx - dx/x + dyy == 0):\n";
    bool ex = true;
    for (int n = 1; n <= 12; ++n) for (auto&[x, y] : pts){
        double v = psiBasis_dxx(n, x, y) - psiBasis_dx(n, x, y) / x + psiBasis_dyy(n, x, y);
        double sc = std::abs(psiBasis_dxx(n, x, y)) + std::abs(psiBasis_dyy(n, x, y)) + 1.0;
        if (std::abs(v)/sc > 1e-10){
            std::cout << "  NONZERO psi_"<<n<<" at ("<<x<<","<<y<<"): " << v << "\n"; ex = false;
        }
    }
    std::cout << (ex ? "  ALL EXACTLY ZERO\n" : "  NONZERO RESIDUALS\n");
    return ok && ex;
}
}

}
// -------------------------------- I/O helpers --------------------------------
namespace {

int findAxisIndex(const gs::ValidationGSGrid& g, const Eigen::VectorXd& psiN) {
    const int nR = g.nR(), nZ = g.nZ();
    const double dR = g.R(1)-g.R(0), dZ = g.Z(1)-g.Z(0);

    int best = -1;
    double bestVal = std::numeric_limits<double>::infinity();
    for (int j = 1; j < nZ - 1; ++j) {
        for (int i = 1; i < nR - 1; ++i) {
            double v  = psiN(g.idx(i, j));
            double vE = psiN(g.idx(i + 1, j));
            double vW = psiN(g.idx(i - 1, j));
            double vN = psiN(g.idx(i, j + 1));
            double vS = psiN(g.idx(i, j - 1));
            if (!(v <= vE && v <= vW && v <= vN && v <= vS)) continue;

            double Hrr = (vE - 2.0 * v + vW) / (dR * dR);
            double Hzz = (vN - 2.0 * v + vS) / (dZ * dZ);
            double vNE = psiN(g.idx(i + 1, j + 1));
            double vNW = psiN(g.idx(i - 1, j + 1));
            double vSE = psiN(g.idx(i + 1, j - 1));
            double vSW = psiN(g.idx(i - 1, j - 1));
            double Hrz = (vNE - vNW - vSE + vSW) / (4.0 * dR * dZ);
            double D = Hrr * Hzz - Hrz * Hrz;
            if (D <= 0.0) continue;

            if (v < bestVal) { bestVal = v; best = g.idx(i, j); }
        }
    }
    return best;
}

double boundaryMinPsi(const gs::ValidationGSGrid& g, const Eigen::VectorXd& psiN) {
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i<g.nR(); ++i) {
        m = std::min(m, psiN(g.idx(i,0)));
        m = std::min(m, psiN(g.idx(i,g.nZ()-1)));
    }
    for (int j = 0; j<g.nZ(); ++j) {
        m = std::min(m, psiN(g.idx(0,j)));
        m = std::min(m, psiN(g.idx(g.nR()-1, j)));
    }
    return m;
}

std::vector<char> buildInsideMask(const gs::ValidationGSGrid& g, const Eigen::VectorXd& psiN,
                                  int axisIdx, double level) {
    const int nR = g.nR(), nZ = g.nZ();
    std::vector<char> inside(static_cast<size_t>(nR)*nZ, 0);
    const double axisVal = psiN(axisIdx);
    const double sgnRef = axisVal - level;
    const double maxDist = std::abs(axisVal - level);

    for (int j = 0; j < nZ; ++j)
        for (int i = 0; i < nR; ++i) {
            int idx = g.idx(i, j);
            double v = psiN(idx) - level;
            bool sameSide = (sgnRef >= 0.0) ? (v >= 0.0) : (v <= 0.0);
            bool withinRange = std::abs(v) <= maxDist;
            inside[idx] = (sameSide && withinRange) ? 1 : 0;
        }
    return inside;
}

void writeFieldCSV(const std::string& path, const gs::ValidationGSGrid& g,
                   const Eigen::VectorXd& pn,
                   const std::function<double(double, double)>& pa) {
    std::ofstream f(path);
    f << std::setprecision(12) << "R, Z,psi_numerical,psi_analytical,abs_error\n";
    for (int j = 0; j < g.nZ(); ++j) {
        double Z = g.Z(j);
        if (j > 0) f << "\n";
        for (int i = 0; i<g.nR(); ++i){
            double R=g.R(i),num=pn(g.idx(i, j)),an=pa(R, Z);
            f<<R<<","<<Z<<","<<num<<","<<an<<","<<std::abs(num-an)<<"\n";
        }
    }
}

void writeMaskedCSV(const std::string& path, const gs::ValidationGSGrid& g,
                    const Eigen::VectorXd& pn,
                    const std::function<double(double, double)>& pa,
                    const std::vector<char>& mask) {
    std::ofstream fi(path);
    fi << std::setprecision(12) << "R, Z,psi_numerical,psi_analytical,abs_error\n";
    for (int j2=0; j2<g.nZ(); ++j2){
        double Z2 = g.Z(j2);
        if (j2 > 0) fi << "\n";
        for (int i2=0; i2<g.nR(); ++i2){
            int idx2 = g.idx(i2,j2);
            double R2=g.R(i2), num2=pn(idx2), an2=pa(R2,Z2);
            if (!mask[idx2]) {
                fi<<R2<<","<<Z2<<","<<"NaN"<<","<<"NaN"<<","<<"NaN"<<"\n";
            } else {
                fi<<R2<<","<<Z2<<","<<num2<<","<<an2<<","<<std::abs(num2-an2)<<"\n";
            }
        }
    }
}

void writeAxisCSV(const std::string& path, const gs::ValidationGSGrid& g,
                  const Eigen::VectorXd& pn,
                  const std::function<double(double, double)>& pa) {
    int jZ=0; double best=1e300;
    for (int j = 0; j<g.nZ(); ++j)
        if (std::abs(g.Z(j))<best){best=std::abs(g.Z(j)); jZ=j;}
    std::ofstream f(path);
    f<<std::setprecision(12)<<"R,psi_numerical,psi_analytical,abs_error\n";
    for (int i = 0; i<g.nR(); ++i){
        double R=g.R(i),num=pn(g.idx(i, jZ)),an=pa(R,g.Z(jZ));
        f<<R<<","<<num<<","<<an<<","<<std::abs(num-an)<<"\n";
    }
}

double maxErr(const gs::ValidationGSGrid& g, const Eigen::VectorXd& pn,
              const std::function<double(double, double)>& pa) {
    double m=0.0;
    for (int j = 0; j<g.nZ(); ++j)
        for (int i = 0; i<g.nR(); ++i)
            m=std::max(m,std::abs(pn(g.idx(i, j))-pa(g.R(i),g.Z(j))));
    return m;
}

void mkdirRecursive(const std::string& path) {
    std::string current;
    for (char c : path) {
        if (c == '/') {
            if (!current.empty()) mkdir(current.c_str(), 0755);
            current += c;
        } else {
            current += c;
        }
    }
    if (!current.empty()) mkdir(current.c_str(), 0755);
}

// ------------------------------ Gnuplot Scripts ------------------------------
bool runGnuplot(const std::string& path) {
    std::string cmd = "gnuplot \"" + path + "\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
}

bool extractContourTable(const std::string& fieldCSV,
                         const std::string& rawTableOut,
                         const std::string& commaTableOut,
                         const std::string& levelsSpec)
{
    {
        std::ofstream plt_tmp(rawTableOut + ".plt");
        plt_tmp <<
        "reset\n"
        "set fit logfile '/dev/null'\n"
        "set datafile separator ','\n"
        "set table '" << rawTableOut << "'\n"
        "set contour base\n"
        "set cntrparam levels " << levelsSpec << "\n"
        "unset surface\n"
        "splot '" << fieldCSV << "' using 1:2:3 with lines\n"
        "unset table\n";
        plt_tmp.close();
        std::string cmd = "gnuplot \"" + rawTableOut + ".plt\" 2>/dev/null";
        if (std::system(cmd.c_str()) != 0) return false;
    }

    std::ifstream in(rawTableOut);
    if (!in) return false;
    std::ofstream out(commaTableOut);
    std::string line;
    while (std::getline(in, line)) {
        size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) { out << "\n"; continue; }
        if (line[b] == '#') { out << "\n"; continue; }
        size_t e = line.find_last_not_of(" \t\r\n");
        std::string trimmed = line.substr(b, e - b + 1);

        std::string csvLine;
        bool inToken = false;
        for (char c : trimmed) {
            if (c == ' ' || c == '\t') {
                if (inToken) { csvLine += ','; inToken = false; }
            } else {
                csvLine += c;
                inToken = true;
            }
        }
        out << csvLine << "\n";
    }
    return true;
}

bool extractLCFSContour(const std::string& fieldCSV,
                        const std::string& rawTableOut,
                        const std::string& commaTableOut,
                        double level)
{
    std::ostringstream spec;
    spec << "discrete " << std::setprecision(15) << level;
    return extractContourTable(fieldCSV, rawTableOut, commaTableOut, spec.str());
}

std::vector<double> buildLevelsInRange(double lo, double hi, int n)
{
    std::vector<double> levels;
    if (n < 1) return levels;
    double a = std::min(lo, hi), b = std::max(lo, hi);
    const double margin = 1e-6;
    for (int k = 0; k < n; ++k) {
        double t = (n > 1) ? static_cast<double>(k) / (n - 1) : 0.5;
        double tInset = margin + (1.0 - 2.0 * margin) * t;
        levels.push_back(a + (b - a) * tInset);
    }
    return levels;
}

bool extractDiscreteLevelContours(const std::string& fieldCSV,
                                  const std::string& rawTableOut,
                                  const std::string& commaTableOut,
                                  const std::vector<double>& levels)
{
    if (levels.empty()) return false;
    std::ostringstream spec;
    spec << "discrete " << std::setprecision(15);
    for (size_t k = 0; k < levels.size(); ++k) {
        if (k) spec << ",";
        spec << levels[k];
    }
    return extractContourTable(fieldCSV, rawTableOut, commaTableOut, spec.str());
}

std::string gnuplotTitleSafe(const std::string& label) {
    std::string s = label;
    for (char& c : s) if (c == '_') c = ' ';
    return s;
}

void writeFieldPlotScript(const std::string& outdir,
                          const std::string& label,
                          const std::string& csvField,
                          const std::string& csvAxis,
                          double Rmin, double Rmax, double Zmin, double Zmax,
                          double lcfsLevel,
                          double colorLo, double colorHi,
                          double domainMin, double domainMax,
                          int nBgLevels,
                          int nFineLevels,
                          const std::vector<double>* bgLevelsOverride,
                          bool execute) {
    std::string dataDir = outdir + "/" + label + "/data";
    std::string pngDir = outdir + "/" + label + "/plots/png";
    std::string pltDir = outdir + "/" + label + "/plots/plt";
    mkdirRecursive(dataDir);
    mkdirRecursive(pngDir);
    mkdirRecursive(pltDir);

    const std::string pngContour = pngDir + "/" + label + "_Contour.png";
    const std::string pngError   = pngDir + "/" + label + "_Error.png";
    const std::string script     = pltDir + "/plot_" + label + ".plt";
    const std::string lcfsRaw    = dataDir + "/." + label + "_LCFS_raw.dat";
    const std::string lcfsComma  = dataDir + "/." + label + "_LCFS.csv";
    std::string title = gnuplotTitleSafe(label);
    if (label == "Single-Null") {
        title = "Single-Null";
    } else if (label == "Solovev") {
        title = "Solovev";
    }
    const std::string csvFieldInside = dataDir + "/" + label + "_FieldInside.csv";
    const std::string fineLinesRaw    = dataDir + "/." + label + "_Finelines_raw.dat";
    const std::string fineLinesComma  = dataDir + "/." + label + "_FineLines.csv";
    const std::string bgLinesRaw      = dataDir + "/." + label + "_BgLines_raw.dat";
    const std::string bgLinesComma    = dataDir + "/." + label + "_BgLines.csv";

    const double xlo = Rmin, xhi = Rmax;
    const double ylo = Zmin, yhi = Zmax;

    const double aspect = (yhi - ylo) / (xhi - xlo);
    const int canvasW = 620;
    const int canvasH = static_cast<int>(200 + 250 * aspect);

    bool haveLCFS = false;
    bool haveFineLines = false;
    bool haveBgLines = false;
    if (execute) {
        haveLCFS = extractLCFSContour(csvField, lcfsRaw, lcfsComma, lcfsLevel);
        auto fineLevels = buildLevelsInRange(colorLo, colorHi, nFineLevels);
        haveFineLines = extractDiscreteLevelContours(csvField, fineLinesRaw, fineLinesComma, fineLevels);
        std::vector<double> bgLevels = bgLevelsOverride
            ? *bgLevelsOverride
            : buildLevelsInRange(domainMin, domainMax, nBgLevels);
        if (!bgLevels.empty()) {
            size_t nearest = 0;
            double bestD = std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < bgLevels.size(); ++k) {
                double d = std::abs(bgLevels[k] - lcfsLevel);
                if (d < bestD) { bestD = d; nearest = k; }
            }
            double offset = lcfsLevel - bgLevels[nearest];
            for (double& lv : bgLevels) lv += offset;
        }
        haveBgLines = extractDiscreteLevelContours(csvField, bgLinesRaw, bgLinesComma, bgLevels);
    }


    std::ofstream gp(script);
    gp <<
    "# ================================================================\n"
    "# Gnuplot script: " << label << "\n"
    "# Run: gnuplot " << script << "\n"
    "# ================================================================\n"
    "reset\n"
    "set fit logfile '/dev/null'\n"
    "set datafile separator ','\n"
    "\n"
    "set terminal pngcairo size " << canvasW << "," << canvasH
        << " enhanced font 'Helvetica,13'\n"
    "set output '" << pngContour << "'\n"
    "\n"
    "set title '" << title << " - Poloidal Flux {/Symbol Y}(R, Z)'\n"
    "set xlabel 'R [m]'\n"
    "set ylabel 'Z [m]'\n"
    "set xrange [" << xlo << ":" << xhi << "]\n"
    "set yrange [" << ylo << ":" << yhi << "]\n"
    "set size ratio -1\n"
    "unset key\n"
    "set view map\n"
    "\n"
    "set lmargin at screen 0.12\n"
    "set rmargin at screen 0.80\n"
    "set tmargin at screen 0.92\n"
    "set bmargin at screen 0.10\n"
    "\n"
    "set tics nomirror\n"
    "\n"
    "set xtics scale 0\n"
    "set ytics scale 0\n"
    "set palette viridis \n"
    "set cblabel '{/Symbol Y} [Tm²]'\n"
    "set cbrange [" << std::min(colorLo,colorHi) << ":" << std::max(colorLo,colorHi) << "]\n"
    "\n";

    if (haveLCFS) {
        gp << "splot '" << csvFieldInside << "' using 1:2:3 with pm3d notitle";
        if (haveBgLines)
            gp << ", \\\n      '" << bgLinesComma   << "' using 1:2:3 with lines lc rgb '#000000' lw 0.3 notitle";
        if (haveFineLines)
            gp << ", \\\n      '" << fineLinesComma << "' using 1:2:3 with lines lc rgb '#000000' lw 0.5 notitle";
        gp << ", \\\n      '" << lcfsComma << "' using 1:2:3 with lines lc rgb 'black' lw 3 notitle\n";
    } else {
        gp <<
        "splot '" << csvField << "' using 1:2:3 with pm3d notitle\n";
    }

    gp <<
    "\n"
    "# ----------------------------------------------------------------\n"
    "# Figure 2: Absolute error along Z_0\n"
    "# ----------------------------------------------------------------\n"
    "reset\n"
    "set fit logfile '/dev/null'\n"
    "set datafile separator ','\n"
    "set terminal pngcairo size 680,400 enhanced font 'Helvetica,13'\n"
    "set output '" << pngError << "'\n"
    "\n"
    "set title '" << title << " - |{/Symbol Y}_{num} - {/Symbol Y}_{an}| along Z_0'\n"
    "set xlabel 'R [m]'\n"
    "set ylabel 'Absolute Error [Tm²]'\n"
    "set key off\n"
    "set grid\n"
    "\n"
    "plot '" << csvAxis << "' using 1:4 with lines lw 2.5 \\\n"
    "     lc rgb '" << (label=="Solovev" ? "#cc2222" : "#2255cc") << "' notitle\n"
    ;
    gp.close();

    if (execute) {
        if (!runGnuplot(script)) {
            std::cout << "  gnuplot failed - is gnuplot installed? "
                         "(brew install gnuplot  /  apt install gnuplot)\n";
        }

        if (!haveLCFS) {
            std::cout << "  note: LCFS contour extraction failed; "
                         "contour plot shows the heatmap only.\n";
        }
    }
}

void writeConvergenceScript(const std::string& outdir,
                            const std::string& label,
                            const std::string& csvPath,
                            bool execute)
{
    std::string pngDir = outdir + "/Convergence/plots/png";
    std::string pltDir = outdir + "/Convergence/plots/plt";
    mkdirRecursive(pngDir);
    mkdirRecursive(pltDir);
    
    const std::string png    = pngDir + "/Conv_" + label + ".png";
    const std::string script = pltDir + "/Conv_" + label + ".plt";
    std::string title = gnuplotTitleSafe(label);
    if (label == "Single-Null") {
        title = "Single-Null";
    } else if (label == "Solovev") {
        title = "Solovev";
    }

    std::ofstream gp(script);
    gp <<
    "reset\n"
    "set fit logfile '/dev/null'\n"
    "set datafile separator ','\n"
    "set terminal pngcairo size 640,440 enhanced font 'Helvetica,13'\n"
    "set output '" << png << "'\n"
    "\n"
    "set logscale xy\n"
    "set title '" << title << " - Grid Convergence'\n"
    "set xlabel 'Grid Size N'\n"
    "set ylabel 'max|{/Symbol Y}_{num} - {/Symbol Y}_{an}| [Tm²]'\n"
    "set key top right\n"
    "set grid\n"
    "\n"
    "a = 0.1\n"
    "f(x) = a * x**(-2)\n"
    "fit f(x) '" << csvPath << "' using 1:2 via a\n"
    "\n"
    "plot '" << csvPath << "' using 1:2 \\\n"
    "         with linespoints pt 7 ps 1.4 lw 2 lc rgb '#cc2222' \\\n"
    "         title 'numerical max error', \\\n"
    "     f(x) with lines lw 1.5 lc rgb '#888888' dt 2 \\\n"
    "         title 'O(N^{-2}) fit'\n"
    ;
    gp.close();

    if (execute) {
        if (!runGnuplot(script)) {
            std::cout << "  gnuplot failed (brew install gnuplot)\n";
        }
    }
}

}
// ----------------------------------- main ------------------------------------
int run_validation_gs(int argc, char** argv) {
    using namespace gs;
    using clock_t = std::chrono::steady_clock;

    std::string outdir = "outputs/Grad-Shafranov/Validation";
    bool doSelftest = false;
    bool doPlot = true;

    for (int a=1; a<argc; ++a) {
        std::string s = argv[a];
        if (s == "--selftest") {
            doSelftest = true;
        } else if (s == "--noplot") {
            doPlot = false;
        } else if (s == "--outdir" && a+1<argc) {
            outdir = argv[++a];
        }
    }

    if (doSelftest) {
        std::cout << "=== Basis-function self-test ===\n";
        bool ok = selftest::checkBasisFunctions();
        std::cout << (ok ? "\nSELF-TEST PASSED\n\n" : "\nSELF-TEST FAILED\n\n");
        if (!ok) return 1;
    }

    mkdirRecursive(outdir + "/Solovev/data");
    mkdirRecursive(outdir + "/Solovev/plots/png");
    mkdirRecursive(outdir + "/Solovev/plots/plt");
    mkdirRecursive(outdir + "/Single-Null/data");
    mkdirRecursive(outdir + "/Single-Null/plots/png");
    mkdirRecursive(outdir + "/Single-Null/plots/plt");
    mkdirRecursive(outdir + "/Convergence/data");
    mkdirRecursive(outdir + "/Convergence/plots/png");
    mkdirRecursive(outdir + "/Convergence/plots/plt");

    std::cout << std::scientific << std::setprecision(4);

    struct TimingRow { std::string label; int iters; double msPerIter; double totalS; };
    std::vector<TimingRow> timingRows;

    const int nRepeatTiming = 20;
    auto recordTiming = [&](const std::string& label, const ValidationGSGrid& g,
                            const std::function<double(double,double)>& src,
                            const std::function<double(double,double)>& bc,
                            const Eigen::VectorXd& psiRef) {
        std::vector<double> times;
        times.reserve(nRepeatTiming);
        Eigen::VectorXd psiN;
        for (int r = 0; r < nRepeatTiming; ++r) {
            auto t0 = clock_t::now();
            psiN = g.solve(src, bc);
            auto t1 = clock_t::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(times.begin(), times.end());
        double medianMs = times[times.size() / 2];
        double err = maxErr(g, psiRef, bc);
        if (label == "Solovev") {
            std::cout << "[grad-shafranov]         " << label << " Timing: 1 Picard Iteration\n"
                      << "[grad-shafranov]                         " << medianMs << "\n"
                      << "[grad-shafranov]                         " << medianMs / 1000.0 << " s Total\n";
        } else if (label == "Single-null") {
            std::cout << "[grad-shafranov]     " << label << " Timing: 1 Picard Iteration\n"
                      << "[grad-shafranov]                         " << medianMs << "\n"
                      << "[grad-shafranov]                         " << medianMs / 1000.0 << " s Total\n";
        }
        timingRows.push_back({label, 1, medianMs, medianMs / 1000.0});
    };

    // Case 1: Solov'ev
    double solovevLcfsLevel = 0.09, solovevColorLo = 0.0, solovevColorHi = 0.0;
    double solovevDomainMin = 0.0, solovevDomainMax = 0.0;
    std::cout << "\n============================= Solov'ev Equilibrium =============================\n\n";
    {
        SolovevCase sol;
        ValidationGSGrid g(0.1, 1.4, -1.3, 1.3, 256, 256);
        auto src = sol.sourceFunc();
        auto bc = [&](double R, double Z)
        {
            return sol.psiAnalytical(R, Z);
        };

        auto psiN = g.solve(src, bc);
        solovevDomainMin = psiN.minCoeff();
        solovevDomainMax = psiN.maxCoeff();
        std::cout << "[grad-shafranov]            max|error| = " << maxErr(g, psiN, bc) << " T/m²\n"
                  << "[grad-shafranov]                    c0 = " << sol.c0() 
                  << "\n[grad-shafranov]                    c1 = " << sol.c1() << "\n";

        recordTiming("Solovev", g, src, bc, psiN);

        int axisIdx = findAxisIndex(g, psiN);
        const double psiLCFS = 0.09;
        const int nFineContours = 10;
        solovevColorLo = psiN(axisIdx);
        const double dPsi = (psiLCFS - solovevColorLo) / (nFineContours - 1);
        solovevLcfsLevel = psiLCFS + dPsi;
        solovevColorHi = solovevLcfsLevel;
        auto mask = buildInsideMask(g, psiN, axisIdx, solovevLcfsLevel);
        std::cout << "[grad-shafranov]  Magnetic Axis (R, Z) = (" << g.R(axisIdx % g.nR())
                  << ", " << g.Z(axisIdx / g.nR()) << ")"
                  << "\n[grad-shafranov]            LCFS Level = " << solovevLcfsLevel << " T/m²\n";

        writeFieldCSV(outdir+"/Solovev/data/Solovev_Field.csv", g, psiN, bc);
        writeMaskedCSV(outdir+"/Solovev/data/Solovev_FieldInside.csv", g, psiN, bc, mask);
        writeAxisCSV(outdir+"/Solovev/data/Solovev_Axis.csv", g, psiN, bc);
    }

    // Case 2: Single-null
    double singleNullLcfsLevel = 0.0, singleNullColorLo = 0.0, singleNullColorHi = 0.0;
    double singleNullDomainMin = 0.0, singleNullDomainMax = 0.0;
    std::cout << "\n=========================== Single-null Equilibrium ============================\n\n";
    {
        SingleNullCase sn;
        ValidationGSGrid g(0.5, 1.5, -1.0, 1.0, 256, 256);
        auto src = sn.sourceFunc();
        auto bc = [&](double R, double Z)
        {
            return sn.psiAnalytical(R, Z);
        };
        auto psiN = g.solve(src, bc);
        singleNullDomainMin = psiN.minCoeff();
        singleNullDomainMax = psiN.maxCoeff();
        std::cout << "[grad-shafranov]            max|error| = " << maxErr(g, psiN, bc) << " T/m²\n"
                  << "[grad-shafranov]    Coefficients c1-c12:\n\n"
                  << sn.coefficients().transpose().transpose() << "\n\n";

        recordTiming("Single-null", g, src, bc, psiN);

        double R_X = sn.xPointR(), Z_X = sn.xPointZ();
        singleNullLcfsLevel = sn.psiAnalytical(R_X,Z_X);
        int axisIdx = findAxisIndex(g, psiN);
        auto mask = buildInsideMask(g, psiN, axisIdx, singleNullLcfsLevel);

        singleNullColorLo = psiN(axisIdx);
        singleNullColorHi = singleNullLcfsLevel;

        std::cout << "[grad-shafranov]  Magnetic Axis (R, Z) = (" << g.R(axisIdx % g.nR())
                  << ", " << g.Z(axisIdx / g.nR()) << ")"
                  << "\n[grad-shafranov]        X-point (R, Z) = (" << R_X << ", " << Z_X << ")"
                  << "\n[grad-shafranov]            LCFS Level = " << singleNullLcfsLevel << " T/m²\n";

        writeFieldCSV(outdir+"/Single-Null/data/Single-Null_Field.csv", g, psiN, bc);
        writeMaskedCSV(outdir+"/Single-Null/data/Single-Null_FieldInside.csv", g, psiN, bc, mask);
        writeAxisCSV(outdir+"/Single-Null/data/Single-Null_Axis.csv", g, psiN, bc);
    }

    // ------------------------------------------------------------------
    // Grid Convergence
    // ------------------------------------------------------------------
    const std::string convSol = outdir+"/Convergence/data/Conv_Solovev.csv";
    const std::string convSn  = outdir+"/Convergence/data/Conv_Single-Null.csv";

    std::cout << "\n=============================== Grid Convergence ===============================\n";
    std::cout << "\n              --------------------- Solov'ev -----------------------\n\n";
    {
        SolovevCase sol;
        std::ofstream csv(convSol); csv << "N,max_error\n";
        double prev=-1.0;
        for (int n:{32,64,128,256,512}) {
            ValidationGSGrid g(0.1,1.4,-1.3,1.3,n,n);
            double e=maxErr(g,g.solve(sol.sourceFunc(),sol.bcFunc()),sol.bcFunc());
            csv<<n<<","<<e<<"\n";
            std::cout<<"[grad-shafranov] N = "<<n<<"  err = "<<e;
            if (prev>0) std::cout<<"  ratio = "<<prev/e;
            std::cout<<"\n"; prev=e;
        }
    }

    std::cout << "\n              -------------------- Single-Null ---------------------\n\n";
    {
        SingleNullCase sn;
        std::ofstream csv(convSn); csv << "N,max_error\n";
        double prev=-1.0;
        for (int n:{32,64,128,256,512}) {
            ValidationGSGrid g(0.5,1.5,-1.0,1.0,n,n);
            double e=maxErr(g,g.solve(sn.sourceFunc(),sn.bcFunc()),sn.bcFunc());
            csv<<n<<","<<e<<"\n";
            std::cout<<"[grad-shafranov] N = "<<n<<"  err = "<<e;
            if (prev>0) std::cout<<"  ratio = "<<prev/e;
            std::cout<<"\n"; prev=e;
        }
    }

    // ------------------------------------------------------------------
    // Generate and run gnuplot scripts
    // ------------------------------------------------------------------
    SolovevCase solForLevels;
    std::vector<double> solovevBgLevels;
    {
        double Rtip = 1.4;
        for (double R = solForLevels.R0(); R <= 1.4; R += 1e-4) {
            if (solForLevels.psiAnalytical(R, 0.0) > solovevColorHi) { Rtip = R; break; }
        }
        const int nBg = 5;
        for (int k = 1; k <= nBg; ++k) {
            double t = static_cast<double>(k) / nBg;
            double R = Rtip + (1.4 - Rtip) * t;
            solovevBgLevels.push_back(solForLevels.psiAnalytical(R, 0.0));
        }
    }

    writeFieldPlotScript(outdir, "Solovev",
                        outdir+"/Solovev/data/Solovev_Field.csv",
                        outdir+"/Solovev/data/Solovev_Axis.csv",
                        0.1, 1.4, -1.3, 1.3, solovevLcfsLevel,
                        solovevColorLo, solovevColorHi,
                        solovevDomainMin, solovevDomainMax,
                        static_cast<int>(solovevBgLevels.size()),
                        7,
                        &solovevBgLevels, doPlot);
    writeFieldPlotScript(outdir, "Single-Null",
                        outdir+"/Single-Null/data/Single-Null_Field.csv",
                        outdir+"/Single-Null/data/Single-Null_Axis.csv",
                        0.5, 1.5, -1.0, 1.0, singleNullLcfsLevel,
                        singleNullColorLo, singleNullColorHi,
                        singleNullDomainMin, singleNullDomainMax,
                        200,
                        0,
                        nullptr, doPlot);
    writeConvergenceScript(outdir, "Solovev",     convSol, doPlot);
    writeConvergenceScript(outdir, "Single-Null", convSn,  doPlot);

    if (!doPlot)
        std::cout << "  Plotting skipped. Run .plt scripts manually:\n"
                  << "    gnuplot " << outdir << "/Solovev/plots/plt/plot_Solovev.plt\n"
                  << "    gnuplot " << outdir << "/Single-Null/plots/plt/plot_Single-Null.plt\n";

    std::cout << "\n";
    std::cout << "======================================================================\n";
    std::cout << " Grad-Shafranov Picard Performance\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::left  << std::setw(14) << " Case"
              << std::right << std::setw(12) << "Iterations"
              << std::setw(18) << "Time/iter [ms]"
              << std::setw(16) << "Total time [s]" << "\n";
    std::cout << "----------------------------------------------------------------------\n";
    for (const auto& row : timingRows) {
        std::cout << std::left  << std::setw(14) << (" " + row.label)
                  << std::right << std::setw(12) << row.iters
                  << std::fixed << std::setprecision(2)
                  << std::setw(18) << row.msPerIter
                  << std::setprecision(4)
                  << std::setw(16) << row.totalS << "\n";
    }
    std::cout << "----------------------------------------------------------------------\n\n";

    std::cerr << "[grad-shafranov] Wrote data to " << outdir << "/Convergence/data\n";
    std::cerr << "[grad-shafranov] Wrote data to " << outdir << "/Solovev/data\n";
    std::cerr << "[grad-shafranov] Wrote data to " << outdir << "/Single-Null/data\n";
    std::cerr << "[grad-shafranov] Wrote plots to " << outdir << "/Convergence/plots/png\n";
    std::cerr << "[grad-shafranov] Wrote plots to " << outdir << "/Solovev/plots/png\n";
    std::cerr << "[grad-shafranov] Wrote plots to " << outdir << "/Single-Null/plots/png\n";

    std::cout << "\nDone.\n";
    return 0;
}