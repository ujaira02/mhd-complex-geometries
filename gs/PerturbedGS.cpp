// =============================================================================
// PerturbedGS.cpp — Perturbation Grad-Shafranov solver
// =============================================================================
// BUILD
//   g++ -std=c++17 -O3 -fopenmp -I /path/to/eigen3 PerturbedGS.cpp -o PerturbedGS
//   (on macOS/Homebrew use -I /opt/homebrew/opt/eigen/include/eigen3)
// 
// RUN
//   ./PerturbedGS --machine st40
//   Useful flags:
//     --mhd-nx N --mhd-ny M   MHD base mesh (default 256 x 512; the thesis
//                             effective resolution is 512 x 1024)
//     --cfl C                 CFL number (default 0.8, as in the thesis)
//     --tfinal T              stop earlier than 0.279
//     --sdf PATH              vessel SDF (default sdf/ST40/st40.dat)
// =============================================================================
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "PerturbedGS.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

static constexpr double MU0 = 4.0e-7 * M_PI;

// ---------------------------------- helpers ----------------------------------
static inline int syscmd(const std::string& cmd) {
    return std::system(cmd.c_str());
}

static inline void mkdir_p(const std::string& p) {
    syscmd("mkdir -p \"" + p + "\"");
}

static inline bool cmd_exists(const std::string& n) {
    return syscmd("command -v " + n + " >/dev/null 2>&1") == 0;
}

// ------------------------ complete elliptic integrals ------------------------
static void ellipKE(double m, double& K, double& E) {
    m = std::min(std::max(m, 0.0), 1.0 - 1e-12);
    double a = 1.0, b = std::sqrt(1.0 - m), c = std::sqrt(m);
    double sum = 0.5 * c * c;
    double pow2 = 1.0;
    for (int it = 0; it < 60 && c > 1e-15; ++it) {
        const double an = 0.5 * (a + b);
        const double bn = std::sqrt(a * b);
        c = 0.5 * (a - b);
        a = an; b = bn;
        pow2 *= 2.0;
        sum += 0.5 * pow2 * c * c;
    }
    K = M_PI / (2.0 * a);
    E = K * (1.0 - sum);
}

static double psiGreen(double R, double Z, double Rc, double Zc) {
    if (R < 1e-10 || Rc < 1e-10) return 0.0;
    const double d2 = (R + Rc) * (R + Rc) + (Z - Zc) * (Z - Zc);
    double m = 4.0 * R * Rc / d2;
    m = std::min(m, 1.0 - 1e-12);
    double K, E;
    ellipKE(m, K, E);
    const double k = std::sqrt(m);
    return MU0 / (2.0 * M_PI) * std::sqrt(R * Rc) / k * ((2.0 - m) * K - 2.0 * E);
}

static double brGreen(double R, double Z, double Rc, double Zc, double h) {
    return -(psiGreen(R, Z + h, Rc, Zc) - psiGreen(R, Z - h, Rc, Zc)) / (2.0 * h * R);
}

static double bzGreen(double R, double Z, double Rc, double Zc, double h) {
    return  (psiGreen(R + h, Z, Rc, Zc) - psiGreen(R - h, Z, Rc, Zc)) / (2.0 * h * R);
}

// ------------------------------------ SDF ------------------------------------
struct SdfTablePGS {
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0, dx = 0, dy = 0;
    int nx = 0, ny = 0;
    std::vector<double> sdf;
    bool loaded = false;

    bool load(const std::string& path, double scale, double dR, double dZ) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[grad-shafranov] Cannot open SDF: " << path << "\n";
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
        if (xs.empty()) {
            std::cerr << "[grad-shafranov] No data in SDF: " << path << "\n";
            return false;
        }

        xmin = *std::min_element(xs.begin(), xs.end());
        xmax = *std::max_element(xs.begin(), xs.end());
        ymin = *std::min_element(ys.begin(), ys.end());
        ymax = *std::max_element(ys.begin(), ys.end());

        nx = 0;
        const double y0r = ys[0];
        for (double yv : ys) {
            if (std::abs(yv - y0r) < 1e-12) {
                ++nx;
            }
            else break;
        }

        if (nx < 2) {
            std::cerr << "[grad-shafranov] Cannot determine SDF nx.\n";
            return false;
        }

        ny = static_cast<int>(xs.size()) / nx;
        if (nx * ny != static_cast<int>(xs.size())) {
            std::cerr << "[grad-shafranov] SDF size " << xs.size() << " not divisible by nx=" << nx << "\n";
            return false;
        }

        dx = (xmax - xmin) / (nx - 1);
        dy = (ymax - ymin) / (ny - 1);
        sdf = vs;

        xmin = xmin * scale + dR;  xmax = xmax * scale + dR;
        ymin = ymin * scale + dZ;  ymax = ymax * scale + dZ;
        dx *= scale;  dy *= scale;
        for (double& v : sdf) v *= scale;

        loaded = true;
        std::cout << "[grad-shafranov] SDF loaded: " << path 
                  << "\n[grad-shafranov] R[" << xmin << ", " << xmax 
                  << "]\n[grad-shafranov] Z[" << ymin << ", " << ymax << "]\n";
        return true;
    }

    double interp(double x, double y) const {
        if (!loaded) {
            return -1.0;
        }

        if (x < xmin - 0.5 * dx || x > xmax + 0.5 * dx || 
            y < ymin - 0.5 * dy || y > ymax + 0.5 * dy) {
            return 1.0;
        }

        const double fi = (x - xmin) / dx;
        const double fj = (y - ymin) / dy;
        const int i0 = std::max(0, std::min((int)std::floor(fi), nx - 2));
        const int j0 = std::max(0, std::min((int)std::floor(fj), ny - 2));
        const double tx = fi - i0, ty = fj - j0;
        const double v00 = sdf[(size_t)j0 * nx + i0];
        const double v10 = sdf[(size_t)j0 * nx + (i0 + 1)];
        const double v01 = sdf[(size_t)(j0 + 1) * nx + i0];
        const double v11 = sdf[(size_t)(j0 + 1) * nx + (i0 + 1)];
        return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11;
    }

    void normal(double x, double y, double h, double& nR, double& nZ) const {
        nR = (interp(x + h, y) - interp(x - h, y)) / (2.0 * h);
        nZ = (interp(x, y + h) - interp(x, y - h)) / (2.0 * h);
        const double n = std::hypot(nR, nZ);
        if (n > 1e-12) { nR /= n; nZ /= n; }
        else { nR = 1.0; nZ = 0.0; }
    }
};

// ---------------------------------- GS Grid ----------------------------------
class PerturbedGSGrid {
    public:
        PerturbedGSGrid(double Rmin, double Rmax, double Zmin, double Zmax, int nR, int nZ)
            : Rmin_(Rmin), Rmax_(Rmax), Zmin_(Zmin), Zmax_(Zmax), nR_(nR), nZ_(nZ) {
            if (nR_ < 3 || nZ_ < 3) {
                throw std::invalid_argument("PerturbedGSGrid: need at least 3 nodes per direction");
            }
            dR_ = (Rmax_ - Rmin_) / (nR_ - 1);
            dZ_ = (Zmax_ - Zmin_) / (nZ_ - 1);
        }

        int nR() const { return nR_; }
        int nZ() const { return nZ_; }
        double dR() const { return dR_; }
        double dZ() const { return dZ_; }
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
            const double dR2 = dR_ * dR_, dZ2 = dZ_ * dZ_;
            for (int j = 0; j < nZ_; ++j)
                for (int i = 0; i < nR_; ++i) {
                    const int row = idx(i, j);
                    if (isBoundary(i, j)) {
                        t.emplace_back(row, row, 1.0);
                        continue;
                    }
                    const double Ri = R(i);
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

        Eigen::VectorXd solveVec(const Eigen::VectorXd& b) const {
            ensureFactorized();
            Eigen::VectorXd psi = solver_.solve(b);
            if (solver_.info() != Eigen::Success) {
                throw std::runtime_error("PerturbedGSGrid: linear solve failed");
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
                throw std::runtime_error("PerturbedGSGrid: SparseLU factorization failed");
            }
            factorized_ = true;
        }
};

// ------------------------------ machine data ---------------------------------
struct Filament { double R, Z, w; };
struct Circuit { std::string name; std::vector<Filament> fil; };
struct IsoPair { double R1, Z1, R2, Z2; };

struct Machine {
    std::string name;
    std::string sdfDefault;
    double Rmin, Rmax, Zmin, Zmax;
    int nRdef, nZdef;
    double sdfScale = 1.0;
    double sdfDR = 0.0;
    double sdfDZ = 0.0;
    std::vector<Circuit> circuits;
    std::vector<std::pair<double, double>> xpoints;
    std::vector<IsoPair> isoflux;
    double Ip, p0, g0;
    double alpha_m = 1.0;
    double alpha_n = 1.0;
    double axR, axZ, axSigma;
    double sliceR1, sliceR2;
    std::vector<double> refCurrents;
    double mhdRmin = 0, mhdRmax = 0, mhdZmin = 0, mhdZmax = 0;
};

static Circuit solenoid(const std::string& name, double R,
                        double Zlo, double Zhi, double turns, int nFil = 24) {
    Circuit c; c.name = name;
    for (int k = 0; k < nFil; ++k) {
        const double t = (k + 0.5) / nFil;
        c.fil.push_back({R, Zlo + t * (Zhi - Zlo), turns / nFil});
    }
    return c;
}

static Circuit coil(const std::string& name, double R, double Z, double turns = 1.0) {
    Circuit c; c.name = name; c.fil.push_back({R, Z, turns}); return c;
}

static Machine makeMAST() {
    Machine m;
    m.name = "MAST";
    m.sdfDefault = "sdf/MAST/mast.dat";
    m.Rmin = 0;  m.Rmax = 2.5;  m.Zmin = -2.5;  m.Zmax = 2.5;
    m.nRdef = 257; m.nZdef = 257;
    m.circuits = {
        coil("P2U", 0.49,  1.76),  coil("P2L", 0.49, -1.76),
        coil("P3U", 1.10,  1.10),  coil("P3L", 1.10, -1.10),
        coil("P4U", 1.51,  1.095), coil("P4L", 1.51, -1.095),
        coil("P5U", 1.66,  0.52),  coil("P5L", 1.66, -0.52),
        coil("P6U", 1.50,  0.90),  coil("P6L", 1.50, -0.90),
        solenoid("Solenoid", 0.15, -1.4, 1.4, 100.0)
    };
    m.refCurrents = {
        19.307, 19.307,
        -5.088, -5.088,
        -36.169, -36.169,
        -116.204, -116.204,
        -59.144, -59.144,
        -119.299
    };
    m.xpoints = {{0.7, -1.1}, {0.7, 1.1}};
    m.isoflux = {{0.7, 1.1, 1.45, 0.0}, {0.7, -1.1, 1.45, 0.0}};
    m.Ip = 7.0e5;  m.p0 = 3.0e3;  m.g0 = 0.2;
    m.alpha_m = 1.0; m.alpha_n = 1.0;
    m.axR = 0.95; m.axZ = 0.0; m.axSigma = 0.35;
    m.sliceR1 = 0.5; m.sliceR2 = 1.7;
    return m;
}

static Machine makeITER() {
    Machine m;
    m.name = "ITER";
    m.sdfDefault = "sdf/ITER/iter.dat";
    m.sdfScale = 1.00; m.sdfDR = 0.30; m.sdfDZ = 0.00;
    m.Rmin = 1;  m.Rmax = 10;  m.Zmin = -7;  m.Zmax = 7;
    m.nRdef = 257; m.nZdef = 513;
    m.circuits = {
        coil("PF1",  3.9431,  7.5741),
        coil("PF2",  8.2851,  6.5398),
        coil("PF3", 11.9919,  3.2752),
        coil("PF4", 11.9630, -2.2336),
        coil("PF5",  8.3908, -6.7269),
        coil("PF6",  4.3340, -7.4665),
        solenoid("Solenoid", 1.696, -6.475, 6.495, 3318.0, 32)
    };
    m.refCurrents = {
        -635.0,
        2345.0,
        -11647.0,
        -6901.0,
        -1435.0,
        6478.0,
        -52.0
    };
    m.xpoints = {{4.9, -3.6}, {4.8, 4.7}};
    m.isoflux = {{4.9, -3.6, 7.0, 0.0}, {4.8, 4.7, 7.0, 0.0}};
    m.Ip = 15.0e6; m.p0 = 1.16864e6; m.g0 = 6.0;
    m.alpha_m = 1.0; m.alpha_n = 1.0;
    m.axR = 6.2; m.axZ = 0.4; m.axSigma = 1.6;
    m.sliceR1 = 2.5; m.sliceR2 = 7.5;
    return m;
}

static Machine makeST40() {
    Machine m;
    m.name = "ST40";
    m.sdfDefault = "sdf/ST40/st40.dat";
    m.Rmin = 0.0; m.Rmax = 1.85; m.Zmin = -1.25; m.Zmax = 1.25;
    m.nRdef = 257; m.nZdef = 513;
    m.circuits = {
        coil("P1U", 0.20,  0.825), coil("P1L", 0.20, -0.825),
        coil("P2U", 1.06,  1.020), coil("P2L", 1.06, -1.020),
        coil("P3U", 1.40,  0.260), coil("P3L", 1.40, -0.260),
        solenoid("Solenoid", 0.1, -0.675, 0.675, 200.0)
    };
    m.refCurrents = {
        17.97,
        17.97,
        -24.80,
        -24.80,
        -4.98,
        -4.98,
        -17.30
    };
    m.xpoints = {{0.342, -0.542}, {0.342, 0.542}};
    m.isoflux = {{0.342, -0.542, 0.721, 0.0}, {0.342, 0.542, 0.721, 0.0}};
    m.Ip = 1.0e6; m.p0 = 1.5e5; m.g0 = 1.21;
    m.alpha_m = 0.6; m.alpha_n = 1.06;
    m.axR = 0.48; m.axZ = 0.0; m.axSigma = 0.16;
    m.sliceR1 = 0.13; m.sliceR2 = 0.85;
    m.mhdRmin = 0.1; m.mhdRmax = 1.1; m.mhdZmin = -1.0; m.mhdZmax = 1.0;
    return m;
}

struct RZ { double R, Z; };

// ------------------------------ solver context -------------------------------
struct Solver {
    Machine machine;
    PerturbedGSGrid grid;
    SdfTablePGS sdf;

    std::vector<double> Rv, Zv;
    Eigen::VectorXd psiPl, psiCoil, psiTot;
    Eigen::VectorXd Jphi;
    std::vector<char> mask;
    Eigen::VectorXd Icoil;

    Eigen::MatrixXd Acon;
    Eigen::VectorXd rowScale;
    std::vector<std::function<double(const Eigen::VectorXd&)>> plasmaTerm;

    std::vector<Eigen::VectorXd> coilPsiGrid;

    double psiAxis = 0, psiBnd = 0, Raxis = 1, Zaxis = 0;
    double aCoef = 0, bCoef = 0;
    double Snorm = 1;
    std::vector<RZ> xpointsRefined;
    bool useDirectBoundary = false;

    double lambda = 0.3;
    Eigen::VectorXd dOld;

    Eigen::VectorXd computeBoundary() const {
        return useDirectBoundary ? boundaryValuesDirect() : boundaryValues();
    }

    Solver(const Machine& m, int nR, int nZ) : machine(m), grid(m.Rmin, m.Rmax, m.Zmin, m.Zmax, nR, nZ) {
        Rv.resize(nR); Zv.resize(nZ);
        for (int i = 0; i < nR; ++i) Rv[i] = grid.R(i);
        for (int j = 0; j < nZ; ++j) Zv[j] = grid.Z(j);
        psiPl  = Eigen::VectorXd::Zero(nR * nZ);
        psiCoil = psiPl; psiTot = psiPl; Jphi = psiPl;
        mask.assign(nR * nZ, 0);
        Icoil = Eigen::VectorXd::Zero((int)m.circuits.size());
        Snorm = shapeIntegral();
        dOld = Eigen::VectorXd::Zero(nR * nZ);
    }

    double jfunc(double x) const {
        if (x <= 0.0) return 1.0;
        if (x >= 1.0) return 0.0;
        return std::pow(1.0 - std::pow(x, machine.alpha_m), machine.alpha_n);
    }

    double shapeIntegral() const {
        const int n = 400; double s = 0.0;
        for (int k = 0; k < n; ++k) s += jfunc((k + 0.5) / n);
        return s / n;
    }

    double sample(const Eigen::VectorXd& f, double R, double Z) const {
        const int nR = grid.nR(), nZ = grid.nZ();
        double fi = (R - Rv[0]) / grid.dR();
        double fj = (Z - Zv[0]) / grid.dZ();
        fi = std::max(0.0, std::min(fi, (double)nR - 1.001));
        fj = std::max(0.0, std::min(fj, (double)nZ - 1.001));
        const int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
        const double tx = fi - i0, ty = fj - j0;
        return (1 - tx) * (1 - ty) * f(grid.idx(i0, j0))
             + tx * (1 - ty) * f(grid.idx(i0 + 1, j0))
             + (1 - tx) * ty * f(grid.idx(i0, j0 + 1))
             + tx * ty * f(grid.idx(i0 + 1, j0 + 1));
    }

    double sampleBR(const Eigen::VectorXd& f, double R, double Z) const {
        const double h = 0.5 * grid.dZ();
        const double Rd = std::max(R, 1e-6);
        return -(sample(f, R, Z + h) - sample(f, R, Z - h)) / (2.0 * h * Rd);
    }

    double sampleBZ(const Eigen::VectorXd& f, double R, double Z) const {
        const double h = 0.5 * grid.dR();
        const double Rd = std::max(R, 1e-6);
        return (sample(f, R + h, Z) - sample(f, R - h, Z)) / (2.0 * h * Rd);
    }

    void precomputeGreens() {
        const int nR = grid.nR(), nZ = grid.nZ(), nC = (int)machine.circuits.size();
        coilPsiGrid.assign(nC, Eigen::VectorXd::Zero(nR * nZ));
        for (int c = 0; c < nC; ++c) {
            for (const auto& fl : machine.circuits[c].fil) {
                for (int j = 0; j < nZ; ++j) {
                    for (int i = 0; i < nR; ++i) {
                        coilPsiGrid[c](grid.idx(i, j)) += fl.w * psiGreen(Rv[i], Zv[j], fl.R, fl.Z);
                    }
                }
            }
        }

        const double hB = 5e-5 * (machine.Rmax - machine.Rmin);
        const int nEq = 2 * (int)machine.xpoints.size() + (int)machine.isoflux.size();
        Acon = Eigen::MatrixXd::Zero(nEq, nC);
        plasmaTerm.clear();
        int r = 0;
        for (const auto& xp : machine.xpoints) {
            const double R = xp.first, Z = xp.second;
            for (int c = 0; c < nC; ++c)
                for (const auto& fl : machine.circuits[c].fil)
                    Acon(r, c) += fl.w * brGreen(R, Z, fl.R, fl.Z, hB);
            plasmaTerm.push_back([this, R, Z](const Eigen::VectorXd& p)
                                 { return -sampleBR(p, R, Z); });
            ++r;
            for (int c = 0; c < nC; ++c)
                for (const auto& fl : machine.circuits[c].fil)
                    Acon(r, c) += fl.w * bzGreen(R, Z, fl.R, fl.Z, hB);
            plasmaTerm.push_back([this, R, Z](const Eigen::VectorXd& p)
                                 { return -sampleBZ(p, R, Z); });
            ++r;
        }

        for (const auto& io : machine.isoflux) {
            for (int c = 0; c < nC; ++c)
                for (const auto& fl : machine.circuits[c].fil)
                    Acon(r, c) += fl.w * (psiGreen(io.R1, io.Z1, fl.R, fl.Z)
                                        - psiGreen(io.R2, io.Z2, fl.R, fl.Z));
            plasmaTerm.push_back([this, io](const Eigen::VectorXd& p)
                { return -(sample(p, io.R1, io.Z1) - sample(p, io.R2, io.Z2)); });
            ++r;
        }

        rowScale = Eigen::VectorXd::Ones(nEq);
        for (int q = 0; q < nEq; ++q) {
            const double n = Acon.row(q).norm();
            if (n > 0) { rowScale(q) = 1.0 / n; Acon.row(q) *= rowScale(q); }
        }
    }

    void solveCoilCurrentsCRATOS() {
        const int nEq = (int)plasmaTerm.size(), nC = (int)machine.circuits.size();
        Eigen::VectorXd b(nEq);
        for (int q = 0; q < nEq; ++q) b(q) = rowScale(q) * plasmaTerm[q](psiPl);

        Eigen::MatrixXd M = Acon.transpose() * Acon;
        Eigen::VectorXd rhs = Acon.transpose() * b;

        if (!machine.refCurrents.empty() && machine.refCurrents.size() == (size_t)nC) {
            std::vector<double> refA;
            for (size_t c = 0; c < machine.refCurrents.size(); ++c)
                refA.push_back(machine.refCurrents[c] * 1e3);

            double regWeight = 1e-6 * M.trace() / (nC * nC);
            if (machine.name == "MAST") regWeight *= 1.0;
            else if (machine.name == "ITER") regWeight *= 2.0;
            else if (machine.name == "ST40") regWeight *= 10.0;

            for (int c = 0; c < nC; ++c) {
                double coilWeight = 1.0;
                if (machine.name == "MAST") {
                    if (c == 2 || c == 3) coilWeight = 2.0;
                    if (c == 10) coilWeight = 0.5;
                }
                if (machine.name == "ITER") {
                    if (c == 0 || c == 4) coilWeight = 2.0;
                    if (c == 6) coilWeight = 0.5;
                }
                if (machine.name == "ST40") {
                    if (c == 2 || c == 3) coilWeight = 3.0;
                    if (c == 4 || c == 5) coilWeight = 3.0;
                    if (c == 6) coilWeight = 2.0;
                }
                M(c, c) += regWeight * coilWeight;
                rhs(c) += regWeight * coilWeight * refA[c];
            }
        }

        const double eps = 1e-12 * M.trace() / (nC * nC);
        M.diagonal().array() += eps;
        Icoil = M.ldlt().solve(rhs);
    }

    void buildCoilPsi() {
        psiCoil.setZero();
        for (int c = 0; c < (int)machine.circuits.size(); ++c)
            psiCoil += Icoil(c) * coilPsiGrid[c];
        psiTot = psiPl + psiCoil;
    }

    bool nearCoil(double R, double Z) const {
        const double d2min = std::pow(0.02 * (machine.Rmax - machine.Rmin), 2);
        for (const auto& c : machine.circuits)
            for (const auto& fl : c.fil) {
                const double d2 = (R - fl.R) * (R - fl.R) + (Z - fl.Z) * (Z - fl.Z);
                if (d2 < d2min) return true;
            }
        return false;
    }

    bool refineCriticalPoint(const Eigen::VectorXd& psi, double& r, double& z,
                             int maxit = 50, double tol = 1e-9) const {
        const double h = 0.5 * std::min(grid.dR(), grid.dZ());
        const double r0 = r, z0 = z;
        const double maxMove = 0.3 * std::max(machine.Rmax - machine.Rmin, machine.Zmax - machine.Zmin);
        for (int it = 0; it < maxit; ++it) {
            const double rc = std::max(Rv[1], std::min(r, Rv[grid.nR() - 2]));
            const double zc = std::max(Zv[1], std::min(z, Zv[grid.nZ() - 2]));
            const double c00 = sample(psi, rc, zc);
            const double Pr  = (sample(psi, rc + h, zc) - sample(psi, rc - h, zc)) / (2.0 * h);
            const double Pz  = (sample(psi, rc, zc + h) - sample(psi, rc, zc - h)) / (2.0 * h);
            const double Prr = (sample(psi, rc + h, zc) - 2.0 * c00 + sample(psi, rc - h, zc)) / (h * h);
            const double Pzz = (sample(psi, rc, zc + h) - 2.0 * c00 + sample(psi, rc, zc - h)) / (h * h);
            const double Prz = (sample(psi, rc + h, zc + h) - sample(psi, rc + h, zc - h)
                              - sample(psi, rc - h, zc + h) + sample(psi, rc - h, zc - h)) / (4.0 * h * h);
            const double D = Prr * Pzz - Prz * Prz;
            if (std::hypot(Pr, Pz) < tol) { r = rc; z = zc; return true; }
            if (std::abs(D) < 1e-300) { r = rc; z = zc; return false; }
            const double dr = (-Pzz * Pr + Prz * Pz) / D;
            const double dz = ( Prz * Pr + Prr * Pz) / D;
            r = rc + dr; z = zc + dz;
            if (std::hypot(r - r0, z - z0) > maxMove) { r = r0; z = z0; return false; }
        }
        return std::hypot(r - r0, z - z0) <= maxMove;
    }

    bool findAxis(double bboxPad = 0.10) {
        double Rlo = 1e30, Rhi = -1e30, Zlo = 1e30, Zhi = -1e30;
        auto acc = [&](double R, double Z) {
            Rlo = std::min(Rlo, R); Rhi = std::max(Rhi, R);
            Zlo = std::min(Zlo, Z); Zhi = std::max(Zhi, Z);
        };
        for (auto& xp : machine.xpoints) acc(xp.first, xp.second);
        for (auto& io : machine.isoflux) { acc(io.R1, io.Z1); acc(io.R2, io.Z2); }
        const double mR = bboxPad * (Rhi - Rlo), mZ = bboxPad * (Zhi - Zlo);
        Rlo -= mR; Rhi += mR; Zlo -= mZ; Zhi += mZ;
        Rlo = std::max(Rlo, machine.Rmin); Rhi = std::min(Rhi, machine.Rmax);
        Zlo = std::max(Zlo, machine.Zmin); Zhi = std::min(Zhi, machine.Zmax);

        const double sgn = (machine.Ip >= 0.0) ? 1.0 : -1.0;
        double best = -1e300; int bi = -1, bj = -1;
        for (int j = 2; j < grid.nZ() - 2; ++j)
            for (int i = 2; i < grid.nR() - 2; ++i) {
                const double R = Rv[i], Z = Zv[j];
                if (R < Rlo || R > Rhi || Z < Zlo || Z > Zhi) continue;
                if (sdf.interp(R, Z) > 0.0) continue;
                if (nearCoil(R, Z)) continue;
                const double v = sgn * psiTot(grid.idx(i, j));
                if (v <= sgn * psiTot(grid.idx(i + 1, j)) ||
                    v <= sgn * psiTot(grid.idx(i - 1, j)) ||
                    v <= sgn * psiTot(grid.idx(i, j + 1)) ||
                    v <= sgn * psiTot(grid.idx(i, j - 1))) continue;
                if (v > best) { best = v; bi = i; bj = j; }
            }
        if (bi < 0) return false;
        Raxis = Rv[bi]; Zaxis = Zv[bj];
        refineCriticalPoint(psiTot, Raxis, Zaxis);
        psiAxis = sample(psiTot, Raxis, Zaxis);

        xpointsRefined.assign(machine.xpoints.size(), {0.0, 0.0});
        double bestD = 1e300;
        for (size_t s = 0; s < machine.xpoints.size(); ++s) {
            double R = machine.xpoints[s].first, Z = machine.xpoints[s].second;
            refineCriticalPoint(psiTot, R, Z);
            xpointsRefined[s] = {R, Z};
            const double v = sample(psiTot, R, Z);
            const double d = std::abs(v - psiAxis);
            if (d < bestD) { bestD = d; psiBnd = v; }
        }
        return std::abs(psiBnd - psiAxis) > 1e-14;
    }

    void buildMask() {
        const int nR = grid.nR(), nZ = grid.nZ();
        std::fill(mask.begin(), mask.end(), 0);
        const double den = psiBnd - psiAxis;
        auto psibar = [&](int i, int j)
            { return (psiTot(grid.idx(i, j)) - psiAxis) / den; };

        int ai = (int)std::round((Raxis - Rv[0]) / grid.dR());
        int aj = (int)std::round((Zaxis - Zv[0]) / grid.dZ());
        std::deque<std::pair<int, int>> q;
        q.push_back({ai, aj});
        mask[grid.idx(ai, aj)] = 1;
        while (!q.empty()) {
            auto [i, j] = q.front(); q.pop_front();
            const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
            for (int k = 0; k < 4; ++k) {
                const int ii = i + di[k], jj = j + dj[k];
                if (ii <= 0 || ii >= nR - 1 || jj <= 0 || jj >= nZ - 1) continue;
                const int id = grid.idx(ii, jj);
                if (mask[id]) continue;
                if (psibar(ii, jj) >= 1.0) continue;
                if (sdf.interp(Rv[ii], Zv[jj]) > 0.0) continue;
                mask[id] = 1;
                q.push_back({ii, jj});
            }
        }
    }

    void buildJphi() {
        const int nR = grid.nR(), nZ = grid.nZ();
        const double dA = grid.dR() * grid.dZ();
        const double den = psiBnd - psiAxis;

        double IR = 0.0, I1R = 0.0;
        for (int j = 0; j < nZ; ++j)
            for (int i = 0; i < nR; ++i) {
                const int id = grid.idx(i, j);
                if (!mask[id]) continue;
                const double x = (psiTot(id) - psiAxis) / den;
                const double jf = jfunc(x);
                IR  += (Rv[i] / Raxis) * jf * dA;
                I1R += (Raxis / Rv[i]) * jf * dA;
            }
        if (IR < 1e-14 || I1R < 1e-14)
            throw std::runtime_error("empty plasma mask — no confined region found");

        aCoef = -machine.p0 * Raxis / (den * Snorm);
        bCoef = (machine.Ip - aCoef * IR) / I1R;

        Jphi.setZero();
        for (int j = 0; j < nZ; ++j)
            for (int i = 0; i < nR; ++i) {
                const int id = grid.idx(i, j);
                if (!mask[id]) continue;
                const double x = (psiTot(id) - psiAxis) / den;
                Jphi(id) = (aCoef * Rv[i] / Raxis + bCoef * Raxis / Rv[i]) * jfunc(x);
            }
    }

    Eigen::VectorXd boundaryValues() const {
        const int nR = grid.nR(), nZ = grid.nZ();
        const Eigen::VectorXd U = solvePlasma(Eigen::VectorXd::Zero(nR * nZ));

        struct BPt { int i, j; double dUdn, dl; };
        std::vector<BPt> bpts;
        bpts.reserve(2 * (nR + nZ));
        const double dR = grid.dR(), dZ = grid.dZ();

        for (int i = 0; i < nR; ++i) {
            {
                const double U1 = U(grid.idx(i, 1)), U2 = U(grid.idx(i, 2));
                const double dUdn = (U2 - 4.0 * U1) / (2.0 * dZ);
                bpts.push_back({i, 0, dUdn, dR});
            }
            {
                const double U1 = U(grid.idx(i, nZ - 2)), U2 = U(grid.idx(i, nZ - 3));
                const double dUdn = (U2 - 4.0 * U1) / (2.0 * dZ);
                bpts.push_back({i, nZ - 1, dUdn, dR});
            }
        }
        for (int j = 1; j < nZ - 1; ++j) {
            {
                const double U1 = U(grid.idx(1, j)), U2 = U(grid.idx(2, j));
                const double dUdn = (U2 - 4.0 * U1) / (2.0 * dR);
                bpts.push_back({0, j, dUdn, dZ});
            }
            {
                const double U1 = U(grid.idx(nR - 2, j)), U2 = U(grid.idx(nR - 3, j));
                const double dUdn = (U2 - 4.0 * U1) / (2.0 * dR);
                bpts.push_back({nR - 1, j, dUdn, dZ});
            }
        }

        const double axisEps = 1e-9;
        std::vector<BPt> bsrc;
        bsrc.reserve(bpts.size());
        for (const auto& bp : bpts)
            if (Rv[bp.i] > axisEps) bsrc.push_back(bp);

        Eigen::VectorXd bnd = Eigen::VectorXd::Zero(nR * nZ);
        for (int i = 0; i < nR; ++i) {
            for (int j : {0, nZ - 1}) {
                if (Rv[i] <= axisEps) { bnd(grid.idx(i, j)) = 0.0; continue; }
                double s = 0.0;
                for (const auto& bp : bsrc)
                    s += psiGreen(Rv[i], Zv[j], Rv[bp.i], Zv[bp.j]) / Rv[bp.i] * bp.dUdn * bp.dl;
                bnd(grid.idx(i, j)) = -s / MU0;
            }
        }
        for (int j = 1; j < nZ - 1; ++j) {
            for (int i : {0, nR - 1}) {
                if (Rv[i] <= axisEps) { bnd(grid.idx(i, j)) = 0.0; continue; }
                double s = 0.0;
                for (const auto& bp : bsrc)
                    s += psiGreen(Rv[i], Zv[j], Rv[bp.i], Zv[bp.j]) / Rv[bp.i] * bp.dUdn * bp.dl;
                bnd(grid.idx(i, j)) = -s / MU0;
            }
        }
        return bnd;
    }

    Eigen::VectorXd boundaryValuesDirect() const {
        const int nR = grid.nR(), nZ = grid.nZ();
        const double dA = grid.dR() * grid.dZ();
        std::vector<std::pair<int, int>> src;
        src.reserve(4096);
        for (int j = 0; j < nZ; ++j)
            for (int i = 0; i < nR; ++i)
                if (mask[grid.idx(i, j)] && std::abs(Jphi(grid.idx(i, j))) > 0.0)
                    src.push_back({i, j});

        Eigen::VectorXd bnd = Eigen::VectorXd::Zero(nR * nZ);
        auto accum = [&](int i, int j) {
            double s = 0.0;
            for (const auto& [is, js] : src)
                s += psiGreen(Rv[i], Zv[j], Rv[is], Zv[js])
                     * Jphi(grid.idx(is, js)) * dA;
            bnd(grid.idx(i, j)) = s;
        };
        for (int i = 0; i < nR; ++i) { accum(i, 0); accum(i, nZ - 1); }
        for (int j = 1; j < nZ - 1; ++j) { accum(0, j); accum(nR - 1, j); }
        return bnd;
    }

    Eigen::VectorXd solvePlasma(const Eigen::VectorXd& bnd) const {
        const int nR = grid.nR(), nZ = grid.nZ();
        Eigen::VectorXd rhs(nR * nZ);
        for (int j = 0; j < nZ; ++j)
            for (int i = 0; i < nR; ++i) {
                const int id = grid.idx(i, j);
                rhs(id) = grid.isBoundary(i, j) ? bnd(id) : -MU0 * Rv[i] * Jphi(id);
            }
        return grid.solveVec(rhs);
    }

    void initialise() {
        const int nR = grid.nR(), nZ = grid.nZ();
        const double dA = grid.dR() * grid.dZ();
        const double s2 = machine.axSigma * machine.axSigma;
        double sum = 0.0;
        for (int j = 0; j < nZ; ++j)
            for (int i = 0; i < nR; ++i) {
                const double g = std::exp(-((Rv[i] - machine.axR) * (Rv[i] - machine.axR)
                                          + (Zv[j] - machine.axZ) * (Zv[j] - machine.axZ)) / s2);
                Jphi(grid.idx(i, j)) = g;
                sum += g * dA;
            }
        Jphi *= machine.Ip / sum;
        std::fill(mask.begin(), mask.end(), 1);

        for (int j = 0; j < nZ; ++j)
            for (int i = 0; i < nR; ++i)
                if (sdf.interp(Rv[i], Zv[j]) > 0.0)
                    { Jphi(grid.idx(i, j)) = 0.0; mask[grid.idx(i, j)] = 0; }
        psiPl = solvePlasma(computeBoundary());
    }

    double Fint(double x) const {
        const int m = 200; double s = 0.0;
        for (int k = 0; k < m; ++k) s += jfunc(x + (1.0 - x) * (k + 0.5) / m);
        return s * (1.0 - x) / m;
    }

    double pressureSI(double psiVal) const {
        const double den = psiBnd - psiAxis;
        double x = (psiVal - psiAxis) / den;
        x = std::max(0.0, std::min(1.0, x));
        return -aCoef / Raxis * den * Fint(x);
    }

    double gfunSI(double psiVal) const {
        const double den = psiBnd - psiAxis;
        double x = (psiVal - psiAxis) / den;
        x = std::max(0.0, std::min(1.0, x));
        const double F2 = machine.g0 * machine.g0
                        - 2.0 * MU0 * Raxis * bCoef * den * Fint(x);
        return std::sqrt(std::max(0.0, F2));
    }

};

// -------------------------------- I/O helpers --------------------------------
static void writeVesselCSV(const std::string& path, const SdfTablePGS& s) {
    if (!s.loaded) return;
    std::ofstream out(path);
    out << std::setprecision(12) << "R,Z,phi\n";
    for (int j = 0; j < s.ny; ++j) {
        if (j > 0) out << "\n";
        for (int i = 0; i < s.nx; ++i)
            out << (s.xmin + i * s.dx) << "," << (s.ymin + j * s.dy)
                << "," << s.sdf[(size_t)j * s.nx + i] << "\n";
    }
}

static bool extractContourTable(const std::string& csv, const std::string& raw,
                                const std::string& comma, const std::string& spec) {
    {
        std::ofstream plt(raw + ".plt");
        plt << "reset\nset datafile separator ','\n"
           << "set table '" << raw << "'\n"
           << "set contour base\nset cntrparam levels " << spec << "\n"
           << "unset surface\n"
           << "splot '" << csv << "' using 1:2:3 with lines\nunset table\n";
    }
    if (syscmd("gnuplot \"" + raw + ".plt\" 2>&1") != 0) return false;
    std::ifstream in(raw);
    if (!in) return false;
    std::ofstream out(comma);
    std::string line;
    while (std::getline(in, line)) {
        const size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos || line[b] == '#') {
            out << "\n"; continue;
        }
        const size_t e = line.find_last_not_of(" \t\r\n");
        std::string t = line.substr(b, e - b + 1), csvLine;
        bool tok = false;
        for (char c : t) {
            if (c == ' ' || c == '\t') {
                if (tok) { csvLine += ','; tok = false; }
            } else {
                csvLine += c; tok = true;
            }
        }
        out << csvLine << "\n";
    }
    return true;
}


static double GAM = 5.0 / 3.0;
static constexpr int    NGH   = 2;
static constexpr double RHO_FLOOR = 1.0e-6;
static constexpr double P_FLOOR   = 1.0e-7;

struct Prim { double rho, uR, uZ, uP, p, BR, BZ, BP, glm; };
struct Cons { double rho, mR, mZ, mP, E, BR, BZ, BP, glm; };

static inline Cons operator+(const Cons& a, const Cons& b) {
    return {a.rho+b.rho, a.mR+b.mR, a.mZ+b.mZ, a.mP+b.mP, a.E+b.E,
            a.BR+b.BR, a.BZ+b.BZ, a.BP+b.BP, a.glm+b.glm};
}
static inline Cons operator-(const Cons& a, const Cons& b) {
    return {a.rho-b.rho, a.mR-b.mR, a.mZ-b.mZ, a.mP-b.mP, a.E-b.E,
            a.BR-b.BR, a.BZ-b.BZ, a.BP-b.BP, a.glm-b.glm};
}
static inline Cons operator*(double s, const Cons& a) {
    return {s*a.rho, s*a.mR, s*a.mZ, s*a.mP, s*a.E,
            s*a.BR, s*a.BZ, s*a.BP, s*a.glm};
}

static inline Cons prim2cons(const Prim& w) {
    Cons u;
    u.rho = w.rho;
    u.mR = w.rho * w.uR; u.mZ = w.rho * w.uZ; u.mP = w.rho * w.uP;
    const double v2 = w.uR*w.uR + w.uZ*w.uZ + w.uP*w.uP;
    const double B2 = w.BR*w.BR + w.BZ*w.BZ + w.BP*w.BP;
    u.E = w.p / (GAM - 1.0) + 0.5 * w.rho * v2 + 0.5 * B2;
    u.BR = w.BR; u.BZ = w.BZ; u.BP = w.BP; u.glm = w.glm;
    return u;
}

static inline Prim cons2prim(const Cons& u) {
    Prim w;
    w.rho = std::max(u.rho, RHO_FLOOR);
    w.uR = u.mR / w.rho; w.uZ = u.mZ / w.rho; w.uP = u.mP / w.rho;
    w.BR = u.BR; w.BZ = u.BZ; w.BP = u.BP; w.glm = u.glm;
    const double v2 = w.uR*w.uR + w.uZ*w.uZ + w.uP*w.uP;
    const double B2 = w.BR*w.BR + w.BZ*w.BZ + w.BP*w.BP;
    w.p = (GAM - 1.0) * (u.E - 0.5 * w.rho * v2 - 0.5 * B2);
    if (!(w.p > P_FLOOR)) w.p = P_FLOOR;
    return w;
}

static inline double cfast(const Prim& w, double Bn) {
    const double a2  = GAM * w.p / w.rho;
    const double B2  = w.BR*w.BR + w.BZ*w.BZ + w.BP*w.BP;
    const double vA2 = B2 / w.rho;
    const double vAn2 = Bn * Bn / w.rho;
    const double s = a2 + vA2;
    const double disc = std::max(0.0, s * s - 4.0 * a2 * vAn2);
    return std::sqrt(std::max(0.0, 0.5 * (s + std::sqrt(disc))));
}

static inline Cons physFlux(const Prim& w, int dir, double ch) {
    const double un = (dir == 0) ? w.uR : w.uZ;
    const double Bn = (dir == 0) ? w.BR : w.BZ;
    const double B2 = w.BR*w.BR + w.BZ*w.BZ + w.BP*w.BP;
    const double v2 = w.uR*w.uR + w.uZ*w.uZ + w.uP*w.uP;
    const double pT = w.p + 0.5 * B2;
    const double E  = w.p / (GAM - 1.0) + 0.5 * w.rho * v2 + 0.5 * B2;
    const double uB = w.uR*w.BR + w.uZ*w.BZ + w.uP*w.BP;
    Cons f;
    f.rho = w.rho * un;
    f.mR = w.rho * un * w.uR - Bn * w.BR + ((dir == 0) ? pT : 0.0);
    f.mZ = w.rho * un * w.uZ - Bn * w.BZ + ((dir == 1) ? pT : 0.0);
    f.mP = w.rho * un * w.uP - Bn * w.BP;
    f.E  = (E + pT) * un - Bn * uB;
    f.BR = (dir == 0) ? w.glm : (un * w.BR - w.uR * Bn);
    f.BZ = (dir == 1) ? w.glm : (un * w.BZ - w.uZ * Bn);
    f.BP = un * w.BP - w.uP * Bn;
    f.glm = ch * ch * Bn;
    return f;
}

static inline Cons geomSource(const Prim& w, double R, double ch) {
    const double invR = 1.0 / R;
    const double B2 = w.BR*w.BR + w.BZ*w.BZ + w.BP*w.BP;
    const double v2 = w.uR*w.uR + w.uZ*w.uZ + w.uP*w.uP;
    const double pT = w.p + 0.5 * B2;
    const double E  = w.p / (GAM - 1.0) + 0.5 * w.rho * v2 + 0.5 * B2;
    const double uB = w.uR*w.BR + w.uZ*w.BZ + w.uP*w.BP;
    Cons s;
    s.rho = -w.rho * w.uR * invR;
    s.mR  = (-(w.rho * w.uR * w.uR - w.BR * w.BR)
             + (w.rho * w.uP * w.uP - w.BP * w.BP)) * invR;
    s.mZ  = -(w.rho * w.uR * w.uZ - w.BR * w.BZ) * invR;
    s.mP  = -2.0 * (w.rho * w.uR * w.uP - w.BR * w.BP) * invR;
    s.E   = -((E + pT) * w.uR - w.BR * uB) * invR;
    s.BR  = 0.0;
    s.BZ  = -(w.uR * w.BZ - w.uZ * w.BR) * invR;
    s.BP  = 0.0;
    s.glm = -ch * ch * w.BR * invR;
    return s;
}

static Cons hll_flux(const Prim& L, const Prim& R, int dir, double ch,
                     double SL, double SR) {
    const Cons UL = prim2cons(L), UR = prim2cons(R);
    const Cons FL = physFlux(L, dir, ch), FR = physFlux(R, dir, ch);
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;
    return (1.0 / (SR - SL)) * ((SR * FL) - (SL * FR) + (SR * SL) * (UR - UL));
}

static Cons hlld_flux(Prim L, Prim R, int dir, double ch) {
    const double BnL = (dir == 0) ? L.BR : L.BZ;
    const double BnR = (dir == 0) ? R.BR : R.BZ;
    const double Bn   = 0.5 * (BnL + BnR) - 0.5 / ch * (R.glm - L.glm);
    const double glms = 0.5 * (L.glm + R.glm) - 0.5 * ch * (BnR - BnL);
    if (dir == 0) { L.BR = Bn; R.BR = Bn; } else { L.BZ = Bn; R.BZ = Bn; }
    L.glm = glms; R.glm = glms;

    const double unL = (dir == 0) ? L.uR : L.uZ;
    const double unR = (dir == 0) ? R.uR : R.uZ;
    const double ut1L = (dir == 0) ? L.uZ : L.uR;
    const double ut1R = (dir == 0) ? R.uZ : R.uR;
    const double ut2L = L.uP, ut2R = R.uP;
    const double Bt1L = (dir == 0) ? L.BZ : L.BR;
    const double Bt1R = (dir == 0) ? R.BZ : R.BR;
    const double Bt2L = L.BP, Bt2R = R.BP;

    const double cfL = cfast(L, Bn), cfR = cfast(R, Bn);
    const double SL = std::min(unL, unR) - std::max(cfL, cfR);
    const double SR = std::max(unL, unR) + std::max(cfL, cfR);

    const Cons FL = physFlux(L, dir, ch), FR = physFlux(R, dir, ch);
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    const double B2L = Bn*Bn + Bt1L*Bt1L + Bt2L*Bt2L;
    const double B2R = Bn*Bn + Bt1R*Bt1R + Bt2R*Bt2R;
    const double pTL = L.p + 0.5 * B2L;
    const double pTR = R.p + 0.5 * B2R;
    const double EL = L.p/(GAM-1.0) + 0.5*L.rho*(unL*unL+ut1L*ut1L+ut2L*ut2L) + 0.5*B2L;
    const double ER = R.p/(GAM-1.0) + 0.5*R.rho*(unR*unR+ut1R*ut1R+ut2R*ut2R) + 0.5*B2R;

    const double rl = L.rho * (SL - unL);
    const double rr = R.rho * (SR - unR);
    const double SM = (rr * unR - rl * unL - pTR + pTL) / (rr - rl);
    const double pTs = pTL + rl * (SM - unL);

    const double rhoLs = rl / (SL - SM);
    const double rhoRs = rr / (SR - SM);
    if (!(rhoLs > 0.0) || !(rhoRs > 0.0))
        return hll_flux(L, R, dir, ch, SL, SR);

    const double dL = rl * (SL - SM) - Bn * Bn;
    const double dR = rr * (SR - SM) - Bn * Bn;
    const double eps = 1e-10 * (std::abs(pTs) + Bn * Bn + 1e-30);

    double ut1Ls, ut2Ls, Bt1Ls, Bt2Ls, ut1Rs, ut2Rs, Bt1Rs, Bt2Rs;
    if (std::abs(dL) < eps) { ut1Ls = ut1L; ut2Ls = ut2L; Bt1Ls = Bt1L; Bt2Ls = Bt2L; }
    else {
        const double c1 = Bn * (SM - unL) / dL;
        const double c2 = (rl * (SL - unL) - Bn * Bn) / dL;
        ut1Ls = ut1L - Bt1L * c1;
        ut2Ls = ut2L - Bt2L * c1;
        Bt1Ls = Bt1L * c2;
        Bt2Ls = Bt2L * c2;
    }
    if (std::abs(dR) < eps) { ut1Rs = ut1R; ut2Rs = ut2R; Bt1Rs = Bt1R; Bt2Rs = Bt2R; }
    else {
        const double c1 = Bn * (SM - unR) / dR;
        const double c2 = (rr * (SR - unR) - Bn * Bn) / dR;
        ut1Rs = ut1R - Bt1R * c1;
        ut2Rs = ut2R - Bt2R * c1;
        Bt1Rs = Bt1R * c2;
        Bt2Rs = Bt2R * c2;
    }

    const double uBL = unL*Bn + ut1L*Bt1L + ut2L*Bt2L;
    const double uBR = unR*Bn + ut1R*Bt1R + ut2R*Bt2R;
    const double uBLs = SM*Bn + ut1Ls*Bt1Ls + ut2Ls*Bt2Ls;
    const double uBRs = SM*Bn + ut1Rs*Bt1Rs + ut2Rs*Bt2Rs;
    const double ELs = ((SL - unL) * EL - pTL * unL + pTs * SM + Bn * (uBL - uBLs)) / (SL - SM);
    const double ERs = ((SR - unR) * ER - pTR * unR + pTs * SM + Bn * (uBR - uBRs)) / (SR - SM);

    const double sqL = std::sqrt(rhoLs), sqR = std::sqrt(rhoRs);
    const double SLs = SM - std::abs(Bn) / sqL;
    const double SRs = SM + std::abs(Bn) / sqR;

    auto packU = [&](double rho, double un, double ut1, double ut2,
                     double bt1, double bt2, double e) {
        Cons u;
        u.rho = rho;
        if (dir == 0) { u.mR = rho * un; u.mZ = rho * ut1; }
        else          { u.mZ = rho * un; u.mR = rho * ut1; }
        u.mP = rho * ut2;
        u.E = e;
        if (dir == 0) { u.BR = Bn; u.BZ = bt1; }
        else          { u.BZ = Bn; u.BR = bt1; }
        u.BP = bt2;
        u.glm = glms;
        return u;
    };

    const Cons ULc = prim2cons(L), URc = prim2cons(R);
    const Cons ULs = packU(rhoLs, SM, ut1Ls, ut2Ls, Bt1Ls, Bt2Ls, ELs);
    const Cons URs = packU(rhoRs, SM, ut1Rs, ut2Rs, Bt1Rs, Bt2Rs, ERs);

    if (std::abs(Bn) < 1e-12 * std::sqrt(std::abs(pTs) + 1e-30)) {
        if (SM >= 0.0) return FL + SL * (ULs - ULc);
        else           return FR + SR * (URs - URc);
    }

    if (SLs >= 0.0) return FL + SL * (ULs - ULc);
    if (SRs <= 0.0) return FR + SR * (URs - URc);

    const double sgn = (Bn > 0.0) ? 1.0 : -1.0;
    const double idenom = 1.0 / (sqL + sqR);
    const double ut1ss = (sqL * ut1Ls + sqR * ut1Rs + (Bt1Rs - Bt1Ls) * sgn) * idenom;
    const double ut2ss = (sqL * ut2Ls + sqR * ut2Rs + (Bt2Rs - Bt2Ls) * sgn) * idenom;
    const double Bt1ss = (sqL * Bt1Rs + sqR * Bt1Ls + sqL * sqR * (ut1Rs - ut1Ls) * sgn) * idenom;
    const double Bt2ss = (sqL * Bt2Rs + sqR * Bt2Ls + sqL * sqR * (ut2Rs - ut2Ls) * sgn) * idenom;
    const double uBss = SM * Bn + ut1ss * Bt1ss + ut2ss * Bt2ss;
    const double ELss = ELs - sqL * (uBLs - uBss) * sgn;
    const double ERss = ERs + sqR * (uBRs - uBss) * sgn;

    const Cons ULss = packU(rhoLs, SM, ut1ss, ut2ss, Bt1ss, Bt2ss, ELss);
    const Cons URss = packU(rhoRs, SM, ut1ss, ut2ss, Bt1ss, Bt2ss, ERss);

    if (SM >= 0.0)
        return FL + SL * (ULs - ULc) + SLs * (ULss - ULs);
    else
        return FR + SR * (URs - URc) + SRs * (URss - URs);
}

// --------------------------- WENO3 reconstruction -----------------------------
static inline double weno3_right(double vm, double v0, double vp) {
    const double eps = 1e-6;
    const double b0 = (v0 - vm) * (v0 - vm);
    const double b1 = (vp - v0) * (vp - v0);
    const double a0 = (1.0 / 3.0) / ((eps + b0) * (eps + b0));
    const double a1 = (2.0 / 3.0) / ((eps + b1) * (eps + b1));
    const double w0 = a0 / (a0 + a1), w1 = a1 / (a0 + a1);
    const double p0 = -0.5 * vm + 1.5 * v0;
    const double p1 =  0.5 * v0 + 0.5 * vp;
    return w0 * p0 + w1 * p1;
}
static inline double weno3_left(double vm, double v0, double vp) {
    const double eps = 1e-6;
    const double b0 = (v0 - vm) * (v0 - vm);
    const double b1 = (vp - v0) * (vp - v0);
    const double a0 = (2.0 / 3.0) / ((eps + b0) * (eps + b0));
    const double a1 = (1.0 / 3.0) / ((eps + b1) * (eps + b1));
    const double w0 = a0 / (a0 + a1), w1 = a1 / (a0 + a1);
    const double p0 = 0.5 * vm + 0.5 * v0;
    const double p1 = 1.5 * v0 - 0.5 * vp;
    return w0 * p0 + w1 * p1;
}

struct MHDGrid {
    int nx = 0, ny = 0;
    double Rmin, Rmax, Zmin, Zmax, dx, dz;
    int sx = 0, sy = 0;

    enum CellType : char { FLUID = 0, GHOST = 1, SOLID = 2 };

    std::vector<Cons>  U;
    std::vector<Prim>  W;
    std::vector<Prim>  W0;
    std::vector<char>  type;
    std::vector<double> Psi;
    std::vector<double> phiLS;
    std::vector<double> nRls, nZls;
    const SdfTablePGS* sdf = nullptr;

    int id(int i, int j) const { return j * sx + i; }
    double Rc(int i) const { return Rmin + (i - NGH + 0.5) * dx; }
    double Zc(int j) const { return Zmin + (j - NGH + 0.5) * dz; }

    void setup(double Rlo, double Rhi, double Zlo, double Zhi,
               int NX, int NY, const SdfTablePGS& s) {
        nx = NX; ny = NY;
        Rmin = Rlo; Rmax = Rhi; Zmin = Zlo; Zmax = Zhi;
        dx = (Rmax - Rmin) / nx;
        dz = (Zmax - Zmin) / ny;
        sx = nx + 2 * NGH; sy = ny + 2 * NGH;
        const size_t N = (size_t)sx * sy;
        U.assign(N, Cons{}); W.assign(N, Prim{}); W0.assign(N, Prim{});
        Psi.assign(N, 0.0);
        phiLS.assign(N, 1.0); nRls.assign(N, 1.0); nZls.assign(N, 0.0);
        type.assign(N, SOLID);
        sdf = &s;

        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i) {
                const double R = Rc(i), Z = Zc(j);
                phiLS[id(i, j)] = s.interp(R, Z);
                double nR, nZ;
                s.normal(R, Z, std::max(dx, dz), nR, nZ);
                nRls[id(i, j)] = nR; nZls[id(i, j)] = nZ;
            }
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i)
                type[id(i, j)] = (phiLS[id(i, j)] <= 0.0) ? FLUID : SOLID;
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i) {
                if (type[id(i, j)] != SOLID) continue;
                bool nearFluid = false;
                for (int dj = -NGH; dj <= NGH && !nearFluid; ++dj)
                    for (int di = -NGH; di <= NGH && !nearFluid; ++di) {
                        const int ii = i + di, jj = j + dj;
                        if (ii < 0 || ii >= sx || jj < 0 || jj >= sy) continue;
                        if (type[id(ii, jj)] == FLUID) nearFluid = true;
                    }
                if (nearFluid) type[id(i, j)] = GHOST;
            }
    }

    bool sampleFluid(const std::vector<Prim>& F, double R, double Z, Prim& out) const {
        double fi = (R - (Rmin + 0.5 * dx)) / dx + NGH;
        double fj = (Z - (Zmin + 0.5 * dz)) / dz + NGH;
        int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
        i0 = std::max(0, std::min(i0, sx - 2));
        j0 = std::max(0, std::min(j0, sy - 2));
        const double tx = std::max(0.0, std::min(1.0, fi - i0));
        const double ty = std::max(0.0, std::min(1.0, fj - j0));
        const int ids[4] = {id(i0, j0), id(i0 + 1, j0), id(i0, j0 + 1), id(i0 + 1, j0 + 1)};
        double wts[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
        double wsum = 0.0;
        for (int k = 0; k < 4; ++k) {
            if (type[ids[k]] != FLUID) wts[k] = 0.0;
            wsum += wts[k];
        }
        if (wsum < 1e-12) {
            double best = 1e300; int bid = -1;
            for (int dj = -2; dj <= 2; ++dj)
                for (int di = -2; di <= 2; ++di) {
                    const int ii = i0 + di, jj = j0 + dj;
                    if (ii < 0 || ii >= sx || jj < 0 || jj >= sy) continue;
                    if (type[id(ii, jj)] != FLUID) continue;
                    const double d2 = std::pow(Rc(ii) - R, 2) + std::pow(Zc(jj) - Z, 2);
                    if (d2 < best) { best = d2; bid = id(ii, jj); }
                }
            if (bid < 0) return false;
            out = F[bid];
            return true;
        }
        auto lerp = [&](auto get) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) v += wts[k] * get(F[ids[k]]);
            return v / wsum;
        };
        out.rho = lerp([](const Prim& w) { return w.rho; });
        out.uR  = lerp([](const Prim& w) { return w.uR;  });
        out.uZ  = lerp([](const Prim& w) { return w.uZ;  });
        out.uP  = lerp([](const Prim& w) { return w.uP;  });
        out.p   = lerp([](const Prim& w) { return w.p;   });
        out.BR  = lerp([](const Prim& w) { return w.BR;  });
        out.BZ  = lerp([](const Prim& w) { return w.BZ;  });
        out.BP  = lerp([](const Prim& w) { return w.BP;  });
        out.glm = lerp([](const Prim& w) { return w.glm; });
        return true;
    }

    Prim sampleAny(const std::vector<Prim>& F, double R, double Z) const {
        double fi = (R - (Rmin + 0.5 * dx)) / dx + NGH;
        double fj = (Z - (Zmin + 0.5 * dz)) / dz + NGH;
        int i0 = (int)std::floor(fi), j0 = (int)std::floor(fj);
        i0 = std::max(0, std::min(i0, sx - 2));
        j0 = std::max(0, std::min(j0, sy - 2));
        const double tx = std::max(0.0, std::min(1.0, fi - i0));
        const double ty = std::max(0.0, std::min(1.0, fj - j0));
        const Prim& a = F[id(i0, j0)];     const Prim& b = F[id(i0 + 1, j0)];
        const Prim& c = F[id(i0, j0 + 1)]; const Prim& d = F[id(i0 + 1, j0 + 1)];
        auto L = [&](double va, double vb, double vc, double vd) {
            return (1 - tx) * (1 - ty) * va + tx * (1 - ty) * vb
                 + (1 - tx) * ty * vc + tx * ty * vd;
        };
        Prim o;
        o.rho = L(a.rho, b.rho, c.rho, d.rho);
        o.uR = L(a.uR, b.uR, c.uR, d.uR);
        o.uZ = L(a.uZ, b.uZ, c.uZ, d.uZ);
        o.uP = L(a.uP, b.uP, c.uP, d.uP);
        o.p  = L(a.p,  b.p,  c.p,  d.p);
        o.BR = L(a.BR, b.BR, c.BR, d.BR);
        o.BZ = L(a.BZ, b.BZ, c.BZ, d.BZ);
        o.BP = L(a.BP, b.BP, c.BP, d.BP);
        o.glm = L(a.glm, b.glm, c.glm, d.glm);
        return o;
    }

    void fillGhosts() {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sy; ++j) {
            for (int i = 0; i < sx; ++i) {
                const int c = id(i, j);
                if (type[c] == FLUID) continue;
                if (type[c] == SOLID) {
                    Prim w = W0[c];
                    w.uR = w.uZ = w.uP = 0.0; w.glm = 0.0;
                    W[c] = w; U[c] = prim2cons(w);
                    Psi[c] = Psi[c];
                    continue;
                }
                const double R = Rc(i), Z = Zc(j);
                const double phi = phiLS[c];
                const double nR = nRls[c], nZ = nZls[c];
                const double xr = R - 2.0 * phi * nR;
                const double zr = Z - 2.0 * phi * nZ;
                Prim wr;
                if (!sampleFluid(W, xr, zr, wr)) {
                    Prim w = W0[c];
                    w.uR = w.uZ = w.uP = 0.0; w.glm = 0.0;
                    W[c] = w; U[c] = prim2cons(w);
                    continue;
                }
                const Prim b0r = sampleAny(W0, xr, zr);
                const Prim b0g = W0[c];

                Prim g;
                g.rho = wr.rho;
                g.p   = wr.p;
                g.glm = wr.glm;
                g.uR = -wr.uR; g.uZ = -wr.uZ; g.uP = -wr.uP;
                const double dBR = wr.BR - b0r.BR;
                const double dBZ = wr.BZ - b0r.BZ;
                const double dBP = wr.BP - b0r.BP;
                const double dBn = dBR * nR + dBZ * nZ;
                const double rBR = dBR - 2.0 * dBn * nR;
                const double rBZ = dBZ - 2.0 * dBn * nZ;
                g.BR = b0g.BR + rBR;
                g.BZ = b0g.BZ + rBZ;
                g.BP = b0g.BP + dBP;
                W[c] = g; U[c] = prim2cons(g);
                double fi = (xr - (Rmin + 0.5 * dx)) / dx + NGH;
                double fj = (zr - (Zmin + 0.5 * dz)) / dz + NGH;
                int ii = std::max(0, std::min((int)std::lround(fi), sx - 1));
                int jj = std::max(0, std::min((int)std::lround(fj), sy - 1));
                Psi[c] = Psi[id(ii, jj)];
            }
        }
    }

    void syncPrims() {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i)
                if (type[id(i, j)] == FLUID) W[id(i, j)] = cons2prim(U[id(i, j)]);
    }

    void speeds(double& chOut, double& dtOut, double cfl) const {
        double ch = 1e-12, inv = 1e-12;
        #pragma omp parallel for reduction(max:ch,inv) schedule(static)
        for (int j = NGH; j < sy - NGH; ++j)
            for (int i = NGH; i < sx - NGH; ++i) {
                const int c = id(i, j);
                if (type[c] != FLUID) continue;
                const Prim& w = W[c];
                const double sR = std::abs(w.uR) + cfast(w, w.BR);
                const double sZ = std::abs(w.uZ) + cfast(w, w.BZ);
                ch = std::max(ch, std::max(sR, sZ));
                inv = std::max(inv, sR / dx + sZ / dz);
            }
        chOut = ch;
        dtOut = cfl / inv;
    }

    void computeRHS(std::vector<Cons>& L, double ch) {
        const size_t N = (size_t)sx * sy;
        static std::vector<Prim> WL_x, WR_x, WL_z, WR_z;
        if (WL_x.size() != N) { WL_x.resize(N); WR_x.resize(N); WL_z.resize(N); WR_z.resize(N); }

        #pragma omp parallel for schedule(static)
        for (int j = 1; j < sy - 1; ++j) {
            for (int i = 1; i < sx - 1; ++i) {
                const int c = id(i, j);
                if (type[c] == SOLID) { WL_x[c] = WR_x[c] = WL_z[c] = WR_z[c] = W[c]; continue; }
                const Prim& wm = W[id(i - 1, j)];
                const Prim& wp = W[id(i + 1, j)];
                const Prim& vm = W[id(i, j - 1)];
                const Prim& vp = W[id(i, j + 1)];
                const Prim& w0 = W[c];

                const double* w0a = &w0.rho; const double* wma = &wm.rho; const double* wpa = &wp.rho;
                const double* vma = &vm.rho; const double* vpa = &vp.rho;
                double* xl = &WL_x[c].rho; double* xr = &WR_x[c].rho;
                double* zl = &WL_z[c].rho; double* zr = &WR_z[c].rho;
                for (int q = 0; q < 9; ++q) {
                    xl[q] = weno3_left (wma[q], w0a[q], wpa[q]);
                    xr[q] = weno3_right(wma[q], w0a[q], wpa[q]);
                    zl[q] = weno3_left (vma[q], w0a[q], vpa[q]);
                    zr[q] = weno3_right(vma[q], w0a[q], vpa[q]);
                }
                WL_x[c].rho = std::max(WL_x[c].rho, RHO_FLOOR); WL_x[c].p = std::max(WL_x[c].p, P_FLOOR);
                WR_x[c].rho = std::max(WR_x[c].rho, RHO_FLOOR); WR_x[c].p = std::max(WR_x[c].p, P_FLOOR);
                WL_z[c].rho = std::max(WL_z[c].rho, RHO_FLOOR); WL_z[c].p = std::max(WL_z[c].p, P_FLOOR);
                WR_z[c].rho = std::max(WR_z[c].rho, RHO_FLOOR); WR_z[c].p = std::max(WR_z[c].p, P_FLOOR);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int j = NGH; j < sy - NGH; ++j) {
            for (int i = NGH; i < sx - NGH; ++i) {
                const int c = id(i, j);
                if (type[c] != FLUID) { L[c] = Cons{}; continue; }
                const Cons Fw = hlld_flux(WR_x[id(i - 1, j)], WL_x[c], 0, ch);
                const Cons Fe = hlld_flux(WR_x[c], WL_x[id(i + 1, j)], 0, ch);
                const Cons Gs = hlld_flux(WR_z[id(i, j - 1)], WL_z[c], 1, ch);
                const Cons Gn = hlld_flux(WR_z[c], WL_z[id(i, j + 1)], 1, ch);
                const Cons S  = geomSource(W[c], Rc(i), ch);
                L[c] = (1.0 / dx) * (Fw - Fe) + (1.0 / dz) * (Gs - Gn) + S;
            }
        }
    }

    double step(double dt, double ch) {
        const size_t N = (size_t)sx * sy;
        static std::vector<Cons> U0, L;
        if (U0.size() != N) { U0.resize(N); L.resize(N); }
        U0 = U;

        computeRHS(L, ch);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i) {
                const int c = id(i, j);
                if (type[c] == FLUID) U[c] = U0[c] + dt * L[c];
            }
        syncPrims();
        fillGhosts();

        computeRHS(L, ch);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i) {
                const int c = id(i, j);
                if (type[c] == FLUID)
                    U[c] = 0.75 * U0[c] + 0.25 * (U[c] + dt * L[c]);
            }
        syncPrims();
        fillGhosts();

        computeRHS(L, ch);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i) {
                const int c = id(i, j);
                if (type[c] == FLUID)
                    U[c] = (1.0 / 3.0) * U0[c] + (2.0 / 3.0) * (U[c] + dt * L[c]);
            }
        syncPrims();
        fillGhosts();

        static std::vector<double> PsiNew;
        if (PsiNew.size() != N) PsiNew.resize(N);
        #pragma omp parallel for schedule(static)
        for (int j = NGH; j < sy - NGH; ++j)
            for (int i = NGH; i < sx - NGH; ++i) {
                const int c = id(i, j);
                if (type[c] != FLUID) { PsiNew[c] = Psi[c]; continue; }
                const Prim& w = W[c];
                const double dpR = (w.uR > 0.0)
                    ? (Psi[c] - Psi[id(i - 1, j)]) / dx
                    : (Psi[id(i + 1, j)] - Psi[c]) / dx;
                const double dpZ = (w.uZ > 0.0)
                    ? (Psi[c] - Psi[id(i, j - 1)]) / dz
                    : (Psi[id(i, j + 1)] - Psi[c]) / dz;
                PsiNew[c] = Psi[c] - dt * (w.uR * dpR + w.uZ * dpZ);
            }
        std::swap(Psi, PsiNew);

        const double cr = 0.18;
        const double damp = std::exp(-dt * ch / cr);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sy; ++j)
            for (int i = 0; i < sx; ++i)
                U[id(i, j)].glm *= damp;

        syncPrims();
        fillGhosts();
        return dt;
    }
};

struct SnapshotWriter {
    std::string dataDir, pltDir, pngDir;
    std::string vesselContourCsv;
    bool haveVesselContour = false;
    bool doPlot = true;
    std::string schlierenPalette = "gray";

    void prepareVesselContour(const std::string& outdir, const Machine& m,
                              const SdfTablePGS& sdf) {
        const std::string sdfCsv = outdir + "/data/Vessel/" + m.name + "_vessel_sdf.csv";
        writeVesselCSV(sdfCsv, sdf);
        const std::string raw = outdir + "/data/Raw/" + m.name + "_mhd_vessel_raw.dat";
        vesselContourCsv = outdir + "/data/Vessel/" + m.name + "_mhd_vessel.csv";
        if (doPlot && cmd_exists("gnuplot"))
            haveVesselContour = extractContourTable(sdfCsv, raw, vesselContourCsv,
                                                    "discrete 0.0");
    }

    void writeField(const MHDGrid& G, const std::vector<double>& f,
                    const std::string& name, const std::string& title,
                    const std::string& suffix, std::string cbrange,
                    const std::string& palette) {
        if (cbrange == "auto") {
            double lo = 1e300, hi = -1e300;
            for (int j = NGH; j < G.sy - NGH; ++j)
                for (int i = NGH; i < G.sx - NGH; ++i)
                    if (G.type[G.id(i, j)] == MHDGrid::FLUID) {
                        lo = std::min(lo, f[G.id(i, j)]);
                        hi = std::max(hi, f[G.id(i, j)]);
                    }
            if (!(hi > lo)) { lo = 0.0; hi = 1.0; }
            std::ostringstream cb;
            cb << "set cbrange [" << std::setprecision(6) << lo << ":" << hi << "]\n"
               << "set cbtics (\"" << std::setprecision(2) << lo << "\" " << lo
               << ", \"" << std::setprecision(3) << 0.25*hi+0.75*lo << "\" " << 0.25*hi+0.75*lo
               << ", \"" << std::setprecision(3) << 0.5*(lo+hi) << "\" " << 0.5*(lo+hi)
               << ", \"" << std::setprecision(3) << 0.75*hi+0.25*lo << "\" " << 0.75*hi+0.25*lo
               << ", \"" << std::setprecision(3) << hi << "\" " << hi << ")";
            cbrange = cb.str();
        }
        const std::string csv = dataDir + "/Perturbations/" + name + suffix + ".csv";
        {
            std::ofstream out(csv);
            out << std::setprecision(9) << "R,Z," << name << "\n";
            for (int j = NGH; j < G.sy - NGH; ++j) {
                if (j > NGH) out << "\n";
                for (int i = NGH; i < G.sx - NGH; ++i) {
                    out << G.Rc(i) << "," << G.Zc(j) << ",";
                    if (G.type[G.id(i, j)] != MHDGrid::FLUID) out << "NaN\n";
                    else out << f[G.id(i, j)] << "\n";
                }
            }
        }
        const std::string plt  = pltDir + "/" + name + suffix + ".plt";
        const std::string png = pngDir + "/" + name + suffix + ".png";
        {
            std::ofstream g(plt);
            g << "reset\nset datafile separator ','\n"
              << "set terminal pngcairo size 560,760 enhanced font 'Helvetica,14'\n"
              << "set output '" << png << "'\n"
              << "set xlabel 'X'\nset ylabel 'Y'\n"
              << "set xrange [" << G.Rmin << ":" << G.Rmax << "]\n"
              << "set yrange [" << G.Zmin << ":" << G.Zmax << "]\n"
              << "set size ratio -1\nunset key\nset view map\n";
            if (palette == "gray")
                g << "set palette gray\n";
            else
                g << "set palette viridis\n";
            g << cbrange << "\n"
              << "set title '" << title << "'\n"
              << "splot '" << csv << "' using 1:2:3 with pm3d notitle";
            if (haveVesselContour)
                g << ", \\\n      '" << vesselContourCsv
                  << "' using 1:2:(0) with lines lc rgb 'red' lw 2 notitle";
            g << "\n";
        }
        if (doPlot && cmd_exists("gnuplot"))
            syscmd("gnuplot \"" + plt + "\" 2>&1");
    }

    void snapshot(const MHDGrid& G, double t) {
        const int tms = (int)std::floor(t * 1000.0 + 0.5);
        std::ostringstream ts;
        ts << std::setw(4) << std::setfill('0') << tms;
        const std::string suffix = ts.str();
        std::ostringstream tl;
        tl << std::fixed << std::setprecision(3) << t;
        const std::string tstr = "t = " + tl.str();
        const size_t N = (size_t)G.sx * G.sy;

        std::vector<double> pmag(N, 0.0), rho(N, 0.0), sch(N, 1.0);
        for (int j = 0; j < G.sy; ++j)
            for (int i = 0; i < G.sx; ++i) {
                const int c = G.id(i, j);
                const Prim& w = G.W[c];
                pmag[c] = 0.5 * (w.BR * w.BR + w.BZ * w.BZ + w.BP * w.BP);
                rho[c]  = w.rho;
            }

        for (int j = NGH; j < G.sy - NGH; ++j)
            for (int i = NGH; i < G.sx - NGH; ++i) {
                const int c = G.id(i, j);
                if (G.type[c] != MHDGrid::FLUID) continue;
                const bool fE = G.type[G.id(i + 1, j)] == MHDGrid::FLUID;
                const bool fW = G.type[G.id(i - 1, j)] == MHDGrid::FLUID;
                const bool fN = G.type[G.id(i, j + 1)] == MHDGrid::FLUID;
                const bool fS = G.type[G.id(i, j - 1)] == MHDGrid::FLUID;
                double dRr = 0.0, dZr = 0.0;
                if (fE && fW)      dRr = (rho[G.id(i + 1, j)] - rho[G.id(i - 1, j)]) / (2 * G.dx);
                else if (fE)       dRr = (rho[G.id(i + 1, j)] - rho[c]) / G.dx;
                else if (fW)       dRr = (rho[c] - rho[G.id(i - 1, j)]) / G.dx;
                if (fN && fS)      dZr = (rho[G.id(i, j + 1)] - rho[G.id(i, j - 1)]) / (2 * G.dz);
                else if (fN)       dZr = (rho[G.id(i, j + 1)] - rho[c]) / G.dz;
                else if (fS)       dZr = (rho[c] - rho[G.id(i, j - 1)]) / G.dz;
                const double grad = std::hypot(dRr, dZr);
                sch[c] = std::exp(-35.0 * grad / (100.0 * std::sqrt(std::max(rho[c], 1e-12))));
            }

        std::cout << "[grad-shafranov] snapshot t = " << t << " s\n";
        writeField(G, G.Psi, "psi", "Poloidal flux {/Symbol Y}, " + tstr, suffix,
                   "auto", "viridis");
        writeField(G, pmag, "pmag", "Magnetic pressure B^2/2, " + tstr, suffix,
                   "auto", "viridis");
        writeField(G, rho, "rho", "Density {/Symbol r}, " + tstr, suffix,
                   "auto", "viridis");
        writeField(G, sch, "schlieren", "Mock-Schlieren, " + tstr, suffix,
                   "set cbrange [0:1]", schlierenPalette);
    }
};

// ----------------------------------- main ------------------------------------
int run_perturbed_gs(int argc, char** argv) {
    std::string machineName, sdfPath, outdir;
    int nR = -1, nZ = -1, maxIter = 60;
    double tolPic = 1e-4;
    bool doPlot = true;
    bool useDirectBoundary = false;

    int mhdNx = 256, mhdNy = 512;
    double cfl = 0.6;
    double tFinal = 0.279;
    bool keepToroidal = false;
    double kp = 2.0, pv = 0.1, rv = 0.1;
    std::string schPalette = "gray";
    std::vector<double> snapTimes = {0.0, 0.07, 0.14, 0.21, 0.25, 0.279};

    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&]() -> std::string { return (a + 1 < argc) ? argv[++a] : ""; };

        if      (s == "--machine")        { machineName = next(); }
        else if (s == "--sdf")            { sdfPath = next(); }
        else if (s == "--nr")             { nR = std::stoi(next()); }
        else if (s == "--nz")             { nZ = std::stoi(next()); }
        else if (s == "--maxiter")        { maxIter = std::stoi(next()); }
        else if (s == "--tol")            { tolPic = std::stod(next()); }
        else if (s == "--outdir")         { outdir = next(); }
        else if (s == "--noplot")         { doPlot = false; }
        else if (s == "--direct-boundary"){ useDirectBoundary = true; }
        else if (s == "--mhd-nx")         { mhdNx = std::stoi(next()); }
        else if (s == "--mhd-ny")         { mhdNy = std::stoi(next()); }
        else if (s == "--cfl")            { cfl = std::stod(next()); }
        else if (s == "--tfinal")         { tFinal = std::stod(next()); }
        else if (s == "--palette")        { schPalette = next(); }
        else if (s == "--keep-toroidal")  { keepToroidal = true; }
        else if (s == "--kp")             { kp = std::stod(next()); }
        else if (s == "--pv")             { pv = std::stod(next()); }
        else if (s == "--rv")             { rv = std::stod(next()); }
        else if (s == "--gamma")          { GAM = std::stod(next()); }
    }

    if (machineName.empty()) {
        std::cout << "\n[grad-shafranov] Select machine:\n"
                  << "  1)  MAST  — sdf/MAST/mast.dat\n"
                  << "  2)  ITER  — sdf/ITER/iter.dat\n"
                  << "  3)  ST40  — sdf/ST40/st40.dat\n"
                  << "[grad-shafranov] Choice [1-3, default 3]: ";
        std::string cs; std::getline(std::cin, cs);
        int ch = 3;
        if (!cs.empty()) { try { ch = std::stoi(cs); } catch (...) { ch = 3; } }
        machineName = (ch == 1) ? "mast" : (ch == 2) ? "iter" : "st40";
    }
    for (char& c : machineName) c = (char)std::tolower((unsigned char)c);

    Machine machine;
    if      (machineName == "mast") {machine = makeMAST(); outdir = "outputs/Grad-Shafranov/Perturbed/MAST";}
    else if (machineName == "iter") {machine = makeITER(); outdir = "outputs/Grad-Shafranov/Perturbed/ITER";}
    else if (machineName == "st40") {machine = makeST40(); outdir = "outputs/Grad-Shafranov/Perturbed/ST40";}
    else { std::cerr << "[grad-shafranov] Unknown machine '" << machineName << "'\n"; return 1; }

    if (sdfPath.empty()) sdfPath = machine.sdfDefault;
    if (nR < 3) nR = machine.nRdef;
    if (nZ < 3) nZ = machine.nZdef;

    std::cout << "\n========================= Perturbation GS Solver: " << machine.name << " =========================\n\n"
              << "[grad-shafranov]             Machine: " << machine.name << "\n"
              << "[grad-shafranov]           GS domain: [" << machine.Rmin << ", " << machine.Rmax << "] x ["
              << machine.Zmin << ", " << machine.Zmax << "]\n"
              << "[grad-shafranov]                 Ip = " << machine.Ip
              << "\n[grad-shafranov]                 p0 = " << machine.p0
              << "Pa\n[grad-shafranov]                 g0 = " << machine.g0 << "\n"
              << "[grad-shafranov] (alpha_m, alpha_n) = (" << machine.alpha_m << ", " << machine.alpha_n << ")\n"
              << "[grad-shafranov]            MHD mesh: " << mhdNx << " x " << mhdNy
              << "\n[grad-shafranov]                CFL = " << cfl 
              << "\n[grad-shafranov]            t_final = " << tFinal << "\n";
    std::cout << "\n";

    Solver S(machine, nR, nZ);
    S.useDirectBoundary = useDirectBoundary;
    if (!S.sdf.load(sdfPath, machine.sdfScale, machine.sdfDR, machine.sdfDZ)) {
        std::cout << "[grad-shafranov]  continuing WITHOUT a vessel SDF (whole domain admissible)\n";
    }

    S.precomputeGreens();
    if (useDirectBoundary)
        std::cout << "[grad-shafranov]  (using direct O(N³) boundary-flux sum, not Von Hagenow's method)\n";
    S.initialise();

    std::string dataDir = outdir + "/data";
    std::string pltDir = outdir + "/plots/plt";
    std::string pngDir = outdir + "/plots/png";
    mkdir_p(pltDir);
    mkdir_p(pngDir);
    mkdir_p(dataDir + "/Raw");
    mkdir_p(dataDir + "/Perturbations");
    mkdir_p(dataDir + "/Vessel");

    // ------------------------------ Picard Loop -----------------------------
    bool converged = false;

    for (int p = 0; p < maxIter; ++p) {
        S.solveCoilCurrentsCRATOS();
        S.buildCoilPsi();
        if (!S.findAxis()) {
            std::cerr << "[grad-shafranov] failed to locate magnetic axis at iter " << p << "\n";
            return 1;
        }
        S.buildMask();
        S.buildJphi();
        Eigen::VectorXd psiNew = S.solvePlasma(S.computeBoundary());

        Eigen::VectorXd dNew = psiNew - S.psiPl;
        const double rel = dNew.lpNorm<Eigen::Infinity>()
                         / std::max(S.psiTot.lpNorm<Eigen::Infinity>(), 1e-300);

        if (p > 0) {
            const Eigen::VectorXd dd = S.dOld - dNew;
            const double n2 = dd.squaredNorm();
            if (n2 > 1e-300) {
                S.lambda = S.lambda + (S.lambda - 1.0) * dd.dot(dNew) / n2;
            }
            S.lambda = std::max(0.0, std::min(0.95, S.lambda));
        }
        const double alphaB = 1.0 - S.lambda;
        S.dOld = dNew;
        S.psiPl += alphaB * dNew;

        if (rel < tolPic) { converged = true; break; }
    }

    S.solveCoilCurrentsCRATOS();
    S.buildCoilPsi();
    S.findAxis();
    S.buildMask();
    S.buildJphi();

    double mR0 = machine.mhdRmin, mR1 = machine.mhdRmax;
    double mZ0 = machine.mhdZmin, mZ1 = machine.mhdZmax;
    if (!(mR1 > mR0) || !(mZ1 > mZ0)) {
        mR0 = 1e30; mR1 = -1e30; mZ0 = 1e30; mZ1 = -1e30;
        for (int j = 0; j < S.sdf.ny; ++j)
            for (int i = 0; i < S.sdf.nx; ++i)
                if (S.sdf.sdf[(size_t)j * S.sdf.nx + i] < 0.0) {
                    mR0 = std::min(mR0, S.sdf.xmin + i * S.sdf.dx);
                    mR1 = std::max(mR1, S.sdf.xmin + i * S.sdf.dx);
                    mZ0 = std::min(mZ0, S.sdf.ymin + j * S.sdf.dy);
                    mZ1 = std::max(mZ1, S.sdf.ymin + j * S.sdf.dy);
                }
        const double padR = 0.05 * (mR1 - mR0), padZ = 0.05 * (mZ1 - mZ0);
        mR0 = std::max(machine.Rmin + 1e-3, mR0 - padR); mR1 += padR;
        mZ0 -= padZ; mZ1 += padZ;
    }

    MHDGrid G;
    G.setup(mR0, mR1, mZ0, mZ1, mhdNx, mhdNy, S.sdf);

    const double pmax = machine.p0;
    for (int j = 0; j < G.sy; ++j)
        for (int i = 0; i < G.sx; ++i) {
            const int c = G.id(i, j);
            const double R = std::max(G.Rc(i), 1e-4), Z = G.Zc(j);
            const double psiEq = S.sample(S.psiTot, R, Z);
            const double pSI = S.pressureSI(psiEq);
            const double g   = S.gfunSI(psiEq);

            Prim w;
            w.rho = pSI / pmax + rv;
            w.p   = kp * (pSI / pmax) + pv;
            w.uR = w.uZ = w.uP = 0.0;
            w.BR = S.sampleBR(S.psiPl, R, Z);
            w.BZ = S.sampleBZ(S.psiPl, R, Z);
            w.BP = keepToroidal ? g / R : 0.0;
            w.glm = 0.0;
            G.W[c] = w;
            G.U[c] = prim2cons(w);
            G.Psi[c] = S.sample(S.psiPl, R, Z);
        }
    G.W0 = G.W;
    G.fillGhosts();

    SnapshotWriter out;
    out.dataDir = dataDir; out.pltDir = pltDir; out.pngDir = pngDir;
    out.doPlot = doPlot;
    out.schlierenPalette = schPalette;
    out.prepareVesselContour(outdir, machine, S.sdf);

    double t = 0.0;
    size_t snapIdx = 0;
    int stepNo = 0;
    while (snapIdx < snapTimes.size() && snapTimes[snapIdx] <= 1e-12) {
        out.snapshot(G, snapTimes[snapIdx]);
        ++snapIdx;
    }

    while (t < tFinal - 1e-12) {
        double ch, dt;
        G.speeds(ch, dt, cfl);
        if (stepNo < 5) dt *= 0.2;
        bool hitSnap = false;
        if (snapIdx < snapTimes.size() && t + dt >= snapTimes[snapIdx] - 1e-12) {
            dt = snapTimes[snapIdx] - t;
            hitSnap = true;
        }
        if (t + dt > tFinal) dt = tFinal - t;
        if (dt < 1e-10) {
            std::cout << "  dt collapsed (" << dt << ") at t = " << t
                      << " — stopping (cf. thesis crash just after t = 0.279).\n";
            break;
        }
        G.step(dt, ch);
        t += dt;
        ++stepNo;
        if (stepNo % 200 == 0)
            std::cout << "[grad-shafranov] step " << stepNo << " t = " << t
                      << "  dt = " << dt << "  ch = " << ch << "\n";
        if (hitSnap && snapIdx < snapTimes.size()
            && std::abs(t - snapTimes[snapIdx]) < 1e-9) {
            out.snapshot(G, snapTimes[snapIdx]);
            ++snapIdx;
        }
    }

    std::cerr << "\n[grad-shafranov] Wrote data to " << outdir << "/data\n";
    std::cerr << "[grad-shafranov] Wrote plots to " << outdir << "/plots/png\n";


    std::cout << "\nDone.\n";
    return converged ? 0 : 2;
}