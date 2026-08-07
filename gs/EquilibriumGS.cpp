// =============================================================================
// EquilibriumGS.cpp — Free-boundary Grad–Shafranov solver
// =============================================================================
// BUILD
// g++ -std=c++17 -O3 -I /opt/homebrew/opt/eigen/include/eigen3 / 
//              EquilibriumGS.cpp -o EquilibriumGS
// 
// RUN
//   ./EquilibriumGS
//   ./EquilibriumGS --machine mast --sdf sdf/MAST/mast.dat
//   ./EquilibriumGS --machine iter --nr 129 --nz 257 --outdir out_iter
// =============================================================================
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "EquilibriumGS.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <chrono>

static constexpr double MU0 = 4.0e-7 * M_PI;

static inline int syscmd(const std::string& cmd) {
    return std::system(cmd.c_str());
}

static inline void mkdir_p(const std::string& p) {
    syscmd("mkdir -p \"" + p + "\"");
}

static inline bool cmd_exists(const std::string& n) {
    return syscmd("command -v " + n + " >/dev/null 2>&1") == 0;
}

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

struct SdfTableEGS {
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0, dx = 0, dy = 0;
    int nx = 0, ny = 0;
    std::vector<double> sdf;
    bool loaded = false;

    bool load(const std::string& path, double scale, double dR, double dZ) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[gs] Cannot open SDF: " << path << "\n";
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
            std::cerr << "[gs] No data in SDF: " << path << "\n";
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
            std::cerr << "[gs] Cannot determine SDF nx.\n";
            return false;
        }

        ny = static_cast<int>(xs.size()) / nx;
        if (nx * ny != static_cast<int>(xs.size())) {
            std::cerr << "[gs] SDF size " << xs.size() << " not divisible by nx=" << nx << "\n";
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
        std::cerr << "\n[grad-shafranov] SDF loaded: " << nx << " x " << ny
                  << "\n[grad-shafranov] R[" << xmin << ", " << xmax 
                  << "]\n[grad-shafranov] Z[" << ymin << ", " << ymax << "]\n\n";
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
        const double v00 = sdf[j0 * nx + i0];
        const double v10 = sdf[j0 * nx + (i0 + 1)];
        const double v01 = sdf[(j0 + 1) * nx + i0];
        const double v11 = sdf[(j0 + 1) * nx + (i0 + 1)];
        return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11;
    }
};

class EquilibriumGSGrid {
    public:
        EquilibriumGSGrid(double Rmin, double Rmax, double Zmin, double Zmax, int nR, int nZ)
            : Rmin_(Rmin), Rmax_(Rmax), Zmin_(Zmin), Zmax_(Zmax), nR_(nR), nZ_(nZ) {
            if (nR_ < 3 || nZ_ < 3) {
                throw std::invalid_argument("EquilibriumGSGrid: need at least 3 nodes per direction");
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
                throw std::runtime_error("EquilibriumGSGrid: linear solve failed");
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
                throw std::runtime_error("EquilibriumGSGrid: SparseLU factorisation failed");
            }
            factorized_ = true;
        }
};
struct FilamentEGS { double R, Z, w; };
struct CircuitEGS { std::string name; std::vector<FilamentEGS> fil; };
struct IsoPairEGS { double R1, Z1, R2, Z2; };

struct MachineEGS {
    std::string name;
    std::string sdfDefault;
    double Rmin, Rmax, Zmin, Zmax;
    int nRdef, nZdef;
    double sdfScale = 1.0;
    double sdfDR = 0.0;
    double sdfDZ = 0.0;
    std::vector<CircuitEGS> circuits;
    std::vector<std::pair<double, double>> xpoints;
    std::vector<IsoPairEGS> isoflux;
    double Ip, p0, g0;
    double alpha_m = 1.0;
    double alpha_n = 1.0;
    double axR, axZ, axSigma;
    double sliceR1, sliceR2;
    std::vector<double> refCurrents;
};

static CircuitEGS solenoid(const std::string& name, double R,
                        double Zlo, double Zhi, double turns, int nFil = 24) {
    CircuitEGS c; c.name = name;
    for (int k = 0; k < nFil; ++k) {
        const double t = (k + 0.5) / nFil;
        c.fil.push_back({R, Zlo + t * (Zhi - Zlo), turns / nFil});
    }
    return c;
}

static CircuitEGS coil(const std::string& name, double R, double Z, double turns = 1.0) {
    CircuitEGS c; c.name = name; c.fil.push_back({R, Z, turns}); return c;
}

static MachineEGS makeMAST() {
    MachineEGS m;
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

static MachineEGS makeITER() {
    MachineEGS m;
    m.name = "ITER";
    m.sdfDefault = "sdf/ITER/iter.dat";
    m.sdfScale = 1.00, m.sdfDR = 0.30, m.sdfDZ = 0.00;
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

static MachineEGS makeST40() {
    MachineEGS m;
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
    return m;
}

struct RZEGS { double R, Z; };

struct DiagnosticsEGS {
    double Rg = 0, Zg = 0, a = 0, Aspect = 0, eps = 0, kappa = 0, Delta = 0;
    double ShafR = 0, ShafZ = 0, li = 0, betap = 0, q95 = 0, Volume = 0, IT = 0;
    double LCFSRmax = 0, LCFSRmin = 0, LCFSZmax = 0, LCFSZmin = 0;
};
struct SolverEGS {
    MachineEGS machine;
    EquilibriumGSGrid grid;
    SdfTableEGS sdf;

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
    std::vector<RZEGS> xpointsRefined;
    bool useDirectBoundary = false;

    double lambda = 0.3;
    Eigen::VectorXd dOld;

    Eigen::VectorXd computeBoundary() const {
        return useDirectBoundary ? boundaryValuesDirect() : boundaryValues();
    }

    SolverEGS(const MachineEGS& m, int nR, int nZ) : machine(m), grid(m.Rmin, m.Rmax, m.Zmin, m.Zmax, nR, nZ) {
        Rv.resize(nR); Zv.resize(nZ);
        for (int i = 0; i < nR; ++i) {
            Rv[i] = grid.R(i);
        }
        for (int j = 0; j < nZ; ++j) {
            Zv[j] = grid.Z(j);
        }
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
        return -(sample(f, R, Z + h) - sample(f, R, Z - h)) / (2.0 * h * R);
    }

    double sampleBZ(const Eigen::VectorXd& f, double R, double Z) const {
        const double h = 0.5 * grid.dR();
        return (sample(f, R + h, Z) - sample(f, R - h, Z)) / (2.0 * h * R);
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
            for (size_t c = 0; c < machine.refCurrents.size(); ++c) {
                refA.push_back(machine.refCurrents[c] * 1e3);
            }

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

    bool findAxis() {

        double Rlo = 1e30, Rhi = -1e30, Zlo = 1e30, Zhi = -1e30;
        auto acc = [&](double R, double Z) {
            Rlo = std::min(Rlo, R); Rhi = std::max(Rhi, R);
            Zlo = std::min(Zlo, Z); Zhi = std::max(Zhi, Z);
        };
        for (auto& xp : machine.xpoints) acc(xp.first, xp.second);
        for (auto& io : machine.isoflux) { acc(io.R1, io.Z1); acc(io.R2, io.Z2); }
        const double mR = 0.10 * (Rhi - Rlo), mZ = 0.10 * (Zhi - Zlo);
        Rlo -= mR; Rhi += mR; Zlo -= mZ; Zhi += mZ;

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

        struct BPtEGS { int i, j; double dUdn, dl; };
        std::vector<BPtEGS> bpts;
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
        std::vector<BPtEGS> bsrc;
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

    double plasmaCurrentNow() const {
        const double dA = grid.dR() * grid.dZ();
        double s = 0.0;
        for (int k = 0; k < Jphi.size(); ++k) s += Jphi(k) * dA;
        return s;
    }

    std::vector<RZEGS> traceFluxSurface(double psiLevel, int Ntheta = 360) const {
        std::vector<RZEGS> pts;
        pts.reserve(Ntheta);
        const double stepCoarse = 0.5 * std::min(grid.dR(), grid.dZ());
        const double rMaxSearch = 2.0 * std::max(machine.Rmax - machine.Rmin, machine.Zmax - machine.Zmin);
        const double signRef = (psiLevel >= psiAxis) ? 1.0 : -1.0;

        for (int k = 0; k < Ntheta; ++k) {
            const double theta = 2.0 * M_PI * k / Ntheta;
            const double ct = std::cos(theta), st = std::sin(theta);
            double sPrev = 0.0, sHit = -1.0;
            for (double s = stepCoarse; s < rMaxSearch; s += stepCoarse) {
                const double R = Raxis + s * ct, Z = Zaxis + s * st;
                if (R < Rv.front() || R > Rv.back() || Z < Zv.front() || Z > Zv.back()) break;
                const double val = sample(psiTot, R, Z);
                if (signRef * (val - psiLevel) >= 0.0) { sHit = s; break; }
                sPrev = s;
            }
            if (sHit < 0.0) { pts.push_back({Raxis, Zaxis}); continue; }
            double lo = sPrev, hi = sHit;
            for (int ref = 0; ref < 40; ++ref) {
                const double sm = 0.5 * (lo + hi);
                const double val = sample(psiTot, Raxis + sm * ct, Zaxis + sm * st);
                if (signRef * (val - psiLevel) >= 0.0) hi = sm; else lo = sm;
            }
            const double sFinal = 0.5 * (lo + hi);
            pts.push_back({Raxis + sFinal * ct, Zaxis + sFinal * st});
        }
        return pts;
    }

    DiagnosticsEGS computeDiagnosticsEGS() const {
        DiagnosticsEGS d{};
        const std::vector<RZEGS> lcfs = traceFluxSurface(psiBnd, 360);

        d.LCFSRmax = -1e300; d.LCFSRmin = 1e300; d.LCFSZmax = -1e300; d.LCFSZmin = 1e300;
        double RatZmax = 0, RatZmin = 0;
        for (const auto& p : lcfs) {
            if (p.R > d.LCFSRmax) d.LCFSRmax = p.R;
            if (p.R < d.LCFSRmin) d.LCFSRmin = p.R;
            if (p.Z > d.LCFSZmax) { d.LCFSZmax = p.Z; RatZmax = p.R; }
            if (p.Z < d.LCFSZmin) { d.LCFSZmin = p.Z; RatZmin = p.R; }
        }
        d.Rg = d.LCFSRmin + 0.5 * (d.LCFSRmax - d.LCFSRmin);
        d.Zg = 0.5 * (d.LCFSZmax + d.LCFSZmin);
        d.a  = 0.5 * (d.LCFSRmax - d.LCFSRmin);
        d.Aspect = d.Rg / std::max(d.a, 1e-300);
        d.eps = 1.0 / std::max(d.Aspect, 1e-300);
        d.kappa = 0.5 * (d.LCFSZmax - d.LCFSZmin) / std::max(d.a, 1e-300);
        d.Delta = 0.5 * ((d.Rg - RatZmax) + (d.Rg - RatZmin)) / std::max(d.a, 1e-300);
        d.ShafR = Raxis - d.Rg;
        d.ShafZ = Zaxis - d.Zg;

        const double dA = grid.dR() * grid.dZ();
        const double denPsi = psiBnd - psiAxis;
        auto Fint = [&](double x) {
            const int m = 100; double s = 0.0;
            for (int k = 0; k < m; ++k) s += jfunc(x + (1.0 - x) * (k + 0.5) / m);
            return s * (1.0 - x) / m;
        };
        double Vol = 0.0, liNum = 0.0, pInt = 0.0;
        for (int j = 0; j < grid.nZ(); ++j)
            for (int i = 0; i < grid.nR(); ++i) {
                const int id = grid.idx(i, j);
                if (!mask[id]) continue;
                const double R = Rv[i];
                const double dV = 2.0 * M_PI * R * dA;
                Vol += dV;
                const double Br = sampleBR(psiTot, R, Zv[j]);
                const double Bz = sampleBZ(psiTot, R, Zv[j]);
                liNum += 2.0 * (Br * Br + Bz * Bz) * dV;
                const double x = std::max(0.0, std::min(1.0, (psiTot(id) - psiAxis) / denPsi));
                const double p = -aCoef / Raxis * denPsi * Fint(x);
                pInt += p * dA;
            }
        d.Volume = Vol;
        d.IT = plasmaCurrentNow();
        d.li = (d.IT > 0.0) ? liNum / (MU0 * MU0 * d.Rg * d.IT * d.IT) : 0.0;
        d.betap = (d.IT > 0.0) ? (8.0 * M_PI / MU0) * pInt / (d.IT * d.IT) : 0.0;

        const std::vector<RZEGS> fs95 = traceFluxSurface(psiAxis + 0.95 * denPsi, 360);
        double qsum = 0.0;
        for (size_t k = 0; k < fs95.size(); ++k) {
            const RZEGS& p0 = fs95[k];
            const RZEGS& p1 = fs95[(k + 1) % fs95.size()];
            const double dl = std::hypot(p1.R - p0.R, p1.Z - p0.Z);
            const double Rm = 0.5 * (p0.R + p1.R), Zm = 0.5 * (p0.Z + p1.Z);
            const double Br = sampleBR(psiTot, Rm, Zm), Bz = sampleBZ(psiTot, Rm, Zm);
            const double Bp = std::sqrt(Br * Br + Bz * Bz);
            const double psiHere = sample(psiTot, Rm, Zm);
            const double x = std::max(0.0, std::min(1.0, (psiHere - psiAxis) / denPsi));
            const double F2 = machine.g0 * machine.g0 - 2.0 * MU0 * Raxis * bCoef * denPsi * Fint(x);
            const double g = std::sqrt(std::max(0.0, F2));
            const double Bt = g / std::max(Rm, 1e-12);
            if (Bp > 1e-12) qsum += (Bt / (Rm * Bp)) * dl;
        }
        d.q95 = qsum / (2.0 * M_PI);
        return d;
    }
};
// -------------------------------- I/O helpers --------------------------------
static void writeFieldCSV(const std::string& path, const SolverEGS& S,
                          const Eigen::VectorXd& f, const std::string& head,
                          bool maskOutsideVessel = false) {
    std::ofstream out(path);
    out << std::setprecision(12) << "R,Z," << head << "\n";
    for (int j = 0; j < S.grid.nZ(); ++j) {
        if (j > 0) out << "\n";
        for (int i = 0; i < S.grid.nR(); ++i) {
            const bool outside = maskOutsideVessel && S.sdf.loaded
                               && S.sdf.interp(S.Rv[i], S.Zv[j]) > 0.0;
            out << S.Rv[i] << "," << S.Zv[j] << ",";
            if (outside) out << "NaN\n"; else out << f(S.grid.idx(i, j)) << "\n";
        }
    }
}

static void writeVesselCSV(const std::string& path, const SdfTableEGS& s) {
    if (!s.loaded) return;
    std::ofstream out(path);
    out << std::setprecision(12) << "R,Z,phi\n";
    for (int j = 0; j < s.ny; ++j) {
        if (j > 0) out << "\n";
        for (int i = 0; i < s.nx; ++i)
            out << (s.xmin + i * s.dx) << "," << (s.ymin + j * s.dy)
                << "," << s.sdf[j * s.nx + i] << "\n";
    }
}

static void writeCoilsCSV(const std::string& path, const MachineEGS& m,
                          const Eigen::VectorXd& I) {
    std::ofstream out(path);
    out << std::setprecision(9) << "R,Z,I_circuit_A\n";
    for (size_t c = 0; c < m.circuits.size(); ++c)
        for (const auto& fl : m.circuits[c].fil)
            out << fl.R << "," << fl.Z << "," << I((int)c) << "\n";
}

static void writeSliceCSV(const std::string& path, const SolverEGS& S) {
    const int n = 256;
    std::ofstream out(path);
    out << std::setprecision(12) << "R,psibar,p_over_paxis,g_over_gaxis\n";
    const double den = S.psiBnd - S.psiAxis;
    auto Fint = [&](double x) {
        const int m = 200; double s = 0.0;
        for (int k = 0; k < m; ++k) s += S.jfunc(x + (1.0 - x) * (k + 0.5) / m);
        return s * (1.0 - x) / m;
    };
    const double Snorm = S.Snorm;
    const double p_axis = S.machine.p0;
    const double g_axis_sq = S.machine.g0 * S.machine.g0
                           - 2.0 * MU0 * S.Raxis * S.bCoef * den * Snorm;
    const double g_axis = std::sqrt(std::max(0.0, g_axis_sq));

    for (int k = 0; k < n; ++k) {
        const double t = (double)k / (n - 1);
        const double R = S.machine.sliceR1 + t * (S.machine.sliceR2 - S.machine.sliceR1);
        const double Z = 0.0;
        const double psi = S.sample(S.psiTot, R, Z);
        double x = (psi - S.psiAxis) / den;
        const double xc = std::max(0.0, std::min(1.0, x));
        const double p = -S.aCoef / S.Raxis * den * Fint(xc);
        const double F2 = S.machine.g0 * S.machine.g0
                        - 2.0 * MU0 * S.Raxis * S.bCoef * den * Fint(xc);
        const double g = std::sqrt(std::max(0.0, F2));
        double psibar = xc;
        double p_norm = (p_axis > 0.0) ? p / p_axis : 0.0;
        double g_norm = (g_axis > 0.0) ? g / g_axis : 0.0;
        out << R << "," << psibar << "," << p_norm << "," << g_norm << "\n";
    }
}

static bool extractContourTable(const std::string& csv, const std::string& raw,
                                const std::string& comma, const std::string& spec) {
    {
        std::ofstream gp(raw + ".plt");
        gp << "reset\nset datafile separator ','\n"
           << "set table '" << raw << "'\n"
           << "set contour base\nset cntrparam levels " << spec << "\n"
           << "unset surface\n"
           << "splot '" << csv << "' using 1:2:3 with lines\nunset table\n";
    }
    if (syscmd("gnuplot \"" + raw + ".plt\" 2>/dev/null") != 0) return false;
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

static void writeEquilibriumPlot(const std::string& outdir, const SolverEGS& S,
                                 bool haveSdf, bool execute) {
    const std::string label = S.machine.name;
    const std::string fieldMask = outdir + "/data/" + label + "_psi_masked.csv";
    const std::string fieldFull = outdir + "/data/" + label + "_psi_full.csv";
    const std::string script = outdir + "/plots/plt/" + label + ".plt";
    const std::string png = outdir + "/plots/png/" + label + ".png";
    const std::string lcfsRaw = outdir + "/data/Raw/" + label + "_lcfs_raw.dat";
    const std::string lcfs = outdir + "/data/" + label + "_lcfs.csv";
    const std::string vesRaw = outdir + "/data/Raw/" + label + "_vessel_raw.dat";
    const std::string vessel = outdir + "/data/" + label + "_vessel.csv";
    const std::string flxRaw = outdir + "/data/Raw/" + label + "_flux_raw.dat";
    const std::string flx = outdir + "/data/" + label + "_flux.csv";

    bool haveLCFS = false, haveVessel = false, haveFlux = false;

    double cblo = 0.0, cbhi = 0.0;
    bool haveCbRange = false;
    if (haveSdf && S.psiBnd != 0.0) {
        cblo = 1e300; cbhi = -1e300;
        const double den = S.psiBnd - S.psiAxis;
        for (int j = 0; j < S.grid.nZ(); ++j) {
            for (int i = 0; i < S.grid.nR(); ++i) {
                const int id = S.grid.idx(i, j);
                const double val = S.psiTot(id);
                const double x = (val - S.psiAxis) / den;
                const bool insideLCFS = (x >= 0.0 && x <= 1.0);
                const bool insideVessel = S.sdf.interp(S.Rv[i], S.Zv[j]) <= 0.0;
                if (insideLCFS && insideVessel && std::isfinite(val)) {
                    cblo = std::min(cblo, val);
                    cbhi = std::max(cbhi, val);
                    haveCbRange = true;
                }
            }
        }
        if (haveCbRange) {
            const double margin = 0.05 * (cbhi - cblo);
            cblo -= margin;
            cbhi += margin;
        }
    }

    if (execute && cmd_exists("gnuplot")) {
        std::ostringstream s1; s1 << "discrete " << std::setprecision(15) << S.psiBnd;
        haveLCFS = extractContourTable(fieldMask, lcfsRaw, lcfs, s1.str());
        if (haveSdf)
            haveVessel = extractContourTable(outdir + "/data/" + label + "_vessel_sdf.csv",
                                             vesRaw, vessel, "discrete 0.0");

        std::vector<double> vals;
        vals.reserve(S.psiTot.size());
        for (int j = 0; j < S.grid.nZ(); ++j) {
            for (int i = 0; i < S.grid.nR(); ++i) {
                const int id = S.grid.idx(i, j);
                const double val = S.psiTot(id);
                const bool insideVessel = S.sdf.interp(S.Rv[i], S.Zv[j]) <= 0.0;
                if (insideVessel && std::isfinite(val)) {
                    vals.push_back(val);
                }
            }
        }

        if (!vals.empty()) {
            std::sort(vals.begin(), vals.end());
            const auto pct = [&](double p) {
                const size_t idx = std::min(vals.size() - 1,
                    (size_t)std::llround(p * (double)(vals.size() - 1)));
                return vals[idx];
            };
            const double p_lo = pct(0.01), p_hi = pct(0.99);
            const double margin = 0.6 * (p_hi - p_lo);
            const double lo = p_lo - margin, hi = p_hi + margin;
            const int nLevels = 0;
            std::ostringstream s2;
            s2 << "incremental " << std::setprecision(15) << lo << "," << (hi - lo) / nLevels << "," << hi;
            haveFlux = extractContourTable(fieldMask, flxRaw, flx, s2.str());
        }
    }

    const double aspect = (S.machine.Zmax - S.machine.Zmin) / (S.machine.Rmax - S.machine.Rmin);
    std::ofstream gp(script);
    gp << "reset\nset datafile separator ','\n"
       << "set terminal pngcairo size 640," << (int)(180 + 260 * aspect)
       << " enhanced font 'Helvetica,13'\n"
       << "set output '" << png << "'\n"
       << "set title '" << label << " - Free-Boundary Equilibrium {/Symbol Y}(R, Z)'\n"
       << "set xlabel 'R [m]'\nset ylabel 'Z [m]'\n"
       << "set xrange [" << S.machine.Rmin << ":" << S.machine.Rmax << "]\n"
       << "set yrange [" << S.machine.Zmin << ":" << S.machine.Zmax << "]\n"
       << "set size ratio -1\nunset key\nset view map\n"
       << "set palette viridis\nset cblabel '{/Symbol Y} [Tm^{-2}]'\n";

    if (haveSdf && haveCbRange) {
        gp << "set cbrange [" << cblo << ":" << cbhi << "]\n";
    }

    gp << "splot '" << fieldMask << "' using 1:2:3 with pm3d notitle";
    if (haveFlux)
        gp << ", \\\n      '" << flx << "' using 1:2:3 with lines lc rgb '#303030' lw 0.5 notitle";
    if (haveLCFS)
        gp << ", \\\n      '" << lcfs << "' using 1:2:3 with lines lc rgb 'black' lw 3 notitle";
    if (haveVessel)
        gp << ", \\\n      '" << vessel << "' using 1:2:3 with lines lc rgb 'red' lw 2 notitle";
    gp << "\n";
    gp.close();

    const std::string script2 = outdir + "/plots/plt/" + label + "_slice.plt";
    std::ofstream g2(script2);
    g2 << "reset\nset datafile separator ','\n"
       << "set terminal pngcairo size 680,440 enhanced font 'Helvetica,13'\n"
       << "set output '" << outdir << "/plots/png/" << label << "_slice.png'\n"
       << "set title '" << label << " midplane cross-section'\n"
       << "set xlabel 'R [m]'\nset ylabel 'Normalised quantity'\n"
       << "set grid\nset key bottom center\n"
       << "set yrange [0:1]\n"
       << "plot '" << outdir << "/data/" << label << "_slice.csv' u 1:2 w l lw 2 lc rgb '#2255cc' "
          "t '{/Symbol Y}/{/Symbol Y}_0', \\\n"
       << "     '' u 1:3 w l lw 2 lc rgb '#228833' t 'p/p_0', \\\n"
       << "     '' u 1:4 w l lw 2 lc rgb '#ee9911' t 'g/g_0'\n";
    g2.close();

    if (execute && cmd_exists("gnuplot")) {
        if (syscmd("gnuplot \"" + script + "\" 2>/dev/null") == 0)
        syscmd("gnuplot \"" + script2 + "\" 2>/dev/null");
    } else if (execute) {
        std::cout << "[grad-shafranov] gnuplot not found — run the .plt scripts manually\n";
    }
}
// ----------------------------------- main ------------------------------------
int run_equilibrium_gs(int argc, char** argv) {
    using clock_t = std::chrono::steady_clock;

    std::string machineName, sdfPath, outdir;
    int nR = -1, nZ = -1, maxIter = 60;
    double tolPic = 1e-4;
    bool doPlot = true;
    bool useDirectBoundary = false;
    double sdfScale = 1.0, sdfDR = 0.0, sdfDZ = 0.0;

    for (int a = 1; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&]() -> std::string { return (a + 1 < argc) ? argv[++a] : ""; };

        if      (s == "--machine")         { machineName = next(); }
        else if (s == "--sdf")             { sdfPath = next(); }
        else if (s == "--nr")              { nR = std::stoi(next()); }
        else if (s == "--nz")              { nZ = std::stoi(next()); }
        else if (s == "--maxiter")         { maxIter = std::stoi(next()); }
        else if (s == "--tol")             { tolPic = std::stod(next()); }
        else if (s == "--outdir")          { outdir = next(); }
        else if (s == "--noplot")          { doPlot = false; }
        else if (s == "--sdf-scale")       { sdfScale = std::stod(next()); }
        else if (s == "--sdf-droff")       { sdfDR = std::stod(next()); }
        else if (s == "--sdf-dzoff")       { sdfDZ = std::stod(next()); }
        else if (s == "--direct-boundary") { useDirectBoundary = true; }
    }

    if (machineName.empty()) {
        std::cout << "\n[grad-shafranov] Select machine:\n"
                  << "  1)  MAST  — sdf/MAST/mast.dat\n"
                  << "  2)  ITER  — sdf/ITER/iter.dat\n"
                  << "  3)  ST40  — sdf/ST40/st40.dat\n"
                  << "[grad-shafranov] Choice [1-3, default 1]: ";
        std::string cs; std::getline(std::cin, cs);
        int ch = 1;
        if (!cs.empty()) { try { ch = std::stoi(cs); } catch (...) { ch = 1; } }
        machineName = (ch == 2) ? "iter" : (ch == 3) ? "st40" : "mast";
    }
    for (char& c : machineName) c = (char)std::tolower((unsigned char)c);

    MachineEGS machine;
    if      (machineName == "mast") {machine = makeMAST(); outdir = "outputs/Grad-Shafranov/Equilibrium/MAST";}
    else if (machineName == "iter") {machine = makeITER(); outdir = "outputs/Grad-Shafranov/Equilibrium/ITER";}
    else if (machineName == "st40") {machine = makeST40(); outdir = "outputs/Grad-Shafranov/Equilibrium/ST40";}
    else { std::cerr << "[grad-shafranov] Unknown machine '" << machineName << "'\n"; return 1; }

    if (sdfPath.empty()) sdfPath = machine.sdfDefault;
    if (nR < 3) nR = machine.nRdef;
    if (nZ < 3) nZ = machine.nZdef;

    std::cout << "\n========================= Equilibrium GS Solver: " << machine.name << " ==========================\n\n"
              << "[grad-shafranov]      Domain: [" << machine.Rmin << "," << machine.Rmax << "] x [" << machine.Zmin << "," << machine.Zmax << "]\n"
              << "[grad-shafranov]        Grid: " << nR << " x " << nZ << "\n"
              << "[grad-shafranov]         Ip = " << machine.Ip << " A\n"
              << "[grad-shafranov]         p0 = " << machine.p0 << " Pa\n"
              << "[grad-shafranov]         g0 = " << machine.g0 << "\n"
              << "[grad-shafranov] (a_m, a_s) = (" << machine.alpha_m << ", " << machine.alpha_n << ")\n";

    SolverEGS S(machine, nR, nZ);
    S.useDirectBoundary = useDirectBoundary;
    if (!S.sdf.load(sdfPath, machine.sdfScale, machine.sdfDR, machine.sdfDZ)) {
        std::cout << "[grad-shafranov]  continuing WITHOUT a vessel SDF (whole domain admissible)\n";
    }

    S.precomputeGreens();
    if (useDirectBoundary)
        std::cout << "[grad-shafranov]  (using direct O(N³) boundary-flux sum, not Von Hagenow's method)\n";
    S.initialise();

    mkdir_p(outdir + "/data/Raw");
    mkdir_p(outdir + "/data/Convergence");
    mkdir_p(outdir + "/data/DiagnosticsEGS");
    mkdir_p(outdir + "/plots/plt");
    mkdir_p(outdir + "/plots/png");
    std::ofstream hist(outdir + "/data/Convergence/" + machine.name + "_convergence.csv");
    hist << "iter,rel_change,lambda,Ip,psi_axis,psi_lcfs,Raxis,Zaxis\n";

    // ------------------------------ Picard Loop ------------------------------
    bool converged = false;
    std::cout << std::scientific << std::setprecision(4);

    auto t_loopStart = clock_t::now();
    int iterCount = 0;
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

        hist << p << "," << rel << "," << S.lambda << "," << S.plasmaCurrentNow()
             << "," << S.psiAxis << "," << S.psiBnd << ","
             << S.Raxis << "," << S.Zaxis << "\n";
        std::cout << "[grad-shafranov] iter " << p
                  << "  |dPsi|/|Psi| = " << rel
                  << "\n[grad-shafranov]               lambda = " << S.lambda
                  << "\n[grad-shafranov]                 axis ~ (" << S.Raxis << ", " << S.Zaxis << ")\n";

        iterCount = p + 1;
        if (rel < tolPic) { converged = true; break; }
    }
    auto t_loopEnd = clock_t::now();
    const double picardTotalSeconds = std::chrono::duration<double>(t_loopEnd - t_loopStart).count();
    const double picardMsPerIter = (iterCount > 0) ? 1000.0 * picardTotalSeconds / iterCount : 0.0;

    S.solveCoilCurrentsCRATOS();
    S.buildCoilPsi();
    S.findAxis();
    S.buildMask();
    S.buildJphi();
    const DiagnosticsEGS D = S.computeDiagnosticsEGS();

    std::cout << "\n[grad-shafranov] " << (converged ? "CONVERGED" : "NOT converged (max iterations)")
              << " TOL_Pic = " << tolPic << "\n"
              << "[grad-shafranov] Picard Iterations to Convergence = " << iterCount << "\n"
              << "[grad-shafranov] Picard Loop Wall-clock Time      = " << picardTotalSeconds << " s"
              << "\n[grad-shafranov]                     (" << picardMsPerIter << " ms/iter)\n\n"
              << "[grad-shafranov] Magnetic Axis (R, Z) = (" << S.Raxis << ", " << S.Zaxis << ")\n"
              << "[grad-shafranov]         Psi_0 (axis) = " << S.psiAxis << " Wb/rad\n"
              << "[grad-shafranov]         Psi_l (LCFS) = " << S.psiBnd  << " Wb/rad\n"
              << "[grad-shafranov]            Ip (post) = " << S.plasmaCurrentNow() << " A\n\n"
              << "[grad-shafranov] Newton-refined X-points (target : actual):\n";
    for (size_t s = 0; s < machine.xpoints.size(); ++s) {
        std::cout << "[grad-shafranov] (" << machine.xpoints[s].first << ", " << machine.xpoints[s].second
                  << ") : (" << S.xpointsRefined[s].R << ", " << S.xpointsRefined[s].Z << ")\n";
    }

    std::cout << "\n[grad-shafranov] Equilibrium DiagnosticsEGS:\n"
              << std::fixed << std::setprecision(4)
              << "[grad-shafranov]   R0(axis) = " << S.Raxis << " m\n"
              << "[grad-shafranov]   Z0(axis) = " << S.Zaxis << " m\n"
              << "[grad-shafranov]         IT = " << D.IT / 1e6 << " MA\n"
              << "[grad-shafranov]      betap = " << D.betap << "\n"
              << "[grad-shafranov]        Vol = " << D.Volume << " m³\n" 
              << "[grad-shafranov]        q95 = " << D.q95 << "\n"
              << "[grad-shafranov]         Rg = " << D.Rg << " m\n"
              << "[grad-shafranov]         Zg = " << D.Zg << " m\n"
              << "[grad-shafranov]          a = " << D.a << " m\n"
              << "[grad-shafranov]          A = " << D.Aspect << "\n"   
              << "[grad-shafranov]        1/A = " << D.eps << "\n"
              << "[grad-shafranov]      kappa = " << D.kappa << "\n" 
              << "[grad-shafranov]      Delta = " << D.Delta << "\n"
              << "[grad-shafranov]        dSR = " << D.ShafR << " m\n"
              << "[grad-shafranov]        dSZ = " << D.ShafZ << " m\n" 
              << "[grad-shafranov]         li = " << D.li << "\n";

    // -------------------------------- output --------------------------------
    {
        std::ofstream dcsv(outdir + "/data/DiagnosticsEGS/" + machine.name + "_diagnostics.csv");
        dcsv << std::setprecision(10)
             << "quantity,value,unit\n"
             << "R0," << S.Raxis << ",m\n" << "Z0," << S.Zaxis << ",m\n"
             << "Psi_l," << S.psiBnd << ",Wb/rad\n" << "Psi_0," << S.psiAxis << ",Wb/rad\n"
             << "IT," << D.IT << ",A\n" << "betap," << D.betap << ",-\n"
             << "Volume," << D.Volume << ",m^3\n" << "q95," << D.q95 << ",-\n"
             << "Rg," << D.Rg << ",m\n" << "Zg," << D.Zg << ",m\n" << "a," << D.a << ",m\n"
             << "Aspect," << D.Aspect << ",-\n" << "eps," << D.eps << ",-\n"
             << "kappa," << D.kappa << ",-\n" << "Delta," << D.Delta << ",-\n"
             << "ShafranovShiftR," << D.ShafR << ",m\n" << "ShafranovShiftZ," << D.ShafZ << ",m\n"
             << "li," << D.li << ",-\n";

        std::ofstream lcsv(outdir + "/data/" + machine.name + "_lcfs_traced.csv");
        lcsv << std::setprecision(10) << "R,Z\n";
        const auto lcfsPts = S.traceFluxSurface(S.psiBnd, 360);
        for (const auto& p : lcfsPts) lcsv << p.R << "," << p.Z << "\n";
        if (!lcfsPts.empty()) lcsv << lcfsPts[0].R << "," << lcfsPts[0].Z << "\n";
    }
    writeFieldCSV(outdir + "/data/" + machine.name + "_psi_masked.csv", S, S.psiTot, "psi", /*maskOutsideVessel=*/true);
    writeFieldCSV(outdir + "/data/" + machine.name + "_psi_full.csv", S, S.psiTot, "psi", /*maskOutsideVessel=*/false);
    writeFieldCSV(outdir + "/data/" + machine.name + "_psi_plasma.csv", S, S.psiPl, "psi_pl");
    writeFieldCSV(outdir + "/data/" + machine.name + "_jphi.csv", S, S.Jphi, "Jphi");
    writeVesselCSV(outdir + "/data/" + machine.name + "_vessel_sdf.csv", S.sdf);
    writeCoilsCSV(outdir + "/data/" + machine.name + "_coils.csv", machine, S.Icoil);
    writeSliceCSV(outdir + "/data/" + machine.name + "_slice.csv", S);
    writeEquilibriumPlot(outdir, S, S.sdf.loaded, doPlot);

    std::cout << "\n";
    std::cout << "======================================================================\n";
    std::cout << " Grad-Shafranov Picard Performance\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::left  << std::setw(14) << " Case"
              << std::right << std::setw(12) << "Iterations"
              << std::setw(24) << "Iteration Time [ms]"
              << std::setw(18) << "Total Time [s]" << "\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::left  << std::setw(14) << (" " + machine.name)
              << std::right << std::setw(12) << iterCount
              << std::fixed << std::setprecision(2)
              << std::setw(24) << picardMsPerIter
              << std::setprecision(3)
              << std::setw(18) << picardTotalSeconds << "\n";
    std::cout << "======================================================================\n\n";

    if (!converged)
        std::cerr << "[grad-shafranov] (NOT converged within " << maxIter << " iterations — treat with caution)\n\n";

    std::cerr << "[grad-shafranov] Wrote data to " << outdir << "/data\n";
    std::cerr << "[grad-shafranov] Wrote plots to " << outdir << "/plots/png\n";

    std::cout << "\nDone.\n";
    return converged ? 0 : 2;
}