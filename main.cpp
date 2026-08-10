// ============================================================================
// main.cpp - Entry for Ideal Magnetohydrodynamics / Grad-Shafranov solver
// ============================================================================
// 
// Author:     Uzair Abdullah
//             Centre for Scientific Computing
//             Department of Physics
//             University of Cambridge
// Email:      ua247@cam.ac.uk
// Date:       07 August 2026
// Supervisor: Dr Maria Nikodemou
// 
// Copyright (C) 2026 Uzair Abdullah
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
// 
// This solver accompanies the MPhil dissertation "Magnetohydrodynamic
// Simulations in Complex Geometries". The menu structure mirrors the
// on-disk source layout:
//   mhd/1d                 -- 1D Riemann problems
//   mhd/2d/blast           -- MHD blast wave
//   mhd/2d/briowu          -- Brio-Wu shock tube
//   mhd/2d/kelvinhelmholtz -- Kelvin-Helmholtz instability
//   mhd/2d/orszagtang      -- Orszag-Tang vortex
//   mhd/2d/reverseb        -- Reverse-B shock tube
//   mhd/2d/rotor           -- MHD rotor
//   mhd/3d                 -- 3D reactor-vessel rotor and blast wave
//   gs                     -- Grad-Shafranov equilibrium solver
// 
// Build instructions:
//   g++ -O3 -std=c++17 -Xpreprocessor -fopenmp \
//       -I$(brew --prefix libomp)/include \
//       -L$(brew --prefix libomp)/lib \
//       -I/opt/homebrew/opt/eigen/include/eigen3 \
//       -Igs \
//       -Imhd/1d \
//       -Imhd/2d/blast -Imhd/2d/briowu -Imhd/2d/kelvinhelmholtz \
//       -Imhd/2d/orszagtang -Imhd/2d/reverseb -Imhd/2d/rotor \
//       -Imhd/3d \
//       -lomp -march=native \
//       main.cpp $(find gs mhd -name '*.cpp') \
//       -o solver
// 
// Usage:
//   ./solver                     Interactive top-level menu (MHD / GS)
//   ./solver --solver mhd        Jump straight into the MHD Solver menu,
//                                skipping the top-level menu
//   ./solver --solver gs         Jump straight into the GS Solver menu,
//                                skipping the top-level menu
//   ./solver <test_id>           Run a single test directly, e.g.
//                                ./solver rotor2dreact
// ============================================================================
#include <iostream>
#include <string>
#include <limits>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include "BrioWu1D.h"
#include "BrioWu2DRot.h"
#include "BrioWu2DRef.h"
#include "Rotor2D.h"
#include "Rotor2DRef.h"
#include "Rotor2DCirc.h"
#include "Rotor2DReactor.h"
#include "Rotor3DReactor.h"
#include "Orszag2D.h"
#include "Blast2D.h"
#include "Blast2DRef.h"
#include "Blast2DCirc.h"
#include "Blast2DReactor.h"
#include "Blast3DReactor.h"
#include "KelvinHelmholtz2D.h"
#include "ShockTrans1D.h"
#include "Rarefac1D.h"
#include "ReverseB1D.h"
#include "ReverseB2DRef.h"

#include "ValidationGS.h"
#include "PerturbedGS.h"
#include "EquilibriumGS.h"

// ============================================================================
// Text Formatting Utilities
// ============================================================================

static size_t utf8_length(const std::string& s) {
    size_t len = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            ++len;
        }
    }
    return len;
}

static std::string justify_text(const std::string& text, int width, bool left_align = true) {
    if (utf8_length(text) >= static_cast<size_t>(width)) {
        return text;
    }
    std::stringstream ss;
    if (left_align) {
        ss << std::left << std::setw(width) << text;
    } else {
        ss << std::right << std::setw(width) << text;
    }
    return ss.str();
}

static void print_justified_paragraph(const std::string& text, int width) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        words.push_back(word);
    }

    if (words.empty()) return;

    std::vector<std::vector<std::string>> lines;
    std::vector<std::string> current;
    int current_len = 0;

    for (const auto& w : words) {
        int wlen = static_cast<int>(utf8_length(w));
        int extended_len = current.empty() ? wlen : current_len + 1 + wlen;

        if (extended_len <= width || current.empty()) {
            current.push_back(w);
            current_len = extended_len;
        } else {
            lines.push_back(current);
            current = {w};
            current_len = wlen;
        }
    }
    if (!current.empty()) {
        lines.push_back(current);
    }

    for (size_t li = 0; li < lines.size(); ++li) {
        const auto& lwords = lines[li];
        bool is_last_line = (li == lines.size() - 1);

        if (is_last_line || lwords.size() == 1) {
            std::string line;
            for (size_t i = 0; i < lwords.size(); ++i) {
                if (i) line += " ";
                line += lwords[i];
            }
            std::cout << line << "\n";
            continue;
        }

        int total_word_len = 0;
        for (const auto& w : lwords) {
            total_word_len += static_cast<int>(utf8_length(w));
        }
        int gaps = static_cast<int>(lwords.size()) - 1;
        int total_spaces = std::max(gaps, width - total_word_len);
        int base_spaces = total_spaces / gaps;
        int remainder = total_spaces % gaps;

        std::string line;
        for (size_t i = 0; i < lwords.size(); ++i) {
            line += lwords[i];
            if (static_cast<int>(i) < gaps) {
                int n_spaces = base_spaces + (static_cast<int>(i) < remainder ? 1 : 0);
                line += std::string(n_spaces, ' ');
            }
        }
        std::cout << line << "\n";
    }
}

static void print_menu_banner(const std::string& text, int width = 80) {
    std::string banner(width, '=');
    std::cout << banner << "\n";
    std::cout << banner << "\n";
    int padding = std::max(0, (width - static_cast<int>(utf8_length(text))) / 2);
    std::cout << std::string(padding, ' ') << text << "\n";
    std::cout << banner << "\n";
    std::cout << banner << "\n\n";
}

static void print_banner(const std::string& text, int width = 80) {
    std::string banner(width, '=');
    std::cout << banner << "\n";
    int padding = std::max(0, (width - static_cast<int>(utf8_length(text))) / 2);
    std::cout << std::string(padding, ' ') << text << "\n";
    std::cout << banner << "\n\n";
}

static void print_section_header(const std::string& text, int width = 80) {
    std::cout << text << "\n";
    std::cout << std::string(text.length(), '-') << "\n\n";
}

static void print_centred_equation(const std::string& eq, int width = 90) {
    int len = static_cast<int>(utf8_length(eq));
    int padding = std::max(0, (width - len) / 2);
    std::cout << "\n" << std::string(padding, ' ') << eq << "\n\n";
}

// ============================================================================
// Test Case Structure
// ============================================================================

struct TestCase {
    std::string name;
    std::string description;
    std::string explanation;
    std::string glm_note;
    std::string reference;
    void (*run)();
};

// ============================================================================
// Display Functions
// ============================================================================

void display_test_info(const TestCase& test, int width = 90) {
    print_banner(test.name, width);
    
    std::cout << "DESCRIPTION:\n";
    print_justified_paragraph(test.description, width);
    std::cout << "\n";
    
    std::cout << "DETAILS:\n";
    print_justified_paragraph(test.explanation, width);
    std::cout << "\n";
    
    std::cout << "HYPERBOLIC DIVERGENCE CLEANING (GLM):\n";
    print_justified_paragraph(test.glm_note, width);
    std::cout << "\n";
    
    std::cout << "REFERENCE: " << test.reference << "\n\n";
}

void display_boundary_condition_info(const std::string& bc_type, int width = 90) {
    std::cout << "\n----- Boundary Condition: " << bc_type << " -----\n\n";
    if (bc_type == "Transmissive") {
        std::string text = "Waves and disturbances pass through the domain boundaries without reflection. The boundary values are copied from the nearest interior cell. This is useful for open domains where disturbances should exit naturally.";
        print_justified_paragraph(text, width);
    } else if (bc_type == "Reflective") {
        std::string text = "Normal components of velocity and magnetic field are zero at the boundary, a perfectly reflecting wall. Tangential components are mirrored. This is useful for confined domains with rigid walls.";
        print_justified_paragraph(text, width);
    }
    std::cout << "\n";
}

// ============================================================================
// SDF & Ghost Fluid Method Boundary Construction
// ============================================================================

void display_sdf_geometry_explanation(int width = 90) {
    std::cout << "SDF & GHOST FLUID METHOD BOUNDARY CONSTRUCTION\n";
    std::cout << std::string(62, '-') << "\n\n";

    print_justified_paragraph(
        "Every reflective wall in this solver, whether square, circular, or reactor-shaped, "
        "is represented implicitly by a signed distance function (SDF) φ(x). This satisfies "
        "the Eikonal equation", width);
    print_centred_equation("|∇φ(x)| = 1", width);
    print_justified_paragraph(
        "almost everywhere, so |φ(x)| is the exact Euclidean distance from x to the nearest "
        "boundary point. By convention φ < 0 inside the fluid domain, φ > 0 in the solid "
        "region, and φ = 0 exactly on the wall. The local outward unit normal is recovered "
        "directly from the gradient,", width);
    print_centred_equation("n(x) = ∇φ(x)", width);
    print_justified_paragraph(
        "Cell classification, ghost-cell reflection at the wall, and ghost-fluid boundary "
        "reconstruction all read from φ and n. Each geometry only has to supply its own "
        "construction of φ, and the rest of the boundary machinery is shared.", width);
    std::cout << "\n";

    std::cout << "SQUARE / RECTANGULAR BOX:\n";
    print_justified_paragraph(
        "The box wall uses an exact, closed-form SDF. After transforming the sample point "
        "into the box's own, possibly rotated, frame, two cases are combined. Outside the "
        "box, the distance is the Euclidean distance to the nearest corner or edge, built "
        "from the clamped per-axis penetration depths", width);
    print_centred_equation("dx = max(x₀ - x, 0, x - x₁)", width);
    print_centred_equation("dy = max(y₀ - y, 0, y - y₁)", width);
    print_justified_paragraph("which gives the outside distance", width);
    print_centred_equation("φ = √(dx² + dy²)", width);
    print_justified_paragraph(
        "Inside the box, φ is instead the negative of the smallest perpendicular distance "
        "to any of the four edges,", width);
    print_centred_equation("φ = -min(x - x₀, x₁ - x, y - y₀, y₁ - y)", width);
    print_justified_paragraph(
        "This is the standard exact SDF for an axis-aligned rectangle, and it stays exact "
        "under rotation because the rotation is applied to the sample point before φ is "
        "evaluated. No tabulation or interpolation is needed.", width);
    std::cout << "\n";

    std::cout << "CIRCULAR BOUNDARY:\n";
    print_justified_paragraph(
        "The circular wall uses the simplest possible analytic SDF, the radial distance "
        "from the optionally translated domain centre minus a fixed radius R,", width);
    print_centred_equation("φ = √(x² + y²) - R", width);
    print_justified_paragraph(
        "This is exact everywhere and costs almost nothing to evaluate. Its gradient, the "
        "outward wall normal, is just the unit radial direction, so no finite-difference "
        "gradient estimate is required. The square-versus-circle comparison isolates the "
        "effect of wall curvature from the additional shaping found in a real reactor "
        "cross-section.", width);
    std::cout << "\n";

    std::cout << "REACTOR GEOMETRY:\n";
    print_justified_paragraph(
        "Realistic tokamak cross-sections admit no closed-form boundary normal or signed "
        "distance, so each vessel's SDF is instead precomputed offline from digitised wall "
        "data and stored as a lookup table on a Cartesian grid. This is consistent with the "
        "level-set representation used internally by the Grad-Shafranov equilibrium solver. "
        "On load, the table is re-centred on its own bounding box and rescaled by a fixed "
        "SDF_SCALE factor so the vessel occupies a consistent footprint in the computational "
        "domain regardless of the resolution of the source table. Values at arbitrary points "
        "are obtained by bilinear interpolation between the four surrounding table nodes, "
        "and the wall normal is recovered by central differencing φ over half the local "
        "table spacing. Since all three vessels are axisymmetric, the three-dimensional SDF "
        "is obtained by revolving the tabulated poloidal cross-section about the device axis "
        "rather than tabulating a full volume. A query point (x, y, z) is projected onto the "
        "cross-section by its distance from the toroidal axis and its axial coordinate,",
        width);
    print_centred_equation("r = √((x - x₀)² + z²) - R₀", width);
    print_centred_equation("ζ = y", width);
    print_justified_paragraph(
        "before (r, ζ) is bilinearly interpolated in the 2D table. R₀, the toroidal major "
        "radius, and a cross-sectional scale factor are configurable per run, placing and "
        "sizing the same tabulated wall contour consistently for each machine.", width);
    std::cout << "\n";

    std::cout << "GHOST FLUID METHOD COUPLING:\n";
    print_justified_paragraph(
        "The GFM reflection rule is unchanged across all geometries. Normal velocity and "
        "field are negated, tangential components preserved, and the GLM auxiliary field ψ "
        "sign-reversed at the boundary. An SDF representation changes only the source of "
        "the normal and interior sample point, not the rule itself. For a ghost cell at "
        "position x_g with φ(x_g) > 0, the interior image point is obtained by reflecting a "
        "distance 2·φ(x_g) along the local normal,", width);
    print_centred_equation("x_i = x_g - 2·φ(x_g)·n(x_g)", width);
    print_justified_paragraph(
        "with the primitive state recovered by bilinear, or trilinear in 3D, interpolation "
        "from the surrounding interior cells. If the mirrored point still lies outside the "
        "vessel, near strongly curved wall sections, a shallower probe along the same "
        "normal is attempted before falling back to a fixed ambient state. Close to the "
        "wall, where the WENO3 stencil would draw on cells whose ghost state is not yet "
        "consistent with local curvature, the solver reverts to first-order HLLD on "
        "cell-centre states, returning to full WENO3-HLLD once the stencil lies entirely "
        "within the smooth interior. This fallback triggers on a local gradient threshold "
        "rather than by test problem, so it is test-agnostic. It is exercised mainly in the "
        "blast wave tests, whose sharper pressure jump stresses the near-wall reconstruction "
        "far more than the rotor's smoother data. As an additional robustness measure near "
        "strongly curved, strongly magnetised walls, each SSP-RK3 substage is checked "
        "against the peak wave speed obtained before the step. It is rejected and retried "
        "at half the time step if that speed grows by more than a factor of two, with "
        "ghost-cell reconstruction, GLM cleaning, and floor enforcement reapplied at every "
        "substage.", width);
    std::cout << "\n";
}

void display_glm_explanation(int width = 90) {
    print_banner("HYPERBOLIC DIVERGENCE CLEANING (GLM)", width);
    
    std::cout << "WHAT IS GLM?\n";
    print_justified_paragraph(
        "GLM, the Generalised Lagrange Multiplier method, is a hyperbolic "
        "divergence-cleaning technique. It maintains the divergence-free constraint",
        width);
    print_centred_equation("∇ · B = 0", width);
    print_justified_paragraph(
        "in ideal MHD by introducing an auxiliary scalar field (Dedner et al., 2002), which "
        "carries spurious magnetic monopoles created by numerical discretisation rapidly "
        "away from the computational domain.", width);
    std::cout << "\n";
    
    std::cout << "WHY IS IT NEEDED?\n";
    std::string text2 = "In numerical simulations, discretisation errors can create spurious magnetic monopoles, non-physical sources and sinks of the B-field. These monopoles produce artificial forces, degrade solution accuracy, and can cause numerical instability. GLM actively suppresses these errors by introducing an auxiliary wave that transports monopoles away from regions of physical interest.";
    print_justified_paragraph(text2, width);
    std::cout << "\n";
    
    std::cout << "HOW DOES GLM WORK?\n";
    std::cout << "  1. Introduces auxiliary field ψ that propagates monopoles away\n";
    std::cout << "  2. Monopoles travel at speed c_h (hyperbolic speed)\n";
    std::cout << "  3. They exit the domain via transmissive boundaries\n";
    std::cout << "  4. For reflective boundaries, monopoles reflect back\n\n";
    
    std::cout << "WHEN TO ENABLE GLM:\n";
    std::cout << "  ENABLE for:\n";
    std::cout << "    - Complex field structures (Orszag-Tang, rotor, blast)\n";
    std::cout << "    - Transmissive boundaries (monopoles must exit)\n";
    std::cout << "    - Long-time integrations\n";
    std::cout << "    - When field rotation or complex reconnection occurs\n\n";
    std::cout << "  DISABLE for:\n";
    std::cout << "    - Simple, smooth initial conditions\n";
    std::cout << "    - Reflective boundaries only (monopoles trapped)\n";
    std::cout << "    - Short-time simulations where cost matters\n\n";
    
    std::cout << "PERFORMANCE IMPACT:\n";
    std::cout << "  - Adds approximately 10-15% computational cost\n";
    std::cout << "  - Requires slightly smaller CFL timesteps\n";
    std::cout << "  - Essential for stability in complex flows\n\n";
    
    std::cout << "REFERENCE: Dedner et al. (2002)\n\n";
}

void display_solver_explanation(int width = 90) {
    print_banner("IDEAL MHD SOLVER", width);
    
    std::cout << "NUMERICAL SCHEME COMPONENTS\n";
    std::string text1 = "This ideal MHD solver integrates multiple advanced numerical techniques to ensure accuracy, stability, and robustness across diverse flow regimes. The scheme combines a finite-volume framework with specialised fluxes, high-order reconstruction, stable time integration, divergence control, and boundary treatments. It carries forward unmodified from the flat-wall validation tests into the complex- and reactor-geometry tests. The only difference between them is the domain boundary representation and its coupling to the ghost fluid method.";
    print_justified_paragraph(text1, width);
    std::cout << "\n";
    
    std::cout << "RIEMANN SOLVER: HLLD\n";
    std::string text2 = "The Harten-Lax-van Leer-Discontinuities (HLLD) Riemann solver (Miyoshi and Kusano, 2005) resolves contact discontinuities and intermediate wave structures more accurately than simpler HLL solvers. This multi-state approach is essential for capturing the complex wave interactions that arise in magnetised flows.";
    print_justified_paragraph(text2, width);
    std::cout << "\n";
    
    std::cout << "SPATIAL RECONSTRUCTION: WENO3\n";
    std::string text3 = "Weighted Essentially Non-Oscillatory (WENO) reconstruction of third order (Jiang and Shu, 1996) provides high-order accuracy whilst suppressing spurious oscillations near discontinuities. WENO achieves this through adaptive weighting of multiple stencil polynomials based on local smoothness indicators.";
    print_justified_paragraph(text3, width);
    std::cout << "\n";
    
    std::cout << "TIME INTEGRATION: SSP-RK3\n";
    std::string text4 = "Strong Stability-Preserving (SSP) Runge-Kutta time stepping of third order (Gottlieb et al., 2001) ensures that the method maintains the stability properties of the spatial discretisation under CFL-limited timesteps, preventing unphysical oscillations and maintaining positivity constraints. All tests use a Courant-Friedrichs-Lewy (CFL) number of 0.6, which controls the maximum allowable timestep based on the fastest wave speed in the domain, ensuring numerical stability and accurate shock capture.";
    print_justified_paragraph(text4, width);
    std::cout << "\n";
    
    std::cout << "DIVERGENCE CLEANING: GLM\n";
    print_justified_paragraph(
        "Hyperbolic divergence cleaning via the Generalised Lagrange Multiplier, or GLM, "
        "method (Dedner et al., 2002) maintains the constraint", width);
    print_centred_equation("∇ · B = 0", width);
    print_justified_paragraph(
        "throughout the integration. This matters most in complex geometries and when wall "
        "interactions generate monopole pollution.", width);
    std::cout << "\n";
    
    std::cout << "BOUNDARY TREATMENT: GHOST FLUID METHOD\n";
    std::string text6 = "The Ghost Fluid Method (GFM) (Fedkiw et al., 1999) handles rigid boundaries by mirroring boundary-normal velocity and magnetic field components while preserving tangential components. This enables accurate reflection of waves at plasma-wall interfaces. The method extends from flat walls to a general signed-distance-function boundary without changing the reflection rule itself.";
    print_justified_paragraph(text6, width);
    std::cout << "\n";

    display_sdf_geometry_explanation(width);
}

// ============================================================================
// Input Handling
// ============================================================================

int get_valid_choice(int min, int max)
{
    int choice;

    while (true) {

        std::cout << "Enter choice (" << min << "-" << max << "): ";

        if (std::cin >> choice) {

            if (choice >= min && choice <= max) {
                std::cin.ignore(
                    std::numeric_limits<std::streamsize>::max(),
                    '\n'
                );
                return choice;
            }

            std::cout << "Invalid option. Please enter a number between "
                      << min << " and " << max << ".\n";
        }
        else {

            std::cout << "Invalid input. Please enter a number.\n";

            std::cin.clear();
            std::cin.ignore(
                std::numeric_limits<std::streamsize>::max(),
                '\n'
            );
        }
    }
}

int get_choice_with_help(int min, int max, bool has_help = false)
{
    std::string input;

    while (true) {

        std::cout << "Enter choice (" << min << "-" << max;
        if (has_help) std::cout << " or ?";
        std::cout << "): ";

        if (std::getline(std::cin, input)) {

            if (has_help && input == "?") {
                return 999;
            }

            try {
                int choice = std::stoi(input);
                if (choice >= min && choice <= max) {
                    return choice;
                }

                std::cout << "Invalid option. Please enter a number between "
                          << min << " and " << max;
                if (has_help) std::cout << " or ?";
                std::cout << ".\n";
            } catch (...) {
                std::cout << "Invalid input. Please enter a number";
                if (has_help) std::cout << " or ?";
                std::cout << ".\n";
            }
        }
    }
}

static void press_enter_to_continue() {
    std::cout << "\nPress Enter to continue...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

// ============================================================================
// Generic Test-List Menu
// ============================================================================

void run_menu(const std::vector<TestCase>& tests, const std::string& category, int width = 90)
{
    while (true) {
        print_menu_banner("SELECT 1D TEST", width);

        for (size_t i = 0; i < tests.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << tests[i].name << "\n";
        }

        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int selection = get_choice_with_help(0, static_cast<int>(tests.size()), true);

        if (selection == 0) {
            return;
        }

        if (selection == 999) {  
            print_banner("1D TESTS OVERVIEW", width);
            
            for (size_t i = 0; i < tests.size(); ++i) {
                display_test_info(tests[i], width);
                if (i < tests.size() - 1) {
                    press_enter_to_continue();
                }
            }
            continue;
        }

        const TestCase& selected_test = tests[selection - 1];

        while (true) {
            print_banner("ARE YOU SURE YOU WANT TO RUN THIS TEST?", width);
            std::cout << "Test: " << selected_test.name << "\n\n";
            std::cout << "  ?. Explain\n";
            std::cout << "  1. Yes, run it\n";
            std::cout << "  0. Back\n\n";

            int confirm = get_choice_with_help(0, 1, true);
            if (confirm == 999) {  
                display_test_info(selected_test, width);
                press_enter_to_continue();
                continue;
            }
            if (confirm == 0) {
                break;
            }

            std::cout << "\nRunning " << selected_test.name << "...\n\n";
            selected_test.run();
            return;
        }
    }
}

// ============================================================================
// Generic Single-Test Menu
// ============================================================================

struct SingleTest {
    std::string title;
    std::string description;
    std::string details;
    std::string glm_note;
    std::string reference;
    void (*run)();
};

void run_single_test_menu(const SingleTest& test, int width = 90)
{
    while (true) {
        print_menu_banner(test.title, width);
        std::cout << "  1. Run\n";
        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 1, true);
        if (choice == 0) return;

        if (choice == 999) {
            print_banner(test.title, width);
            std::cout << "DESCRIPTION:\n";
            print_justified_paragraph(test.description, width);
            std::cout << "\n";
            std::cout << "DETAILS:\n";
            print_justified_paragraph(test.details, width);
            std::cout << "\n";
            std::cout << "HYPERBOLIC DIVERGENCE CLEANING (GLM):\n";
            print_justified_paragraph(test.glm_note, width);
            std::cout << "\n";
            std::cout << "REFERENCE: " << test.reference << "\n\n";
            continue;
        }

        std::cout << "\nRunning " << test.title << "...\n\n";
        test.run();
        return;
    }
}

// ============================================================================
// Brio-Wu Shock Tube
// ============================================================================

void run_brio_menu(int width = 90)
{
    while (true) {
        print_menu_banner("BRIO-WU SHOCK TUBE", width);
        std::cout << "  1. Rotated Discontinuity\n";
        std::cout << "  2. Reflective Box\n";
        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 2, true);
        if (choice == 0) return;

        if (choice == 999) {  
            print_banner("BRIO-WU SHOCK TUBE - ROTATED DISCONTINUITY", width);
            
            std::cout << "DESCRIPTION:\n";
            std::string desc1 = "Two-dimensional shock tube with shock at 45-degree angle to grid, testing oblique discontinuities and multi-dimensional interactions.";
            print_justified_paragraph(desc1, width);
            std::cout << "\n";
            
            std::cout << "DETAILS:\n";
            std::string details1 = "The shock is positioned at a 45-degree angle to the computational grid. This tests the code's ability to handle oblique discontinuities and multi-dimensional MHD interactions that naturally arise in real physical systems. The rotated configuration is particularly useful for open domains where disturbances should exit the computational region naturally without reflecting from artificial boundaries.";
            print_justified_paragraph(details1, width);
            std::cout << "\n";
            
            std::cout << "OUTPUTS & VISUALISATION:\n";
            std::string out1 = "One snapshot at final time t=0.1 producing a 2D density field (ρ) contour plot. The plot shows the shock structure at a 45-degree angle and the surrounding flow patterns.";
            print_justified_paragraph(out1, width);
            std::cout << "\n";
            
            std::cout << "HYPERBOLIC DIVERGENCE CLEANING (GLM):\n";
            std::string glm1 = "STRONGLY RECOMMENDED: Enable GLM. Two-dimensional extensions of the Brio-Wu test involve complex field structures and potential divergence pollution from oblique waves.";
            print_justified_paragraph(glm1, width);
            std::cout << "\nBoundary Condition: Transmissive (waves pass through boundaries)\n\n";
            
            std::cout << "\nPress Enter for the reflective variant...";
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            
            print_banner("BRIO-WU SHOCK TUBE - REFLECTIVE BOX", width);
            
            std::cout << "DESCRIPTION:\n";
            std::string desc2 = "Two-dimensional shock tube confined in a rectangular domain with reflective walls, testing wave reflections and stability in confined geometries.";
            print_justified_paragraph(desc2, width);
            std::cout << "\n";
            
            std::cout << "DETAILS:\n";
            std::string details2 = "The shock is aligned with the computational grid and confined within a rectangular domain with reflecting walls. Outgoing waves reflect from the boundaries and propagate back into the domain, creating complex multi-wave structures and testing the solver's stability when waves interact with rigid boundaries. Together with the reversed-field problem, this validates the ghost fluid method's flat-wall reflection rule before extending it to a general SDF boundary.";
            print_justified_paragraph(details2, width);
            std::cout << "\n";
            
            std::cout << "OUTPUTS & VISUALISATION:\n";
            std::string out2 = "Seven snapshots at times t=0.0,0.1,0.2,0.3,0.4,0.5,0.6 producing a 2D density field (ρ) contour plot at each output time. This sequence captures the shock propagation, wall reflections, and complex interaction patterns as waves bounce between boundaries.";
            print_justified_paragraph(out2, width);
            std::cout << "\n";
            
            std::cout << "HYPERBOLIC DIVERGENCE CLEANING (GLM):\n";
            std::string glm2 = "STRONGLY RECOMMENDED: Enable GLM. GLM is essential for maintaining stability in both rotated and reflective variants, especially where waves interact with boundaries.";
            print_justified_paragraph(glm2, width);
            std::cout << "\nBoundary Condition: Reflective (normal velocity/B-field zero at walls)\n\n";

            std::cout << "REFERENCE: Brio and Wu (1988); Miyoshi and Kusano (2005)\n\n";
            continue;
        }

        if (choice == 1) {
            std::cout << "\nRunning Brio-Wu rotated test...\n\n";
            run_brio_rot();
        } else {
            std::cout << "\nRunning Brio-Wu reflective test...\n\n";
            run_brio_ref();
        }
        return;
    }
}

// ============================================================================
// Reverse-B Shock Tube
// ============================================================================

const SingleTest& reverseb_2d_test()
{
    static const SingleTest test = {
        "REVERSE-B SHOCK TUBE",
        "Two-dimensional shock tube with a reversed transverse magnetic field in a reflective box.",
        "This test features a shock tube where the transverse magnetic field (B_y) not only "
        "changes magnitude but reverses sign and direction across the discontinuity. In the "
        "reflective box configuration, waves reflect off the walls and interact with the "
        "field reversal region, creating complex multi-wave structures. This scenario models "
        "situations similar to magnetic reconnection and tests the solver's robustness when "
        "handling strongly anti-parallel magnetic fields and their interactions with "
        "boundaries. OUTPUTS: Five snapshots at times t=0.0,0.16,0.32,0.48,0.53 producing a "
        "2D density field (ρ) contour plot at each output time.",
        "STRONGLY RECOMMENDED: Enable GLM. Anti-parallel magnetic field reversal in two "
        "dimensions with boundary reflections creates significant challenges for maintaining "
        "div(B) = 0. GLM actively suppresses monopole pollution generated by the field "
        "reversal and wall interactions.",
        "Brio and Wu (1988)",
        run_reverse_b_ref
    };
    return test;
}

void run_reverseb_menu(int width = 90)
{
    run_single_test_menu(reverseb_2d_test(), width);
}

// ============================================================================
// Orszag-Tang Vortex
// ============================================================================

const SingleTest& orszag_test()
{
    static const SingleTest test = {
        "ORSZAG-TANG VORTEX",
        "Complex vortex with interactions between magnetic field and velocity shear.",
        "The Orszag-Tang vortex is a canonical two-dimensional ideal MHD test combining a "
        "prescribed velocity vortex with an anti-parallel magnetic field configuration. The "
        "interaction between the shear flow and magnetic field creates complex nonlinear "
        "dynamics, generating small-scale current-sheet structures. It is particularly "
        "stringent for divergence-cleaning methods. Without GLM, solutions typically "
        "diverge or develop unphysical oscillations, whereas GLM preserves the fine-scale "
        "filament structures characteristic of developed MHD turbulence.",
        "ESSENTIAL: Enable GLM. This test is particularly sensitive to divergence errors "
        "because the anti-parallel field configuration and high vorticity naturally generate "
        "spurious monopoles. Without GLM, solutions typically diverge or develop unphysical "
        "oscillations. With GLM, the method remains stable and accurate.",
        "Miyoshi and Kusano (2005)",
        run_orszag
    };
    return test;
}

void run_orszag_menu(int width = 90)
{
    run_single_test_menu(orszag_test(), width);
}

// ============================================================================
// Kelvin-Helmholtz Instability
// ============================================================================

const SingleTest& kelvin_helmholtz_test()
{
    static const SingleTest test = {
        "KELVIN-HELMHOLTZ INSTABILITY",
        "Shear-driven instability at a density/velocity discontinuity with a weak magnetic field.",
        "The Kelvin-Helmholtz instability develops at a shear layer where two fluid streams "
        "with different densities and velocities meet, with a weak transverse magnetic field "
        "present throughout. This test is included in the solver as a supplementary "
        "long-time-integration case but is not part of the dissertation's validation "
        "chapters.",
        "RECOMMENDED: Enable GLM. The Kelvin-Helmholtz instability creates growing shear "
        "layers with complex field interactions. GLM helps maintain numerical stability "
        "during the nonlinear growth phase.",
        "Miyoshi and Kusano (2005)",
        run_kelvin_helmholtz2d
    };
    return test;
}

void run_kh_menu(int width = 90)
{
    run_single_test_menu(kelvin_helmholtz_test(), width);
}

// ============================================================================
// MHD Rotor
// ============================================================================

void run_rotor_menu(int width = 90)
{
    while (true) {
        print_menu_banner("MHD ROTOR", width);
        std::cout << "  Transmissive Boundaries\n";
        std::cout << "  1. Transmissive\n\n";
        std::cout << "  Reflective Boundaries\n";
        std::cout << "  2. Square\n";
        std::cout << "  3. Circular\n";
        std::cout << "  4. Reactor\n\n";
        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 4, true);
        if (choice == 0) return;

        if (choice == 999) {
            print_banner("MHD ROTOR", width);

            std::cout << "DESCRIPTION:\n";
            std::string desc =
                "Dense, rapidly rotating magnetised disc in a uniform static background, with "
                "transmissive, square, circular, and reactor-vessel boundary variants.";
            print_justified_paragraph(desc, width);
            std::cout << "\n";

            std::cout << "DETAILS:\n";
            print_justified_paragraph(
                "A dense, rapidly rotating disc is embedded in a uniform, magnetised, static "
                "background, following the construction of Miyoshi and Kusano. Writing r for "
                "the distance from the disc centre, density and velocity are piecewise. "
                "There is a solid-body rotating core for r ≤ r₀ = 0.10, a linearly tapered "
                "transition for r₀ < r ≤ r₁ = 0.115, and a static ambient medium for r > r₁, "
                "with uniform pressure p = 1.0 and background field", width);
            print_centred_equation("B = (0, 2.5 / √(4π), 0)", width);
            print_justified_paragraph(
                "The transmissive variant uses an open, unbounded domain. The square and "
                "circular variants then isolate wall curvature. The square boundary is "
                "rotated 45 degrees so the wall is not aligned with the grid, and its "
                "corners introduce visible asymmetry into the otherwise near-axisymmetric "
                "rotor pattern, while on the circular wall the normal instead rotates "
                "continuously along the boundary and the pattern stays closer to "
                "axisymmetric. The reactor variant applies the same initial data to a "
                "realistic ITER, MAST, or ST40 vessel cross-section, prompting for a vessel "
                "choice when run. The dissertation demonstrates this on the shaped, "
                "tight-aspect-ratio MAST and ST40 vessels, though this solver allows any "
                "vessel to be selected.", width);
            std::cout << "\n";

            std::cout << "OUTPUTS & VISUALISATION:\n";
            std::cout << "  Transmissive: Two snapshots at t=0.1 and t=0.25\n";
            std::cout << "  Square: Five snapshots at t=0.0,0.1,0.25,0.5,1.0\n";
            std::cout << "  Circular: Five snapshots at t=0.0,0.1,0.25,0.5,1.0\n";
            std::cout << "  Reactor: Five snapshots at t=0.0,0.1,0.25,0.5,1.0\n";
            std::cout << "  All variants output a 2D density field (ρ) contour plot showing rotor structure evolution.\n\n";

            std::cout << "HYPERBOLIC DIVERGENCE CLEANING (GLM):\n";
            std::string glm =
                "ESSENTIAL: Enable GLM. The rotating magnetic field structure wrapping around "
                "the rotor creates complex div(B) challenges, especially for the longer-time "
                "reflective variants, where internal or curved boundary structure creates "
                "additional sites for monopole pollution to accumulate.";
            print_justified_paragraph(glm, width);
            std::cout << "\n";

            display_sdf_geometry_explanation(width);
            std::cout << "REFERENCE: Miyoshi and Kusano (2005)\n\n";
            continue;
        }

        if (choice == 1) {
            std::cout << "\nRunning transmissive rotor...\n\n";
            run_rotor();
        } else if (choice == 2) {
            std::cout << "\nRunning square (rotated box) reflective rotor...\n\n";
            run_rotor_ref();
        } else if (choice == 3) {
            std::cout << "\nRunning circular reflective rotor...\n\n";
            run_rotor_circ();
        } else {
            std::cout << "\nRunning reactor-vessel rotor test...\n\n";
            run_rotor_reactor();
        }
        return;
    }
}

// ============================================================================
// MHD Blast Wave
// ============================================================================

void run_blast_menu(int width = 90)
{
    while (true) {
        print_menu_banner("MHD BLAST WAVE", width);
        std::cout << "  Transmissive Boundaries\n";
        std::cout << "  1. Transmissive\n\n";
        std::cout << "  Reflective Boundaries\n";
        std::cout << "  2. Square\n";
        std::cout << "  3. Circular\n";
        std::cout << "  4. Reactor\n\n";
        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 4, true);
        if (choice == 0) return;

        if (choice == 999) {
            print_banner("MHD BLAST WAVE", width);

            std::cout << "DESCRIPTION:\n";
            std::string desc =
                "High-pressure region explosion with strong magnetisation, with transmissive, "
                "square, circular, and reactor-vessel boundary variants.";
            print_justified_paragraph(desc, width);
            std::cout << "\n";

            std::cout << "DETAILS:\n";
            std::string details =
                "A high-pressure region in the domain centre undergoes rapid expansion, "
                "creating a strong outgoing shock wave. The strongly magnetised fluid in "
                "the high-pressure region is expelled as a shock front. This test models "
                "scenarios like supernova explosions or intense stellar flares. Unlike the "
                "rotor, no exact solution exists for the blast wave, so each variant instead "
                "confirms that reflection proceeds as expected under a considerably stronger "
                "shock and that the boundary treatment remains robust rather than generating "
                "spurious oscillations or a failed update as the shock passes close to the "
                "wall. The square and circular variants isolate wall curvature the same way "
                "as the rotor. The square again shows visible corner-induced asymmetry "
                "absent from the circular case, and the blast wave is the more demanding of "
                "the two boundary tests. Its larger pressure jump sharpens gradients once "
                "the shock reflects, particularly where curvature is high, stressing the "
                "near-wall reconstruction far more than the rotor's smoother data. The "
                "reactor variant then applies the same initial data to a realistic ITER, "
                "MAST, or ST40 vessel cross-section, prompting for a vessel choice when run. "
                "The dissertation demonstrates this on the much larger ITER vessel, whose "
                "scale places the initial disturbance well inside the wall rather than close "
                "to it, though this solver allows any vessel to be selected.";
            print_justified_paragraph(details, width);
            std::cout << "\n";

            std::cout << "OUTPUTS & VISUALISATION:\n";
            std::cout << "  Transmissive: Two snapshots at t=0.05 and t=0.1\n";
            std::cout << "  Square: Six snapshots at t=0.0,0.05,0.1,0.2,0.3,0.35\n";
            std::cout << "  Circular: Six snapshots at t=0.0,0.05,0.1,0.2,0.3,0.35\n";
            std::cout << "  Reactor: Six snapshots at t=0.0,0.05,0.1,0.2,0.3,0.35\n";
            std::cout << "  All variants output a 2D density field (ρ) contour plot illustrating shock\n";
            std::cout << "  expansion and wave patterns.\n\n";

            std::cout << "HYPERBOLIC DIVERGENCE CLEANING (GLM):\n";
            std::string glm =
                "CRITICAL: Enable GLM. Extreme pressure ratios and the associated large "
                "magnetic field gradients near the blast centre generate significant "
                "discretisation errors that manifest as monopole pollution. GLM is essential "
                "here to prevent numerical instability, and doubly so for the circular and "
                "reactor-vessel variants where curved and internal boundaries provide "
                "further sites for monopole accumulation.";
            print_justified_paragraph(glm, width);
            std::cout << "\n";

            display_sdf_geometry_explanation(width);
            std::cout << "REFERENCE: Dumbser et al. (2008)\n\n";
            continue;
        }

        if (choice == 1) {
            std::cout << "\nRunning transmissive blast wave test...\n\n";
            run_blast2d();
        } else if (choice == 2) {
            std::cout << "\nRunning square (rectangular box) reflective blast wave test...\n\n";
            run_blast2d_ref();
        } else if (choice == 3) {
            std::cout << "\nRunning circular reflective blast wave test...\n\n";
            run_blast2d_circ();
        } else {
            std::cout << "\nRunning reactor-vessel blast wave test...\n\n";
            run_blast_reactor();
        }
        return;
    }
}

// ============================================================================
// 1D Riemann Problems
// ============================================================================

void run_1d_menu(int width = 90)
{
    const std::vector<TestCase> riemann_problems = {
        {"RP1: Brio-Wu Shock Wave",
        "Transverse magnetic field discontinuity (By = +1 vs By = -1) (Brio and Wu, 1988)",
        "The transverse magnetic field discontinuity between the left and right states drives "
        "the formation of a compound wave structure, in which the initial discontinuity "
        "evolves into shocks, rarefaction waves, and contact discontinuities. OUTPUT: One "
        "snapshot at t=0.1 showing six profiles in a 3x2 grid: density (ρ), pressure (p), "
        "velocity x-component (v_x), velocity y-component (v_y), magnetic field y-component "
        "(B_y), and magnetic field z-component (B_z).",
        "RECOMMENDED: Enable GLM. The transverse magnetic field (B_y) reverses sign and "
        "changes magnitude across the shock. GLM helps control numerical divergence errors "
        "that can arise from reconstructing discontinuous fields.",
        "Brio and Wu (1988)",
        run_brio_wu},

        {"RP2: Transverse Trans-Alfvenic Shock",
        "Ryu and Jones' seven-wave shock test with a transverse trans-Alfvenic shock structure",
        "Multiple discontinuities and intermediate states, strongly influenced by the magnetic "
        "field configuration, validate the solver's ability to capture magnetically dominated "
        "shock interactions. OUTPUT: One snapshot at t=0.2 showing six profiles in a 3x2 grid: "
        "density (ρ), pressure (p), v_x, v_y, B_y, and B_z.",
        "OPTIONAL: GLM can help. Trans-Alfvenic shocks naturally have some numerical "
        "divergence due to the complex wave interactions. GLM is beneficial but not "
        "essential for well-designed schemes.",
        "Ryu and Jones (1995)",
        run_shock_trans},

        {"RP3: Rarefaction Wave",
        "Rarefaction-dominated wave structure from large pressure and density differences",
        "Instead of a shock, the initial discontinuity expands into a smooth rarefaction fan. "
        "The density profile decreases continuously through the fan while the transverse "
        "magnetic field varies smoothly, testing the solver's handling of smooth expansions "
        "and negative-velocity flows. OUTPUT: One snapshot at t=0.15 showing six profiles in "
        "a 3x2 grid: density (ρ), pressure (p), v_x, v_y, B_y, and B_z.",
        "OPTIONAL: GLM provides stability. Out-of-plane fields can generate numerical "
        "artefacts in some schemes. GLM provides extra stability insurance but may not be "
        "strictly necessary.",
        "Toro (2009)",
        run_rarefac},

        {"RP4: Reversed Magnetic Field Discontinuity",
        "Transverse magnetic-field reversal across the interface, normal field held constant",
        "The transverse magnetic field changes sign across the interface while the normal "
        "magnetic field remains constant, producing a combination of shocks, contact "
        "discontinuities, and magnetic transitions from the field reversal. OUTPUT: One "
        "snapshot at t=0.16 showing six profiles in a 3x2 grid: density (ρ), pressure (p), "
        "v_x, v_y, B_y, and B_z.",
        "STRONGLY RECOMMENDED: Enable GLM. Anti-parallel fields are prone to numerical "
        "divergence issues. GLM actively cleans up the monopole pollution that naturally "
        "arises from the opposing fields.",
        "Brio and Wu (1988)",
        run_reverse_b}
    };

    run_menu(riemann_problems, "1D Riemann Problems", width);
}

// ============================================================================
// 2D Test Menu
// ============================================================================

void run_2d_menu(int width = 90)
{
    while (true) {
        print_menu_banner("SELECT 2D TEST", width);
        std::cout << "  1. Blast Wave\n";
        std::cout << "  2. Brio-Wu Shock Tube\n";
        std::cout << "  3. Kelvin-Helmholtz Instability\n";
        std::cout << "  4. Orszag-Tang Vortex\n";
        std::cout << "  5. Reverse-B Shock Tube\n";
        std::cout << "  6. Rotor\n";
        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 6, true);
        if (choice == 0) return;

        if (choice == 999) {
            print_banner("2D TESTS OVERVIEW", width);
            std::string text =
                "Each entry corresponds to its own source folder. Blast and rotor each offer "
                "transmissive, square, circular, and reactor-vessel (ITER, MAST, ST40) "
                "boundary variants. Briowu offers rotated and reflective-box variants. "
                "Kelvinhelmholtz, orszagtang, and reverseb each hold a single test. Select an "
                "entry, then '?' within it for the full description, GLM guidance, and "
                "reference for that test family.";
            print_justified_paragraph(text, width);
            std::cout << "\n";
            continue;
        }

        switch (choice) {
            case 1: run_blast_menu(width);   break;
            case 2: run_brio_menu(width);    break;
            case 3: run_kh_menu(width);      break;
            case 4: run_orszag_menu(width);  break;
            case 5: run_reverseb_menu(width); break;
            case 6: run_rotor_menu(width);   break;
        }
        return;
    }
}

// ============================================================================
// 3D Reactor-Vessel Tests
// ============================================================================

void run_3d_menu(int width = 90)
{
    while (true) {
        print_menu_banner("SELECT 3D TEST", width);
        std::cout << "  1. Rotor\n";
        std::cout << "  2. Blast Wave\n";
        std::cout << "  ?. Explain\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 2, true);
        if (choice == 0) return;

        if (choice == 999) {
            print_banner("3D TESTS OVERVIEW", width);
            std::string text =
                "The rotor and blast wave are extended into three dimensions on a "
                "reactor-shaped reflective vessel, ITER, MAST, or ST40, chosen when the "
                "test is run. Since all three vessels are axisymmetric, the three-dimensional "
                "SDF is obtained by revolving the same tabulated 2D poloidal cross-section "
                "about the device axis rather than tabulating a full volume, so each 3D test "
                "reuses the identical vessel data as its 2D counterpart in mhd/2d/rotor and "
                "mhd/2d/blast. Across all three vessels the SDF-based ghost fluid method "
                "reproduces the qualitative behaviour expected of a rigid, reflective wall, "
                "with no visible leakage beyond the tabulated contour even where the "
                "cross-section departs substantially from a circle. Runs are parallelised "
                "across a shared-memory node with OpenMP, and the dissertation's runtime "
                "study shows parallelisation benefits converge by around 10 threads.";
            print_justified_paragraph(text, width);
            std::cout << "\n";
            display_sdf_geometry_explanation(width);
            std::cout << "REFERENCE: Miyoshi and Kusano (2005); Dumbser et al. (2008)\n\n";
            continue;
        }

        if (choice == 1) {
            std::cout << "\nRunning 3D reactor-vessel rotor test...\n\n";
            run_rotor_3d();
        } else {
            std::cout << "\nRunning 3D reactor-vessel blast wave test...\n\n";
            run_blast_3d();
        }
        return;
    }
}

// ============================================================================
// MHD Solver Top-Level Menu
// ============================================================================

void run_mhd_solver_menu(int width = 90)
{
    while (true) {
        print_menu_banner("MHD SOLVER", width);
        std::cout << "  1. 1D Tests\n";
        std::cout << "  2. 2D Tests\n";
        std::cout << "  3. 3D Tests\n";
        std::cout << "  ?. Explain the Solver\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 3, true);

        if (choice == 0) {
            return;
        }

        if (choice == 999) {
            display_solver_explanation(width);
            press_enter_to_continue();
            continue;
        }

        if (choice == 1) {
            run_1d_menu(width);
        } else if (choice == 2) {
            run_2d_menu(width);
        } else {
            run_3d_menu(width);
        }
    }
}

// ============================================================================
// Grad-Shafranov Solver Menu
// ============================================================================

void run_gs_solver_menu(int width = 90)
{
    while (true) {
        print_menu_banner("GS SOLVER", width);
        std::cout << "  1. Validation\n";
        std::cout << "  2. Reactor Vessel Equilibria\n";
        std::cout << "  3. Perturbed Equilibria\n";
        std::cout << "  ?. Explain the GS Solver\n";
        std::cout << "  0. Back\n\n";

        int choice = get_choice_with_help(0, 3, true);
        if (choice == 0) return;

        if (choice == 999) {
            print_banner("GRAD-SHAFRANOV (GS) SOLVER", width);

            std::cout << "DERIVATION:\n";
            print_justified_paragraph(
                "Assuming negligible flow, the ideal MHD momentum equation reduces to the "
                "static force balance", width);
            print_centred_equation("∇p = J × B", width);
            print_justified_paragraph("Writing", width);
            print_centred_equation("B = ∇ × A", width);
            print_justified_paragraph(
                "and imposing axisymmetry, the poloidal flux Ψ(R, Z) and toroidal field "
                "function f(R, Z) allow the field to be written", width);
            print_centred_equation("B = ∇Ψ × ∇φ + f∇φ", width);
            print_justified_paragraph(
                "Substituting into the force balance shows that f = f(Ψ) and p = p(Ψ) are "
                "surface quantities. This yields the Grad-Shafranov equation", width);
            print_centred_equation("Δ*Ψ = -μ₀R² dp/dΨ - f df/dΨ", width);
            print_justified_paragraph(
                "a nonlinear, two-dimensional elliptic PDE in which Ψ appears both as the "
                "dependent variable and, through p(Ψ) and f(Ψ), as an implicit argument of "
                "its own right-hand side.", width);
            std::cout << "\n";

            std::cout << "PLASMA PROFILES & PICARD ITERATION:\n";
            std::string picard =
                "The pressure and toroidal field profiles are specified as double-power "
                "functions of the normalised poloidal flux, with shape exponents chosen per "
                "reactor. Picard iteration is used to resolve the nonlinearity. This "
                "evaluates the nonlinear source term at the current flux estimate, solves "
                "the resulting linear problem for an updated estimate, and repeats until the "
                "solution stops changing appreciably. It is the same approach adopted by the "
                "independent FreeGS solver used for validation. The flux is split into a "
                "plasma-generated component and a coil-generated component, the latter "
                "obtained from a set of poloidal field coil filaments via the Green's "
                "function for a circular filament, with coil currents determined so total "
                "plasma current and peak pressure match target values.";
            print_justified_paragraph(picard, width);
            std::cout << "\n";

            std::cout << "VALIDATION:\n";
            std::string val =
                "The solver is validated against the analytic Solov'ev equilibrium (Solov'ev, "
                "1968) and the Cerfon-Freidberg analytic single-null equilibrium (Cerfon and "
                "Freidberg, 2010), together with a grid convergence study, before being applied "
                "to compute equilibria for ITER, MAST, and ST40 and compared against the "
                "independent FreeGS solver (Dudson, 2019).";
            print_justified_paragraph(val, width);
            std::cout << "\n";

            std::cout << "REACTOR VESSEL EQUILIBRIA:\n";
            std::string reactor =
                "Equilibrium profiles are computed for the ITER, MAST, and ST40 reactor "
                "vessels, the same three SDF wall geometries used by the reactor MHD tests, "
                "giving a physically consistent flux surface distribution for each machine.";
            print_justified_paragraph(reactor, width);
            std::cout << "\n";

            std::cout << "PERTURBED EQUILIBRIA:\n";
            std::string pert =
                "One perturbation class is applied to a converged equilibrium to drive a "
                "time-dependent disruption simulation from physically consistent initial "
                "conditions, handed off to the finite-volume MHD solver. Two further "
                "perturbation classes are reserved as future work in the dissertation.";
            print_justified_paragraph(pert, width);
            std::cout << "\n";

            std::cout << "REFERENCES: Solov'ev (1968); Cerfon and Freidberg (2010);\n";
            std::cout << "            Dudson (2019); Farmakalides et al. (2025)\n\n";
            continue;
        }

        int ret = 0;
        switch (choice) {
            case 1:
                std::cout << "\nRunning GS Validation (Solov'ev & Single-Null)...\n\n";
                ret = run_validation_gs(0, nullptr);
                break;
            case 2:
                std::cout << "\nRunning GS Equilibrium for Reactor Vessels (ITER/MAST/ST40)...\n\n";
                ret = run_equilibrium_gs(0, nullptr);
                break;
            case 3:
                std::cout << "\nRunning GS Perturbed Equilibrium (Disruption Simulation)...\n\n";
                ret = run_perturbed_gs(0, nullptr);
                break;
            default:
                continue;
        }

        if (ret == 0) {
            std::cout << "\nSolver completed successfully.\n";
        } else {
            std::cout << "\nSolver returned with error code " << ret << ".\n";
        }
        press_enter_to_continue();
    }
}

// ============================================================================
// About / Licence / Acknowledgements & References / Read Me
// ============================================================================

void display_about(int width = 90)
{
    print_banner("ABOUT THIS PROGRAM", width);
    std::string text =
        "This solver accompanies the MPhil dissertation 'Magnetohydrodynamic Simulations in "
        "Complex Geometries' by Uzair Abdullah, Department of Physics, University of "
        "Cambridge (ua247@cam.ac.uk), 2026. The MHD Solver menu mirrors the mhd source "
        "tree. It covers the 1D Riemann problems in mhd/1d, the six 2D test families in "
        "mhd/2d, namely blast, briowu, kelvinhelmholtz, orszagtang, reverseb, and rotor, and "
        "the 3D reactor-vessel tests in mhd/3d. The GS Solver menu mirrors the gs source "
        "tree, covering validation, reactor vessel equilibria, and perturbed equilibria. "
        "Select a solver from the main menu to see its available tests and options.";
    print_justified_paragraph(text, width);
    std::cout << "\n";
}

void display_license(int width = 90) {
    print_banner("LICENCE", width);
    std::cout << "\n";
    std::cout << "                    GNU GENERAL PUBLIC LICENSE\n";
    std::cout << "                       Version 3, 29 June 2007\n\n";
    std::cout << " Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>\n";
    std::cout << " Everyone is permitted to copy and distribute verbatim copies\n";
    std::cout << " of this license document, but changing it is not allowed.\n\n";
    std::cout << " you can redistribute it and/or modify\n";
    std::cout << " it under the terms of the GNU General Public License as published by\n";
    std::cout << " the Free Software Foundation, either version 3 of the License, or\n";
    std::cout << " (at your option) any later version.\n\n";
    std::cout << " This program is distributed in the hope that it will be useful,\n";
    std::cout << " but WITHOUT ANY WARRANTY; without even the implied warranty of\n";
    std::cout << " MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n";
    std::cout << " GNU General Public License for more details.\n\n";
    std::cout << " You should have received a copy of the GNU General Public License\n";
    std::cout << " along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\n";
    std::cout << " The full text of the GPLv3 license is included in the file LICENSE\n";
    std::cout << " that accompanies this software.\n\n";
}

void display_acknowledgements_references(int width = 90)
{
    print_banner("ACKNOWLEDGEMENTS & REFERENCES", width);

    std::cout << "PRIMARY SOURCE:\n";
    std::string primary =
        "Abdullah, U. Magnetohydrodynamic Simulations in Complex Geometries. MPhil "
        "dissertation, Department of Physics, University of Cambridge, 2026. Supervised by "
        "Dr Maria Nikodemou. This solver implements the methods and reproduces the test "
        "problems described therein.";
    print_justified_paragraph(primary, width);
    std::cout << "\n";

    std::cout << "ACKNOWLEDGEMENTS:\n";
    std::string ack =
        "This solver builds on decades of published work in computational magnetohydrodynamics "
        "and finite-volume methods for hyperbolic conservation laws, and on the CRATOS-GS "
        "Grad-Shafranov solver developed at the Laboratory for Scientific Computing. Thanks to "
        "the wider open-source scientific computing community, and to OpenMP for the "
        "shared-memory parallelism used throughout the solver. Free-GS, from Dr Ben Dudson, is "
        "also used as an independent reference for Grad-Shafranov validation.";
    print_justified_paragraph(ack, width);
    std::cout << "\n";

    std::cout << "NUMERICAL METHOD REFERENCES:\n";
    std::cout << "  - Brio, M. and Wu, C. C. (1988). An upwind differencing scheme for the\n";
    std::cout << "    equations of ideal magnetohydrodynamics. J. Comput. Phys.\n";
    std::cout << "  - Ryu, D. and Jones, T. W. (1995). Numerical magnetohydrodynamics in\n";
    std::cout << "    astrophysics: algorithm and tests for one-dimensional flow. Astrophys. J.,\n";
    std::cout << "    442:228-258. doi:10.1086/175437\n";
    std::cout << "  - Miyoshi, T. and Kusano, K. (2005). A multi-state HLL approximate Riemann\n";
    std::cout << "    solver for ideal magnetohydrodynamics. J. Comput. Phys.\n";
    std::cout << "  - Jiang, G.-S. and Shu, C.-W. (1996). Efficient implementation of weighted\n";
    std::cout << "    ENO schemes. J. Comput. Phys.\n";
    std::cout << "  - Gottlieb, S., Shu, C.-W. and Tadmor, E. (2001). Strong stability-preserving\n";
    std::cout << "    high-order time discretization methods. SIAM Review.\n";
    std::cout << "  - Dedner, A. et al. (2002). Hyperbolic divergence cleaning for the MHD\n";
    std::cout << "    equations. J. Comput. Phys.\n";
    std::cout << "  - Fedkiw, R. P. et al. (1999). A non-oscillatory Eulerian approach to\n";
    std::cout << "    interfaces in multimaterial flows (the ghost fluid method). J. Comput. Phys.\n";
    std::cout << "  - Dumbser, M. et al. (2008). A unified framework for the construction of\n";
    std::cout << "    one-step finite volume and discontinuous Galerkin schemes. J. Comput. Phys.\n";
    std::cout << "  - Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid\n";
    std::cout << "    Dynamics. Springer.\n\n";

    std::cout << "GRAD-SHAFRANOV / EQUILIBRIUM REFERENCES:\n";
    std::cout << "  - Solov'ev, L. S. (1968). The theory of hydromagnetic stability of toroidal\n";
    std::cout << "    plasma configurations. Soviet Physics JETP, 26:400-407.\n";
    std::cout << "  - Cerfon, A. J. and Freidberg, J. P. (2010). \"One size fits all\" analytic\n";
    std::cout << "    solutions to the Grad-Shafranov equation. Phys. Plasmas, 17(3):032502.\n";
    std::cout << "    doi:10.1063/1.3328818\n";
    std::cout << "  - Dudson, B. (2019). FreeGS: Free Boundary Grad-Shafranov Solver.\n";
    std::cout << "    https://github.com/freegs-plasma/freegs\n";
    std::cout << "  - Farmakalides, A., Nikiforakis, N., Millmore, S. et al. (2025). CRATOS-GS:\n";
    std::cout << "    a free-boundary, hierarchical adaptive mesh refinement Grad-Shafranov\n";
    std::cout << "    solver. AIP Advances, 15.\n";
    std::cout << "  - Freidberg, J. P. (1987). Ideal Magnetohydrodynamics. Cambridge University\n";
    std::cout << "    Press.\n\n";
}

void display_readme(int width = 90)
{
    print_banner("READ ME", width);

    std::cout << "AUTHOR:  Uzair Abdullah, University of Cambridge (ua247@cam.ac.uk)\n";
    std::cout << "DATE:    07 August 2026\n\n";

    std::cout << "LICENSING:\n";
    std::cout << "  The source code is released under the GNU General Public License,\n";
    std::cout << "  version 3 (GPLv3). See LICENSE for the full text.\n";
    std::cout << "  The documentation (README, explanatory text) is released under the\n";
    std::cout << "  Creative Commons Attribution 4.0 International License (CC BY 4.0).\n";
    std::cout << "  See LICENCE-CC-BY for the full text.\n\n";

    std::cout << "BUILDING:\n";
    std::cout << "  g++ -O3 -std=c++17 -Xpreprocessor -fopenmp \\\n";
    std::cout << "      -I$(brew --prefix libomp)/include \\\n";
    std::cout << "      -L$(brew --prefix libomp)/lib \\\n";
    std::cout << "      -I/opt/homebrew/opt/eigen/include/eigen3 \\\n";
    std::cout << "      -Igs -Imhd/1d \\\n";
    std::cout << "      -Imhd/2d/blast -Imhd/2d/briowu -Imhd/2d/kelvinhelmholtz \\\n";
    std::cout << "      -Imhd/2d/orszagtang -Imhd/2d/reverseb -Imhd/2d/rotor \\\n";
    std::cout << "      -Imhd/3d \\\n";
    std::cout << "      -lomp -march=native \\\n";
    std::cout << "      main.cpp $(find gs mhd -name '*.cpp') -o solver\n\n";

    std::cout << "RUNNING INTERACTIVELY:\n";
    std::cout << "  ./solver\n";
    std::cout << "  Opens the top-level MHD/GS solver menu. Navigate with the printed numbers,\n";
    std::cout << "  '?' for an explanation of the current menu, and '0' to go back.\n\n";

    std::cout << "SKIPPING THE TOP-LEVEL MENU:\n";
    std::cout << "  ./solver --solver mhd\n";
    std::cout << "      Jumps straight into the MHD Solver menu (1D / 2D / 3D tests).\n";
    std::cout << "  ./solver --solver gs\n";
    std::cout << "      Jumps straight into the GS Solver menu.\n";
    std::cout << "  In both cases the top-level MHD/GS solver menu is skipped entirely. The\n";
    std::cout << "  program exits once you back out ('0') of the chosen solver's menu.\n\n";

    std::cout << "RUNNING A SPECIFIC TEST DIRECTLY:\n";
    std::cout << "  ./solver <test_id>\n";
    std::cout << "  e.g. ./solver rotor2dreact\n\n";
    std::cout << "  Available test IDs:\n";
    std::cout << "    briowu1d, rarefac1d, reverseb1d, shocktrans1d, briowu2drot, briowu2dref,\n";
    std::cout << "    reverseb2dref, orszag2d, rotor2d, rotor2dref, rotor2dcirc, rotor2dreact,\n";
    std::cout << "    rotor3dreact, blast2d, blast2dref, blast2dcirc, blast2dreact, blast3dreact,\n";
    std::cout << "    kh2d, validationgs, equilibriumgs, perturbedgs\n\n";

    std::cout << "STRUCTURE:\n";
    std::string text =
        "The menu structure mirrors the on-disk source layout. Under the MHD Solver, 1D "
        "Tests maps to mhd/1d, the four Riemann problems. 2D Tests maps to mhd/2d, with one "
        "submenu per subfolder, namely blast, briowu, kelvinhelmholtz, orszagtang, "
        "reverseb, and rotor. 3D Tests maps to mhd/3d, the reactor-vessel rotor and blast "
        "wave. Under the GS Solver, the three options map to the three files in gs, "
        "ValidationGS, EquilibriumGS, and PerturbedGS. Each test file builds on a shared "
        "numerical core made up of the HLLD Riemann solver, WENO3 reconstruction, SSP-RK3 "
        "time stepping, and optional GLM divergence cleaning. See the Solver explanation, "
        "reached from the MHD Solver menu with '?', for the full numerical scheme, and the "
        "SDF construction section within it for how each boundary geometry is built.";
    print_justified_paragraph(text, width);
    std::cout << "\n";
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    
    const int BANNER_WIDTH = 80;

    if (argc > 1) {
        const std::string arg1 = argv[1];

        if (arg1 == "--solver") {
            if (argc < 3) {
                std::cerr << "Usage: ./solver --solver <mhd|gs>\n";
                return 1;
            }

            const std::string which = argv[2];
            if (which == "mhd") {
                run_mhd_solver_menu(BANNER_WIDTH);
            } else if (which == "gs") {
                run_gs_solver_menu(BANNER_WIDTH);
            } else {
                std::cerr << "Unknown --solver option: " << which
                          << " (expected 'mhd' or 'gs')\n";
                return 1;
            }
            return 0;
        }

        const std::string test = arg1;

        if      (test == "briowu1d")      run_brio_wu();
        else if (test == "rarefac1d")     run_rarefac();
        else if (test == "reverseb1d")    run_reverse_b();
        else if (test == "shocktrans1d")  run_shock_trans();
        else if (test == "briowu2drot")   run_brio_rot();
        else if (test == "briowu2dref")   run_brio_ref();
        else if (test == "reverseb2dref") run_reverse_b_ref();
        else if (test == "orszag2d")      run_orszag();
        else if (test == "rotor2d")       run_rotor();
        else if (test == "rotor2dref")    run_rotor_ref();
        else if (test == "rotor2dcirc")   run_rotor_circ();
        else if (test == "rotor2dreact")  run_rotor_reactor();
        else if (test == "rotor3dreact")  run_rotor_3d();
        else if (test == "blast2d")       run_blast2d();
        else if (test == "blast2dref")    run_blast2d_ref();
        else if (test == "blast2dcirc")   run_blast2d_circ();
        else if (test == "blast2dreact")  run_blast_reactor();
        else if (test == "blast3dreact")  run_blast_3d();
        else if (test == "kh2d")          run_kelvin_helmholtz2d();
        else if (test == "validationgs")  run_validation_gs(0, nullptr);
        else if (test == "equilibriumgs") run_equilibrium_gs(0, nullptr);
        else if (test == "perturbedgs")   run_perturbed_gs(0, nullptr);
        else {
            std::cerr << "Unknown test: " << test << '\n';
            return 1;
        }

        return 0;
    }

        while (true) {
        print_menu_banner("Ideal Magnetohydrodynamics & Grad-Shafranov Solver", BANNER_WIDTH);
        std::cout << "\n";
        std::cout << "  Author:      Uzair Abdullah\n";
        std::cout << "  Affiliation: Centre for Scientific Computing\n";
        std::cout << "               Department of Physics\n";
        std::cout << "               University of Cambridge\n";
        std::cout << "  Email:       ua247@cam.ac.uk\n";
        std::cout << "  Date:        07 August 2026\n";
        std::cout << "  Supervisor:  Dr Maria Nikodemou\n";
        std::cout << "\n";
        std::cout << "  This solver accompanies the Master of Philosophy dissertation\n";
        std::cout << "  \"Magnetohydrodynamic Simulations in Complex Geometries\".\n";
        std::cout << "\n";
        std::cout << "  The source code is released under the GNU General Public License, version 3\n";
        std::cout << "  (GPLv3). See LICENSE for the full text.\n";
        std::cout << "\n\n";
        std::cout << std::string(80, '=') << '\n';
        std::cout << std::string(80, '=') << '\n';

        std::cout << "\n\n  1. MHD Solver\n";
        std::cout << "  2. Grad-Shafranov Solver\n";
        std::cout << "  ?. About this Program\n";
        std::cout << "  A. Acknowledgements & References\n";
        std::cout << "  L. Licence\n";
        std::cout << "  R. Read Me\n";
        std::cout << "  0. Exit\n\n";

        std::string input;
        std::cout << "Enter choice (0, 1, 2, ?, L, A, R): ";
        std::getline(std::cin, input);

        input.erase(0, input.find_first_not_of(" \t\r\n"));
        input.erase(input.find_last_not_of(" \t\r\n") + 1);

        std::transform(input.begin(), input.end(), input.begin(), ::toupper);

        if (input == "0") {
            std::cout << "\nExiting. Thank you for using the solver.\n";
            break;
        }
        else if (input == "?") {
            display_about(BANNER_WIDTH);
            press_enter_to_continue();
            continue;
        }
        else if (input == "1") {
            run_mhd_solver_menu(BANNER_WIDTH);
        }
        else if (input == "2") {
            run_gs_solver_menu(BANNER_WIDTH);
        }
        else if (input == "L") {
            display_license(BANNER_WIDTH);
            press_enter_to_continue();
        }
        else if (input == "A") {
            display_acknowledgements_references(BANNER_WIDTH);
            press_enter_to_continue();
        }
        else if (input == "R") {
            display_readme(BANNER_WIDTH);
            press_enter_to_continue();
        }
        else {
            std::cout << "Invalid option. Please enter 0, 1, 2, ?, L, A, or R.\n";
        }
    }

    return 0;
}