# Code for: Eco-evolutionary dynamics of pathogen epidemic timing in a seasonal environment

This repository contains the simulation and analysis code needed to reproduce the numerical results and figures in the manuscript.

## Requirements

- [Julia](https://julialang.org/) (≥ 1.9 recommended)
- The following Julia packages (install via `using Pkg; Pkg.add([...])` or `pkg> add ...`):

```
DifferentialEquations
DiffEqCallbacks
QuadGK
SpecialFunctions
Plots
DelimitedFiles
Interpolations
CSV
DataFrames
Statistics
Printf
```

## File overview

| File | Description | Output files |
|------|-------------|--------------|
| `bifurcation_diagram.jl` | Scans for the critical seasonality threshold δ_c across transmission rates β₀ | `output/critical_delta_vs_beta.csv` |
| `figs_bifurcation.jl` | Computes selection gradients, evolutionarily stable strategies (ESS), invasion fitness, and pairwise invasibility plots (PIP) data | `output/selection_gradient_0.01.tsv`, `output/selection_gradient_0.1.tsv`, `output/evo_points.tsv`, `output/ES_pattern_long.tsv`, `output/rho_list_figure_*.tsv` |
| `oligomorph_extend_fig_fast_detailed_online_check_old.jl` | Runs oligomorphic dynamics simulations across (β₀, δ) parameter space for the baseline and four robustness-check parameter sets | `output/oligo_data/traj_for_figure_M2_fast_large_var_detailed2_online_old_*.csv` |
| `Fig_revised.nb` | Mathematica notebook that reads the above output files and generates all manuscript figures | `output/figure/Fig*.pdf` |

## How to run

All output files are saved to an `output/` subdirectory created automatically in the same folder as the scripts.

```bash
# Step 1: critical delta curve (fast, ~minutes)
julia bifurcation_diagram.jl

# Step 2: selection gradients, ESS, and invasion fitness (moderate, ~tens of minutes)
julia figs_bifurcation.jl

# Step 3: oligomorphic simulations (slow — uses multi-threading; allow several hours)
julia --threads auto oligomorph_extend_fig_fast_detailed_online_check_old.jl
```

Steps 1–3 are independent and can be run in any order or in parallel.

### Step 4: generate figures (Mathematica)

Open `Fig_revised.nb` in [Wolfram Mathematica](https://www.wolfram.com/mathematica/) (≥ 14.1 recommended) and run all cells in order.  
The first cell runs `SetDirectory[NotebookDirectory[]]` and creates the `output/figure/` directory automatically.  
Figures are exported as PDF to `output/figure/`.

### Changing the output directory

By default, results are written to `./output/` relative to the script location.  
To save elsewhere, set the `OUTPUT_DIR` environment variable before running:

```bash
# macOS / Linux
export OUTPUT_DIR=/path/to/your/output
julia figs_bifurcation.jl

# or inline (single run)
OUTPUT_DIR=/path/to/your/output julia figs_bifurcation.jl
```

## Parameter conventions

| Symbol | Meaning | Default |
|--------|---------|---------|
| β₀ | baseline transmission rate | varies |
| δ | amplitude of seasonal forcing | varies |
| d | natural death/birth rate | 0.1 |
| γ | recovery rate | 1.0 |
| c | waning immunity rate | 0.5 |
| κ (= a) | von Mises concentration (sharpness of seasonal peak) | 1.0 |

## Robustness checks (oligomorphic simulations)

`oligomorph_extend_fig_fast_detailed_online_check_old.jl` runs five parameter sets in sequence:

| Run | Parameter changed | Value |
|-----|-------------------|-------|
| baseline | — | d = 0.1, c = 0.5 |
| c = 0.1 | waning immunity | c = 0.1 |
| c = 1.0 | waning immunity | c = 1.0 |
| d = 0.05 | death rate | d = 0.05 |
| d = 0.2 | death rate | d = 0.2 |

## License

[CC BY 4.0]
