# Algorithmic proxy-learning thesis simulation.
#
# Run from the project root with:
#     include("run_all.jl")
#
# The pipeline writes every table, figure, and log under OUTPUT_DIR.

const PROJECT_ROOT = @__DIR__
const OUTPUT_DIR = raw"C:\Users\mvsma\Videos\Thesis\Algorithmic Experimentation"

import Pkg
Pkg.activate(PROJECT_ROOT)

include(joinpath(PROJECT_ROOT, "src", "posterior.jl"))
include(joinpath(PROJECT_ROOT, "src", "environment.jl"))
include(joinpath(PROJECT_ROOT, "src", "qlearning.jl"))
include(joinpath(PROJECT_ROOT, "src", "value_iteration.jl"))
include(joinpath(PROJECT_ROOT, "src", "plotting.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiments.jl"))

result = run_pipeline(OUTPUT_DIR)
