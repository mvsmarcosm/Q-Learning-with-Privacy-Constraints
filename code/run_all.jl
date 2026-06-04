# Algorithmic proxy-learning thesis simulation.
#
# Recommended command from the project root:
#     julia --project=. run_all.jl
#
# Fast quality-check mode:
#     julia --project=. run_all.jl --fast
#
# The pipeline writes tables, figures, and logs under OUTPUT_DIR. Paths are
# computed from this file location, so the script can be launched from anywhere.

const PROJECT_ROOT = abspath(@__DIR__)
const OUTPUT_DIR = raw"C:\Users\mvsma\Videos\Thesis\Algorithmic Experimentation"

import Pkg
Pkg.activate(PROJECT_ROOT)
Pkg.instantiate()

include(joinpath(PROJECT_ROOT, "src", "posterior.jl"))
include(joinpath(PROJECT_ROOT, "src", "environment.jl"))
include(joinpath(PROJECT_ROOT, "src", "qlearning.jl"))
include(joinpath(PROJECT_ROOT, "src", "value_iteration.jl"))
include(joinpath(PROJECT_ROOT, "src", "plotting.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiments.jl"))
include(joinpath(PROJECT_ROOT, "src", "robustness.jl"))

function parse_runner_args(args)
    mode = :default
    run_baseline = true
    run_robustness = true
    for arg in args
        if arg == "--fast"
            mode = :fast
        elseif arg == "--paper"
            mode = :paper
        elseif arg == "--skip-baseline"
            run_baseline = false
        elseif arg == "--skip-robustness"
            run_robustness = false
        elseif arg == "--robustness-only"
            run_baseline = false
            run_robustness = true
        else
            error("Unknown argument: $arg")
        end
    end
    return mode, run_baseline, run_robustness
end

function runner_episode_settings(mode::Symbol)
    if mode == :fast
        return (
            baseline_main = 2_000,
            baseline_grid = 500,
            robustness = 2_000,
            robustness_grid = 500,
            robustness_vi = 2_000,
        )
    elseif mode == :paper
        return (
            baseline_main = 50_000,
            baseline_grid = 5_000,
            robustness = 50_000,
            robustness_grid = 5_000,
            robustness_vi = 50_000,
        )
    end
    return (
        baseline_main = 50_000,
        baseline_grid = 5_000,
        robustness = 10_000,
        robustness_grid = 3_000,
        robustness_vi = 10_000,
    )
end

mode, run_baseline, run_robustness = parse_runner_args(ARGS)
settings = runner_episode_settings(mode)

baseline_result = nothing
robustness_result = nothing
q_audit_path = joinpath(OUTPUT_DIR, "logs", "figure_audit.txt")

if run_baseline
    baseline_result = run_pipeline(OUTPUT_DIR;
        main_episodes=settings.baseline_main,
        grid_episodes=settings.baseline_grid,
    )
end

if run_robustness
    robustness_result = run_robustness_pipeline(OUTPUT_DIR;
        episodes=settings.robustness,
        grid_episodes=settings.robustness_grid,
        vi_episodes=settings.robustness_vi,
        q_audit_path=isfile(q_audit_path) ? q_audit_path : nothing,
    )
end

result = Dict(:baseline => baseline_result, :robustness => robustness_result)
