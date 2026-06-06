# Robustness-only runner for the proxy-learning thesis simulation.
#
# Recommended command from the project root:
#     julia --project=. run_robustness_only.jl --fast
#
# Paths are computed from this file location, so the script can be launched from
# the project root, from src, or from another working directory.

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

function parse_robustness_runner_args(args)
    mode = :default
    for arg in args
        if arg == "--fast"
            mode = :fast
        elseif arg == "--paper"
            mode = :paper
        else
            error("Unknown argument: $arg")
        end
    end
    return mode
end

function robustness_runner_episode_settings(mode::Symbol)
    if mode == :fast
        return (episodes = 2_000, grid_episodes = 500, vi_episodes = 2_000)
    elseif mode == :paper
        return (episodes = 50_000, grid_episodes = 5_000, vi_episodes = 50_000)
    end
    return (episodes = 10_000, grid_episodes = 3_000, vi_episodes = 10_000)
end

mode = parse_robustness_runner_args(ARGS)
settings = robustness_runner_episode_settings(mode)
q_audit_path = joinpath(OUTPUT_DIR, "logs", "figure_audit.txt")

result = run_robustness_pipeline(OUTPUT_DIR;
    episodes=settings.episodes,
    grid_episodes=settings.grid_episodes,
    vi_episodes=settings.vi_episodes,
    q_audit_path=isfile(q_audit_path) ? q_audit_path : nothing,
)
