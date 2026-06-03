# Algorithmic Proxy Learning Under Privacy Caps

This Julia project simulates a two-consumer proxy-learning model for a thesis
chapter on algorithmic experimentation and privacy caps.

The platform can collect direct precision about a high-value target consumer
`h = 1` or proxy precision about a correlated low-value consumer `l = 2`.
The written rule in the main treatment caps only target own-data KL:

```julia
K_own_h(beta_h) <= kappa_h
```

It does not block proxy data, even when proxy data increase total inference
about the target. The robust counterfactual blocks any action that would make
total target KL exceed `kappa_h`.

## Run

From this folder:

```julia
include("run_all.jl")
```

All files are written under:

```julia
raw"C:\Users\mvsma\Videos\Thesis\Algorithmic Experimentation"
```

The simulation logic uses Julia standard libraries and writes CSV tables without
CSV.jl or DataFrames.jl. Figures use Plots.jl with the GR backend so the output
has publication-style fonts, margins, legends, and colorbars. No external data
or GPU is required.

## Files

```text
Project.toml
src/
    posterior.jl
    environment.jl
    qlearning.jl
    value_iteration.jl
    experiments.jl
    plotting.jl
run_all.jl
README.md
```

## Outputs

```text
tables/
    main_experiment_summary.csv
    placebo_low_correlation.csv
    placebo_low_value.csv
    robust_regulation_summary.csv
    grid_results.csv
    q_vs_value_iteration.csv
    figure_data_beta_trajectory.csv
    figure_data_kl_trajectory.csv
    figure_data_q_values_cap_binds.csv
    figure_data_violation_magnitude.csv

figures/
    learning_curve_main.png
    learning_curve_main.pdf
    proxy_share_over_time_main.png
    proxy_share_over_time_main.pdf
    beta_trajectory_main.png
    beta_trajectory_main.pdf
    kl_trajectory_main.png
    kl_trajectory_main.pdf
    q_values_when_cap_binds.png
    q_values_when_cap_binds.pdf
    heatmap_proxy_share.png
    heatmap_proxy_share.pdf
    heatmap_total_KL_violation_magnitude.png
    heatmap_total_KL_violation_magnitude.pdf
    comparison_direct_vs_robust.png
    comparison_direct_vs_robust.pdf

logs/
    run_log.txt
    figure_audit.txt
```

`logs/figure_audit.txt` records PNG and PDF existence, file sizes, nonempty-data
status, Q-value cap-state validity, and whether manual axis-label annotations
were used. The plotting code uses Plots.jl `xlabel`, `ylabel`, `title`, `label`,
and `colorbar_title` attributes rather than manual label placement.

## Interpretation

The main test is whether Q-learning collects direct target data until the
written own-data cap binds, then continues to learn about the target through the
proxy when correlation and target value are high.

The robust-total-KL counterfactual asks whether the proxy channel disappears
when the regulator caps total inference about the target instead of only direct
own-data inference.
