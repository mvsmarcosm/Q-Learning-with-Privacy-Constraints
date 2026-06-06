using Statistics
using Printf

const DEFAULT_OUTPUT_DIR = raw"C:\Users\mvsma\Videos\Thesis\Algorithmic Experimentation"

"""
    ensure_output_dirs(output_dir)

Create project output folders for tables, figures, and logs.
"""
function ensure_output_dirs(output_dir::AbstractString)::Dict{Symbol,String}
    mkpath(output_dir)
    tables = joinpath(output_dir, "tables")
    figures = joinpath(output_dir, "figures")
    logs = joinpath(output_dir, "logs")
    mkpath(tables)
    mkpath(figures)
    mkpath(logs)
    return Dict(:root => output_dir, :tables => tables, :figures => figures, :logs => logs)
end

"""
    csv_escape(x)

Convert a value to a safe CSV cell.
"""
function csv_escape(x)::String
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

"""
    write_table(path, headers, rows)

Write rows to CSV without external packages.
"""
function write_table(path::AbstractString, headers, rows)
    open(path, "w") do io
        println(io, join(csv_escape.(headers), ","))
        for row in rows
            println(io, join(csv_escape.(row), ","))
        end
    end
end

"""
    moving_average(x, window)

Return a trailing moving average series.
"""
function moving_average(x::Vector{Float64}, window::Int)::Vector{Float64}
    out = similar(x)
    acc = 0.0
    for i in eachindex(x)
        acc += x[i]
        if i > window
            acc -= x[i - window]
        end
        out[i] = acc / min(i, window)
    end
    return out
end

"""
    mean_last(x, n)

Return the mean of the last `n` observations.
"""
function mean_last(x::Vector{Float64}, n::Int)::Float64
    isempty(x) && return 0.0
    lo = max(1, length(x) - n + 1)
    return mean(@view x[lo:end])
end

"""
    run_named_experiment(name, params, lp; do_vi=true, verbose=false)

Train Q-learning, evaluate its greedy policy, and optionally compare it with
exact finite-horizon value iteration.
"""
function run_named_experiment(name::String, params::ModelParams, lp::LearningParams; do_vi::Bool=true, verbose::Bool=false)::Dict{Symbol,Any}
    println("Running $name: rho=$(params.rho), r_h=$(params.r_h), regulation=$(params.regulation)")
    env = make_env(params)
    train = train_q_learning(env, lp; verbose=verbose)
    q_eval = evaluate_q_policy(env, train[:Q])
    vi = nothing
    vi_eval = nothing
    comparison = Dict{Symbol,Float64}()
    if do_vi
        vi = solve_value_iteration(env)
        vi_eval = evaluate_vi_policy(env, vi)
        comparison = compare_q_to_value_iteration(env, train[:Q], vi)
    end
    cap_state = best_learned_cap_binding_state(env, train[:Q])
    q_cap = q_values_at_state(train[:Q], env, cap_state)
    return Dict{Symbol,Any}(
        :name => name,
        :params => params,
        :env => env,
        :train => train,
        :q_eval => q_eval,
        :vi => vi,
        :vi_eval => vi_eval,
        :comparison => comparison,
        :cap_state => cap_state,
        :q_values_cap => q_cap,
    )
end

"""
    summary_row(result)

Build one summary-table row from an experiment result.
"""
function summary_row(result::Dict{Symbol,Any})
    p = result[:params]
    m = result[:q_eval][:final_metrics]
    return [
        result[:name], p.regulation, p.rho, p.r_h, p.r_l, p.kappa_h,
        m[:beta_h], m[:beta_l], m[:K_own_h], m[:K_h], m[:L_h],
        m[:proxy_share_after_cap], m[:total_KL_violation], m[:total_reward],
        mean_last(result[:train][:episode_rewards], min(1000, length(result[:train][:episode_rewards]))),
    ]
end

"""
    summary_headers()

Column names for experiment summary tables.
"""
summary_headers() = [
    "experiment", "regulation", "rho", "r_h", "r_l", "kappa_h",
    "final_beta_h", "final_beta_l", "K_own_h", "total_K_h", "L_h",
    "proxy_share_after_cap", "total_KL_violation", "evaluation_reward",
    "avg_reward_last_1000",
]

"""
    q_vi_row(result)

Build one value-iteration validation row.
"""
function q_vi_row(result::Dict{Symbol,Any})
    c = result[:comparison]
    isempty(c) && return [result[:name], "", "", "", ""]
    return [
        result[:name],
        get(c, :all_state_match_rate, NaN),
        get(c, :path_match_rate, NaN),
        get(c, :avg_exact_regret, NaN),
        get(c, :max_exact_regret, NaN),
    ]
end

"""
    run_grid_experiment(base_params, grid_lp)

Train Q-learning over `(rho, r_h)` pairs under direct own-data regulation.
"""
function run_grid_experiment(base_params::ModelParams, grid_lp::LearningParams)::Vector{Dict{Symbol,Any}}
    rho_values = vcat(collect(0.0:0.1:0.9), [0.95])
    r_h_values = [1.0, 2.0, 5.0, 10.0, 20.0]
    rows = Dict{Symbol,Any}[]
    total = length(rho_values) * length(r_h_values)
    k = 0
    for r_h in r_h_values
        for rho in rho_values
            k += 1
            println(@sprintf("  grid %d/%d: rho=%.2f r_h=%.1f", k, total, rho, r_h))
            params = ModelParams(;
                rho=rho, r_h=r_h, r_l=base_params.r_l, theta=base_params.theta,
                cost_h=base_params.cost_h, cost_l=base_params.cost_l,
                cost_exploit=base_params.cost_exploit, gamma=base_params.gamma,
                kappa_h=base_params.kappa_h, terminal_multiplier=base_params.terminal_multiplier,
                dbeta=base_params.dbeta, beta_max=base_params.beta_max, T=base_params.T,
                regulation=:direct_cap,
            )
            env = make_env(params)
            train = train_q_learning(env, grid_lp; verbose=false)
            eval = evaluate_q_policy(env, train[:Q])
            m = eval[:final_metrics]
            push!(rows, Dict{Symbol,Any}(
                :rho => rho,
                :r_h => r_h,
                :final_beta_h => m[:beta_h],
                :final_beta_l => m[:beta_l],
                :total_K_h => m[:K_h],
                :K_own_h => m[:K_own_h],
                :L_h => m[:L_h],
                :proxy_share_after_cap => m[:proxy_share_after_cap],
                :total_KL_violation => m[:total_KL_violation],
                :avg_reward_last_1000 => mean_last(train[:episode_rewards], min(1000, length(train[:episode_rewards]))),
                :episodes => grid_lp.episodes,
            ))
        end
    end
    return rows
end

"""
    write_grid_results(path, rows)

Write grid experiment rows to CSV.
"""
function write_grid_results(path::AbstractString, rows::Vector{Dict{Symbol,Any}})
    headers = [
        "rho", "r_h", "final_beta_h", "final_beta_l", "total_K_h", "K_own_h",
        "L_h", "proxy_share_after_cap", "total_KL_violation",
        "avg_reward_last_1000", "episodes",
    ]
    table = [[r[Symbol(h)] for h in headers] for r in rows]
    write_table(path, headers, table)
end

"""
    grid_matrix(rows, key)

Convert grid rows to a matrix indexed by `r_h` rows and `rho` columns.
"""
function grid_matrix(rows::Vector{Dict{Symbol,Any}}, key::Symbol)
    rho_values = sort(unique(Float64(r[:rho]) for r in rows))
    rh_values = sort(unique(Float64(r[:r_h]) for r in rows))
    mat = zeros(Float64, length(rh_values), length(rho_values))
    for r in rows
        iy = findfirst(==(Float64(r[:r_h])), rh_values)
        ix = findfirst(==(Float64(r[:rho])), rho_values)
        mat[iy, ix] = Float64(r[key])
    end
    return rho_values, rh_values, mat
end

"""
    assert_plot_data(name, values)

Throw an error if a figure would be generated from empty or non-finite data.
"""
function assert_plot_data(name::String, values)
    vals = Float64[]
    for v in vec(collect(values))
        isfinite(Float64(v)) && push!(vals, Float64(v))
    end
    isempty(vals) && error("Empty/non-finite plotting data for $name")
    return true
end

"""
    figure_file_status(path, empty_data, q_cap_valid; manual_axis_annotations=false)

Return one audit row for a figure path.
"""
function figure_file_status(path::AbstractString, empty_data::Bool, q_cap_valid; manual_axis_annotations::Bool=false)
    png_exists = isfile(path)
    png_kb = png_exists ? round(stat(path).size / 1024; digits=1) : 0.0
    pdf_path = splitext(path)[1] * ".pdf"
    pdf_exists = isfile(pdf_path)
    pdf_kb = pdf_exists ? round(stat(pdf_path).size / 1024; digits=1) : 0.0
    return [path, png_exists, png_kb, pdf_path, pdf_exists, pdf_kb, empty_data, q_cap_valid, manual_axis_annotations]
end

"""
    write_figure_audit(outdirs, audit_rows)

Write a figure audit file with existence, file size, empty-data status, and
Q-value cap-state validity.
"""
function write_figure_audit(outdirs, audit_rows)
    path = joinpath(outdirs[:logs], "figure_audit.txt")
    open(path, "w") do io
        println(io, "png_path,png_exists,png_size_kb,pdf_path,pdf_exists,pdf_size_kb,empty_data,q_values_cap_state_valid,manual_axis_label_annotations")
        for row in audit_rows
            println(io, join(row, ","))
        end
    end
    return path
end

"""
    build_all_figures(outdirs, results, grid_rows)

Generate every requested PNG figure.
"""
function build_all_figures(outdirs, results::Dict{String,Dict{Symbol,Any}}, grid_rows)
    figs = outdirs[:figures]
    tables = outdirs[:tables]
    main = results["main_direct_cap"]
    train = main[:train]
    q_eval = main[:q_eval]
    audit_rows = Any[]

    beta_rows = Any[]
    for t in 0:(length(q_eval[:beta_h_path]) - 1)
        push!(beta_rows, [t, q_eval[:beta_h_path][t + 1], q_eval[:beta_l_path][t + 1]])
    end
    write_table(joinpath(tables, "figure_data_beta_trajectory.csv"),
        ["time", "beta_h_direct", "beta_l_proxy"], beta_rows)

    kl_rows = Any[]
    for t in 0:(length(q_eval[:K_own_path]) - 1)
        push!(kl_rows, [t, q_eval[:K_own_path][t + 1], q_eval[:K_total_path][t + 1], main[:params].kappa_h])
    end
    write_table(joinpath(tables, "figure_data_kl_trajectory.csv"),
        ["time", "K_h_own", "K_h_total", "kappa_h"], kl_rows)

    cap_state = main[:cap_state]
    ih, il, tt = decode_state(main[:env], cap_state)
    cap_beta_h, cap_beta_l = beta_values(main[:env], ih, il)
    qvals_raw = main[:q_values_cap]
    q_cap_valid = direct_cap_binds(main[:env], ih) &&
        is_allowed_action(main[:env], cap_state, ACTION_PROXY_L) &&
        tt < main[:params].T
    q_cap_valid || error("Could not find a valid cap-binding state for Q-value plot.")
    q_labels = [
        "Exploit",
        is_allowed_action(main[:env], cap_state, ACTION_DIRECT_H) ? "Direct h" : "Direct h\n(blocked)",
        "Proxy ℓ",
    ]
    q_plot_values = [isfinite(Float64(v)) ? Float64(v) : 0.0 for v in qvals_raw]
    q_rows = Any[]
    for i in 1:3
        action = i - 1
        push!(q_rows, [
            action_name(action), q_labels[i], q_plot_values[i],
            is_allowed_action(main[:env], cap_state, action), cap_beta_h, cap_beta_l, tt,
        ])
    end
    write_table(joinpath(tables, "figure_data_q_values_cap_binds.csv"),
        ["action", "label", "q_value_plotted", "feasible", "beta_h", "beta_l", "time"], q_rows)

    violation_rows = Any[]
    for r in grid_rows
        mag = max(Float64(r[:total_K_h]) - main[:params].kappa_h, 0.0)
        push!(violation_rows, [r[:rho], r[:r_h], r[:total_K_h], main[:params].kappa_h, mag])
    end
    write_table(joinpath(tables, "figure_data_violation_magnitude.csv"),
        ["rho", "r_h", "K_h_total", "kappa_h", "violation_magnitude"], violation_rows)

    f = joinpath(figs, "learning_curve_main.png")
    assert_plot_data("learning_curve_main", train[:episode_rewards])
    plot_line_series(
        f,
        [("Moving-average reward", moving_average(train[:episode_rewards], 500), BLUE)];
        title="Q-learning convergence", xlabel="Episode", ylabel="Moving-average reward",
        legend=false, ylims=(28.0, 36.5),
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    f = joinpath(figs, "proxy_share_over_time_main.png")
    assert_plot_data("proxy_share_over_time_main", train[:proxy_after_cap])
    plot_line_series(
        f,
        [("Proxy share after cap binds", moving_average(train[:proxy_after_cap], 500), GREEN)];
        title="Proxy use after target cap binds", xlabel="Episode",
        ylabel="Proxy action share after cap binds", legend=false, ylims=(0.0, 1.0),
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    f = joinpath(figs, "beta_trajectory_main.png")
    assert_plot_data("beta_trajectory_main beta_h", q_eval[:beta_h_path])
    assert_plot_data("beta_trajectory_main beta_l", q_eval[:beta_l_path])
    plot_line_series(
        f,
        [("βₕ direct", q_eval[:beta_h_path], BLUE), ("βₗ proxy", q_eval[:beta_l_path], GREEN)];
        title="Precision trajectory under direct own-data cap", xlabel="Episode/time",
        ylabel="Precision", legend=:outerright, yzero=true,
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    f = joinpath(figs, "kl_trajectory_main.png")
    assert_plot_data("kl_trajectory_main own", q_eval[:K_own_path])
    assert_plot_data("kl_trajectory_main total", q_eval[:K_total_path])
    plot_line_series(
        f,
        [("Kₕ own", q_eval[:K_own_path], BLUE), ("Kₕ total", q_eval[:K_total_path], RED)];
        title="Own-data KL versus total KL", xlabel="Episode/time",
        ylabel="KL inference about target", capline=main[:params].kappa_h,
        caplabel="κₕ cap", legend=:outerright, yzero=true,
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    f = joinpath(figs, "q_values_when_cap_binds.png")
    assert_plot_data("q_values_when_cap_binds", q_plot_values)
    plot_bar_chart(
        f,
        q_labels, q_plot_values;
        title="Learned Q-values when target cap binds",
        xlabel="Action",
        ylabel="Q-value", colors=[BLUE, RED, GREEN], yzero=true,
    )
    push!(audit_rows, figure_file_status(f, false, q_cap_valid))

    rho_vals, rh_vals, proxy_mat = grid_matrix(grid_rows, :proxy_share_after_cap)
    f = joinpath(figs, "heatmap_proxy_share.png")
    assert_plot_data("heatmap_proxy_share", proxy_mat)
    plot_heatmap(
        f,
        [@sprintf("%.2f", x) for x in rho_vals], [@sprintf("%.0f", y) for y in rh_vals], proxy_mat;
        title="Proxy use after target cap binds", xlabel="Correlation ρ",
        ylabel="Target value rₕ", colorbar_title="Proxy share",
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    mag_rows = Dict{Symbol,Any}[]
    for r in grid_rows
        rr = copy(r)
        rr[:violation_magnitude] = max(Float64(r[:total_K_h]) - main[:params].kappa_h, 0.0)
        push!(mag_rows, rr)
    end
    _, _, mag_mat = grid_matrix(mag_rows, :violation_magnitude)
    f = joinpath(figs, "heatmap_total_KL_violation_magnitude.png")
    assert_plot_data("heatmap_total_KL_violation_magnitude", mag_mat)
    plot_heatmap(
        f,
        [@sprintf("%.2f", x) for x in rho_vals], [@sprintf("%.0f", y) for y in rh_vals], mag_mat;
        title="Magnitude of total KL violation",
        xlabel="Correlation ρ", ylabel="Target value rₕ",
        colorbar_title="max(Kₕ total - κₕ, 0)",
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    direct = results["main_direct_cap"][:q_eval][:final_metrics]
    robust = results["robust_total_kl"][:q_eval][:final_metrics]
    values = [
        direct[:beta_l] direct[:K_h];
        robust[:beta_l] robust[:K_h]
    ]
    f = joinpath(figs, "comparison_direct_vs_robust.png")
    assert_plot_data("comparison_direct_vs_robust", values)
    plot_grouped_bars(
        f,
        ["Direct own-data cap", "Robust total-KL cap"], ["βₗ proxy", "Kₕ total"], values;
        title="Direct cap versus robust total-KL regulation",
        xlabel="Regulation regime", ylabel="Level",
    )
    push!(audit_rows, figure_file_status(f, false, ""))

    audit_path = write_figure_audit(outdirs, audit_rows)
    return audit_path
end

"""
    write_run_log(outdirs, lines)

Write a plain-text log of run settings and interpretation.
"""
function write_run_log(outdirs, lines::Vector{String})
    open(joinpath(outdirs[:logs], "run_log.txt"), "w") do io
        for line in lines
            println(io, line)
        end
    end
end

"""
    run_pipeline(output_dir; main_episodes=50000, grid_episodes=5000)

Execute the full thesis simulation pipeline from scratch.
"""
function run_pipeline(output_dir::AbstractString=DEFAULT_OUTPUT_DIR; main_episodes::Int=50_000, grid_episodes::Int=5_000)
    outdirs = ensure_output_dirs(output_dir)
    run_formula_tests()

    base = ModelParams()
    lp = LearningParams(episodes=main_episodes, seed=20260603)
    grid_lp = LearningParams(episodes=grid_episodes, seed=20260604)

    results = Dict{String,Dict{Symbol,Any}}()
    results["main_direct_cap"] = run_named_experiment("main_direct_cap", base, lp; do_vi=true, verbose=true)
    results["placebo_low_correlation"] = run_named_experiment("placebo_low_correlation", ModelParams(; rho=0.05), lp; do_vi=true, verbose=false)
    results["placebo_low_value"] = run_named_experiment("placebo_low_value", ModelParams(; r_h=1.0), lp; do_vi=true, verbose=false)
    robust_params = ModelParams(; regulation=:total_kl_cap)
    results["robust_total_kl"] = run_named_experiment("robust_total_kl", robust_params, lp; do_vi=true, verbose=false)

    println("Running grid experiment with $grid_episodes Q-learning episodes per cell")
    grid_rows = run_grid_experiment(base, grid_lp)

    tables = outdirs[:tables]
    write_table(joinpath(tables, "main_experiment_summary.csv"), summary_headers(), [summary_row(results["main_direct_cap"])])
    write_table(joinpath(tables, "placebo_low_correlation.csv"), summary_headers(), [summary_row(results["placebo_low_correlation"])])
    write_table(joinpath(tables, "placebo_low_value.csv"), summary_headers(), [summary_row(results["placebo_low_value"])])
    write_table(joinpath(tables, "robust_regulation_summary.csv"), summary_headers(), [
        summary_row(results["main_direct_cap"]),
        summary_row(results["robust_total_kl"]),
    ])
    write_grid_results(joinpath(tables, "grid_results.csv"), grid_rows)
    write_table(joinpath(tables, "q_vs_value_iteration.csv"),
        ["experiment", "all_state_match_rate", "path_match_rate", "avg_exact_regret", "max_exact_regret"],
        [q_vi_row(results[k]) for k in ["main_direct_cap", "placebo_low_correlation", "placebo_low_value", "robust_total_kl"]],
    )

    figure_audit_path = build_all_figures(outdirs, results, grid_rows)

    main_m = results["main_direct_cap"][:q_eval][:final_metrics]
    robust_m = results["robust_total_kl"][:q_eval][:final_metrics]
    placebo_corr_m = results["placebo_low_correlation"][:q_eval][:final_metrics]
    placebo_value_m = results["placebo_low_value"][:q_eval][:final_metrics]
    interpretation = String[
        "Algorithmic proxy-learning run complete.",
        @sprintf("Main direct-cap: beta_h=%.3f beta_l=%.3f K_own_h=%.4f total_K_h=%.4f L_h=%.4f proxy_after_cap=%.3f",
            main_m[:beta_h], main_m[:beta_l], main_m[:K_own_h], main_m[:K_h], main_m[:L_h], main_m[:proxy_share_after_cap]),
        @sprintf("Low-correlation placebo proxy_after_cap=%.3f final beta_l=%.3f", placebo_corr_m[:proxy_share_after_cap], placebo_corr_m[:beta_l]),
        @sprintf("Low-target-value placebo proxy_after_cap=%.3f final beta_l=%.3f", placebo_value_m[:proxy_share_after_cap], placebo_value_m[:beta_l]),
        @sprintf("Robust total-KL: beta_l=%.3f total_K_h=%.4f proxy_after_cap=%.3f",
            robust_m[:beta_l], robust_m[:K_h], robust_m[:proxy_share_after_cap]),
        "Interpretation: under direct own-data regulation, high correlation and high target value can make proxy data attractive after the written target cap binds. The own-data cap is respected, but total inference about the target can exceed the intended cap. Under robust total-KL regulation, proxy actions that would exceed the target inference cap are blocked, so the proxy channel is sharply reduced.",
        "Tables: $(tables)",
        "Figures: $(outdirs[:figures])",
        "Logs: $(outdirs[:logs])",
        "Figure audit: $(figure_audit_path)",
    ]
    write_run_log(outdirs, interpretation)
    for line in interpretation
        println(line)
    end
    println("Figure formatting complete.")
    println("All figures saved in: ", outdirs[:figures])
    println("Figure audit saved in: ", figure_audit_path)
    return Dict(:outdirs => outdirs, :results => results, :grid_rows => grid_rows)
end
