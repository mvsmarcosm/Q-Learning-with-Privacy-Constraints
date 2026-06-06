using Statistics
using Printf

const ROBUSTNESS_RHO_GRID = vcat(collect(0.0:0.1:0.9), [0.95])
const ROBUSTNESS_RH_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
const ROBUSTNESS_COST_L_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
const ROBUSTNESS_KAPPA_GRID = [0.01, 0.025, 0.05, 0.10, 0.20]
const ROBUSTNESS_GAMMA_GRID = [0.50, 0.75, 0.90, 0.95, 0.99]
const ROBUSTNESS_T_GRID = [20, 40, 60, 100]
const ROBUSTNESS_SEEDS = collect(20260601:20260610)
const ROBUSTNESS_HEATMAP_RHO_GRID = [0.0, 0.3, 0.6, 0.85, 0.95]
const ROBUSTNESS_HEATMAP_RH_GRID = [1.0, 5.0, 10.0, 20.0]

const ROBUSTNESS_LONG_HEADERS = [
    "robustness_type", "scenario", "varied_parameter", "varied_value",
    "rho", "r_h", "cost_l", "kappa_h", "gamma", "T", "dbeta", "beta_max",
    "regulation", "seed", "episodes",
    "final_beta_h", "final_beta_l", "K_own_h", "K_h", "L_h",
    "proxy_share_after_cap", "total_KL_violation", "evaluation_reward",
    "avg_reward_last_1000", "all_state_match_rate", "path_match_rate",
    "avg_exact_regret", "max_exact_regret",
]

"""
    ensure_robustness_dirs(output_dir)

Create the robustness output tree under the project output directory.
"""
function ensure_robustness_dirs(output_dir::AbstractString)::Dict{Symbol,String}
    root = joinpath(output_dir, "robustness")
    tables = joinpath(root, "tables")
    figures = joinpath(root, "figures")
    logs = joinpath(root, "logs")
    mkpath(tables)
    mkpath(figures)
    mkpath(logs)
    return Dict(:root => root, :tables => tables, :figures => figures, :logs => logs)
end

"""
    params_with(base; kwargs...)

Return a `ModelParams` copy with selected fields replaced.
"""
function params_with(base::ModelParams;
    rho=base.rho,
    r_h=base.r_h,
    r_l=base.r_l,
    theta=base.theta,
    cost_h=base.cost_h,
    cost_l=base.cost_l,
    cost_exploit=base.cost_exploit,
    gamma=base.gamma,
    kappa_h=base.kappa_h,
    terminal_multiplier=base.terminal_multiplier,
    dbeta=base.dbeta,
    beta_max=base.beta_max,
    T=base.T,
    regulation=base.regulation,
)
    return ModelParams(;
        rho=Float64(rho),
        r_h=Float64(r_h),
        r_l=Float64(r_l),
        theta=Float64(theta),
        cost_h=Float64(cost_h),
        cost_l=Float64(cost_l),
        cost_exploit=Float64(cost_exploit),
        gamma=Float64(gamma),
        kappa_h=Float64(kappa_h),
        terminal_multiplier=Float64(terminal_multiplier),
        dbeta=Float64(dbeta),
        beta_max=Float64(beta_max),
        T=Int(T),
        regulation=Symbol(regulation),
    )
end

"""
    robustness_learning_params(episodes, seed)

Create deterministic Q-learning parameters for one robustness case.
"""
robustness_learning_params(episodes::Int, seed::Int)::LearningParams =
    LearningParams(episodes=episodes, seed=seed)

"""
    collect_metrics(result)

Collect the common terminal and learning metrics from one robustness result.
"""
function collect_metrics(result::Dict{Symbol,Any})::Dict{Symbol,Any}
    p = result[:params]
    lp = result[:learning_params]
    m = result[:q_eval][:final_metrics]
    cmp = get(result, :comparison, Dict{Symbol,Float64}())
    return Dict{Symbol,Any}(
        :rho => p.rho,
        :r_h => p.r_h,
        :cost_l => p.cost_l,
        :kappa_h => p.kappa_h,
        :gamma => p.gamma,
        :T => p.T,
        :dbeta => p.dbeta,
        :beta_max => p.beta_max,
        :regulation => p.regulation,
        :seed => lp.seed,
        :episodes => lp.episodes,
        :final_beta_h => m[:beta_h],
        :final_beta_l => m[:beta_l],
        :K_own_h => m[:K_own_h],
        :K_h => m[:K_h],
        :L_h => m[:L_h],
        :proxy_share_after_cap => m[:proxy_share_after_cap],
        :total_KL_violation => m[:total_KL_violation],
        :evaluation_reward => m[:total_reward],
        :avg_reward_last_1000 => mean_last(result[:train][:episode_rewards], min(1000, length(result[:train][:episode_rewards]))),
        :all_state_match_rate => get(cmp, :all_state_match_rate, ""),
        :path_match_rate => get(cmp, :path_match_rate, ""),
        :avg_exact_regret => get(cmp, :avg_exact_regret, ""),
        :max_exact_regret => get(cmp, :max_exact_regret, ""),
    )
end

"""
    run_robustness_case(name, params, lp; do_vi=false)

Train Q-learning for one robustness case, evaluate the greedy policy, and
optionally solve the exact finite-horizon dynamic program for validation.
"""
function run_robustness_case(name::String, params::ModelParams, lp::LearningParams; do_vi::Bool=false)::Dict{Symbol,Any}
    println(@sprintf("  robustness %-34s rho=%.2f r_h=%.1f cost_l=%.3f kappa=%.3f gamma=%.2f T=%d regulation=%s seed=%d episodes=%d",
        name, params.rho, params.r_h, params.cost_l, params.kappa_h, params.gamma,
        params.T, string(params.regulation), lp.seed, lp.episodes))
    env = make_env(params)
    train = train_q_learning(env, lp; verbose=false)
    q_eval = evaluate_q_policy(env, train[:Q])
    vi = nothing
    vi_eval = nothing
    comparison = Dict{Symbol,Float64}()
    if do_vi
        vi = solve_value_iteration(env)
        vi_eval = evaluate_vi_policy(env, vi)
        comparison = compare_q_to_value_iteration(env, train[:Q], vi)
    end
    return Dict{Symbol,Any}(
        :name => name,
        :params => params,
        :learning_params => lp,
        :env => env,
        :train => train,
        :q_eval => q_eval,
        :vi => vi,
        :vi_eval => vi_eval,
        :comparison => comparison,
    )
end

function long_row(kind::String, scenario::String, parameter::String, value, result::Dict{Symbol,Any})
    m = collect_metrics(result)
    return [
        kind, scenario, parameter, value,
        m[:rho], m[:r_h], m[:cost_l], m[:kappa_h], m[:gamma], m[:T],
        m[:dbeta], m[:beta_max], m[:regulation], m[:seed], m[:episodes],
        m[:final_beta_h], m[:final_beta_l], m[:K_own_h], m[:K_h], m[:L_h],
        m[:proxy_share_after_cap], m[:total_KL_violation], m[:evaluation_reward],
        m[:avg_reward_last_1000], m[:all_state_match_rate], m[:path_match_rate],
        m[:avg_exact_regret], m[:max_exact_regret],
    ]
end

function metric_row(result::Dict{Symbol,Any}; robustness_type::String, scenario::String, varied_parameter::String, varied_value)
    return long_row(robustness_type, scenario, varied_parameter, varied_value, result)
end

function run_parameter_sweep(kind::String, parameter::Symbol, values, base::ModelParams, episodes::Int, seed_start::Int)
    rows = Any[]
    results = Dict{String,Dict{Symbol,Any}}()
    for (i, value) in enumerate(values)
        kwargs = Dict{Symbol,Any}(parameter => value, :regulation => :direct_cap)
        p = params_with(base; kwargs...)
        lp = robustness_learning_params(episodes, seed_start + i - 1)
        scenario = string(parameter, "_", value)
        result = run_robustness_case(scenario, p, lp)
        push!(rows, metric_row(result; robustness_type=kind, scenario=scenario,
            varied_parameter=string(parameter), varied_value=value))
        results[scenario] = result
    end
    return rows, results
end

function direct_total_pair_rows(base::ModelParams, scenario::String, params::ModelParams, episodes::Int, seed::Int)
    direct_params = params_with(params; regulation=:direct_cap)
    robust_params = params_with(params; regulation=:total_kl_cap)
    direct = run_robustness_case(scenario * "_direct_cap", direct_params, robustness_learning_params(episodes, seed))
    robust = run_robustness_case(scenario * "_total_kl_cap", robust_params, robustness_learning_params(episodes, seed + 10_000))
    md = collect_metrics(direct)
    mr = collect_metrics(robust)
    row = [
        scenario, seed, episodes, params.rho, params.r_h, params.cost_l, params.kappa_h,
        md[:evaluation_reward], mr[:evaluation_reward], md[:evaluation_reward] - mr[:evaluation_reward],
        md[:K_h], mr[:K_h], md[:K_h] - mr[:K_h],
        md[:final_beta_l], mr[:final_beta_l], md[:final_beta_l] - mr[:final_beta_l],
        md[:proxy_share_after_cap], mr[:proxy_share_after_cap],
        md[:K_own_h], mr[:K_own_h],
    ]
    long = [
        metric_row(direct; robustness_type="regime_comparison", scenario=scenario * "_direct_cap", varied_parameter="regulation", varied_value="direct_cap"),
        metric_row(robust; robustness_type="regime_comparison", scenario=scenario * "_total_kl_cap", varied_parameter="regulation", varied_value="total_kl_cap"),
    ]
    return row, long, direct, robust
end

function run_regime_comparison(base::ModelParams, episodes::Int)
    scenarios = [
        ("baseline", base),
        ("low_correlation", params_with(base; rho=0.05)),
        ("high_correlation", params_with(base; rho=0.95)),
        ("low_target_value", params_with(base; r_h=1.0)),
        ("high_target_value", params_with(base; r_h=20.0)),
    ]
    rows = Any[]
    long_rows = Any[]
    results = Dict{String,Dict{Symbol,Any}}()
    for (i, (name, p)) in enumerate(scenarios)
        row, lr, direct, robust = direct_total_pair_rows(base, name, p, episodes, 20260801 + i)
        push!(rows, row)
        append!(long_rows, lr)
        results[name * "_direct_cap"] = direct
        results[name * "_total_kl_cap"] = robust
    end
    return rows, long_rows, results
end

function run_q_vi_robustness(base::ModelParams, vi_episodes::Int)
    scenarios = [
        ("baseline_direct_cap", params_with(base; regulation=:direct_cap)),
        ("baseline_total_kl_cap", params_with(base; regulation=:total_kl_cap)),
        ("low_rho_direct_cap", params_with(base; rho=0.05, regulation=:direct_cap)),
        ("high_rho_direct_cap", params_with(base; rho=0.95, regulation=:direct_cap)),
        ("low_r_h_direct_cap", params_with(base; r_h=1.0, regulation=:direct_cap)),
        ("high_r_h_direct_cap", params_with(base; r_h=20.0, regulation=:direct_cap)),
    ]
    rows = Any[]
    long_rows = Any[]
    results = Dict{String,Dict{Symbol,Any}}()
    for (i, (name, p)) in enumerate(scenarios)
        result = run_robustness_case(name, p, robustness_learning_params(vi_episodes, 20260901 + i); do_vi=true)
        m = collect_metrics(result)
        push!(rows, [
            name, p.rho, p.r_h, p.cost_l, p.kappa_h, p.gamma, p.T, p.regulation,
            m[:seed], m[:episodes], m[:all_state_match_rate], m[:path_match_rate],
            m[:avg_exact_regret], m[:max_exact_regret],
        ])
        push!(long_rows, metric_row(result; robustness_type="q_vs_vi", scenario=name,
            varied_parameter="validation", varied_value=name))
        results[name] = result
    end
    return rows, long_rows, results
end

function run_seed_robustness(base::ModelParams, episodes::Int)
    rows = Any[]
    long_rows = Any[]
    results = Dict{String,Dict{Symbol,Any}}()
    for seed in ROBUSTNESS_SEEDS
        name = string("baseline_seed_", seed)
        result = run_robustness_case(name, params_with(base; regulation=:direct_cap), robustness_learning_params(episodes, seed))
        m = collect_metrics(result)
        push!(rows, ["seed", seed, "", m[:final_beta_l], m[:K_h], m[:proxy_share_after_cap], m[:evaluation_reward]])
        push!(long_rows, metric_row(result; robustness_type="seed_robustness", scenario=name,
            varied_parameter="seed", varied_value=seed))
        results[name] = result
    end
    metrics = [:final_beta_l, :K_h, :proxy_share_after_cap, :evaluation_reward]
    values_by_metric = Dict{Symbol,Vector{Float64}}()
    for metric in metrics
        values_by_metric[metric] = [Float64(row[findfirst(==(String(metric)), ["row_type", "seed", "statistic", "final_beta_l", "K_h", "proxy_share_after_cap", "evaluation_reward"])]) for row in rows]
    end
    for statname in ["mean", "std", "min", "max"]
        vals = Any[]
        for metric in metrics
            x = values_by_metric[metric]
            v = statname == "mean" ? mean(x) :
                statname == "std" ? std(x) :
                statname == "min" ? minimum(x) : maximum(x)
            push!(vals, v)
        end
        push!(rows, ["summary", "", statname, vals...])
    end
    return rows, long_rows, results
end

function run_heatmap_robustness(base::ModelParams, grid_episodes::Int)
    rows = Any[]
    long_rows = Any[]
    results = Dict{String,Dict{Symbol,Any}}()
    k = 0
    for rh in ROBUSTNESS_HEATMAP_RH_GRID
        for rho in ROBUSTNESS_HEATMAP_RHO_GRID
            k += 1
            p = params_with(base; rho=rho, r_h=rh, regulation=:direct_cap)
            name = @sprintf("heatmap_rho_%.2f_rh_%.1f", rho, rh)
            result = run_robustness_case(name, p, robustness_learning_params(grid_episodes, 20261000 + k))
            m = collect_metrics(result)
            violation_magnitude = max(Float64(m[:K_h]) - p.kappa_h, 0.0)
            push!(rows, [
                rho, rh, p.cost_l, p.kappa_h, p.gamma, p.T, p.regulation,
                m[:seed], m[:episodes], m[:final_beta_l], m[:K_h],
                m[:proxy_share_after_cap], violation_magnitude,
            ])
            push!(long_rows, metric_row(result; robustness_type="heatmap_rho_r_h", scenario=name,
                varied_parameter="rho_r_h", varied_value=@sprintf("%.2f|%.1f", rho, rh)))
            results[name] = result
        end
    end
    return rows, long_rows, results
end

function robust_grid_matrix(rows, value_index::Int)
    rho_values = sort(unique(Float64(row[1]) for row in rows))
    rh_values = sort(unique(Float64(row[2]) for row in rows))
    mat = zeros(Float64, length(rh_values), length(rho_values))
    for row in rows
        ix = findfirst(==(Float64(row[1])), rho_values)
        iy = findfirst(==(Float64(row[2])), rh_values)
        mat[iy, ix] = Float64(row[value_index])
    end
    return rho_values, rh_values, mat
end

function row_value(row, header, key::String)
    idx = findfirst(==(key), header)
    idx === nothing && error("Missing column $key")
    return row[idx]
end

function line_from_rows(path, rows, header; xkey::String, yseries, title::String, xlabel::String, ylabel::String, legend=:best, ylims=nothing, yzero=false, reference_equal::Bool=false)
    xs = [Float64(row_value(row, header, xkey)) for row in rows]
    p = plot(;
        xlabel=xlabel, ylabel=ylabel, title=title,
        thesis_plot_kwargs(legend=legend)...,
    )
    colors = [BLUE, GREEN, RED, ORANGE, PURPLE]
    all_y = Float64[]
    for (j, (label, key)) in enumerate(yseries)
        ys = [Float64(row_value(row, header, key)) for row in rows]
        append!(all_y, ys)
        plot!(p, xs, ys; label=label, color=colors[mod1(j, length(colors))], marker=:circle)
    end
    if reference_equal
        lo = minimum(xs)
        hi = maximum(xs)
        plot!(p, [lo, hi], [lo, hi]; label="κₕ reference", color=GRAY, linestyle=:dash, linewidth=2.2)
        append!(all_y, [lo, hi])
    end
    if ylims !== nothing
        ylims!(p, ylims)
    elseif yzero
        _, hi = padded_limits(all_y; include_zero=true)
        ylims!(p, (0.0, hi))
    end
    return save_plot_pair(p, path)
end

function bar_from_rows(path, rows, header; xkey::String, ykey::String, title::String, xlabel::String, ylabel::String)
    labels = string.([row_value(row, header, xkey) for row in rows])
    values = [Float64(row_value(row, header, ykey)) for row in rows]
    return plot_bar_chart(path, labels, values; title=title, xlabel=xlabel, ylabel=ylabel, colors=fill(BLUE, length(values)), yzero=false)
end

function seed_scatter(path, seed_rows, ykey::String, title::String, ylabel::String)
    header = ["row_type", "seed", "statistic", "final_beta_l", "K_h", "proxy_share_after_cap", "evaluation_reward"]
    rows = [row for row in seed_rows if row[1] == "seed"]
    x = [Float64(row_value(row, header, "seed")) for row in rows]
    y = [Float64(row_value(row, header, ykey)) for row in rows]
    p = scatter(
        x, y;
        xlabel="Seed", ylabel=ylabel, title=title,
        label="", color=BLUE, markersize=6,
        thesis_plot_kwargs(legend=false)...,
    )
    plot!(p, x, y; label="", color=BLUE, alpha=0.45)
    return save_plot_pair(p, path)
end

function build_robustness_figures(outdirs, tables)
    figs = outdirs[:figures]
    audit = Any[]
    long_header = ROBUSTNESS_LONG_HEADERS
    corr = tables[:correlation]
    target = tables[:target_value]
    cost = tables[:proxy_cost]
    kappa = tables[:privacy_cap]
    gamma = tables[:discount_factor]
    horizon = tables[:horizon]
    regime_header = [
        "scenario", "seed", "episodes", "rho", "r_h", "cost_l", "kappa_h",
        "evaluation_reward_direct", "evaluation_reward_total_kl", "reward_loss",
        "K_h_direct", "K_h_total_kl", "KL_reduction",
        "beta_l_direct", "beta_l_total_kl", "proxy_reduction",
        "proxy_share_direct", "proxy_share_total_kl", "K_own_h_direct", "K_own_h_total_kl",
    ]

    f = joinpath(figs, "robustness_rho_proxy_share.png")
    line_from_rows(f, corr, long_header; xkey="rho", yseries=[("Proxy share after cap", "proxy_share_after_cap")],
        title="Proxy use by correlation", xlabel="Correlation ρ", ylabel="Proxy action share after cap binds", legend=false, ylims=(0.0, 1.0))
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_rho_final_beta_l.png")
    line_from_rows(f, corr, long_header; xkey="rho", yseries=[("βₗ proxy", "final_beta_l")],
        title="Proxy precision by correlation", xlabel="Correlation ρ", ylabel="Proxy precision βₗ", legend=false, yzero=true)
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_rh_proxy_share.png")
    line_from_rows(f, target, long_header; xkey="r_h", yseries=[("Proxy share after cap", "proxy_share_after_cap")],
        title="Proxy use by target value", xlabel="Target value rₕ", ylabel="Proxy action share after cap binds", legend=false, ylims=(0.0, 1.0))
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_rh_total_KL.png")
    line_from_rows(f, target, long_header; xkey="r_h", yseries=[("Kₕ total", "K_h")],
        title="Total target KL by target value", xlabel="Target value rₕ", ylabel="Kₕ total", legend=false, yzero=true)
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_cost_proxy.png")
    line_from_rows(f, cost, long_header; xkey="cost_l", yseries=[("βₗ proxy", "final_beta_l")],
        title="Proxy precision by proxy cost", xlabel="Proxy experimentation cost", ylabel="Proxy precision βₗ", legend=false, yzero=true)
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_cost_proxy_share.png")
    line_from_rows(f, cost, long_header; xkey="cost_l", yseries=[("Proxy share after cap", "proxy_share_after_cap")],
        title="Proxy use by proxy cost", xlabel="Proxy experimentation cost", ylabel="Proxy action share after cap binds", legend=false, ylims=(0.0, 1.0))
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_kappa_total_KL.png")
    line_from_rows(f, kappa, long_header; xkey="kappa_h", yseries=[("Kₕ own", "K_own_h"), ("Kₕ total", "K_h")],
        title="Target KL by privacy cap", xlabel="Own-data cap κₕ", ylabel="Target KL", legend=:outerright, yzero=true, reference_equal=true)
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_gamma_proxy.png")
    line_from_rows(f, gamma, long_header; xkey="gamma", yseries=[("βₗ proxy", "final_beta_l"), ("Proxy share after cap", "proxy_share_after_cap")],
        title="Proxy learning by discount factor", xlabel="Discount factor γ", ylabel="Level", legend=:outerright, yzero=true)
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_horizon_proxy.png")
    line_from_rows(f, horizon, long_header; xkey="T", yseries=[("βₗ proxy", "final_beta_l"), ("Kₕ total", "K_h")],
        title="Proxy learning by horizon", xlabel="Horizon T", ylabel="Level", legend=:outerright, yzero=true)
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_regime_reward_loss.png")
    bar_from_rows(f, tables[:regime_comparison], regime_header; xkey="scenario", ykey="reward_loss",
        title="Private reward loss from robust total-KL regulation", xlabel="Scenario", ylabel="Reward loss")
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_regime_KL_reduction.png")
    bar_from_rows(f, tables[:regime_comparison], regime_header; xkey="scenario", ykey="KL_reduction",
        title="Inference reduction from robust total-KL regulation", xlabel="Scenario", ylabel="Kₕ reduction")
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_seed_distribution_K_h.png")
    seed_scatter(f, tables[:seed_summary], "K_h", "Seed robustness of total target KL", "Kₕ total")
    push!(audit, figure_file_status(f, false, ""))

    f = joinpath(figs, "robustness_seed_distribution_final_beta_l.png")
    seed_scatter(f, tables[:seed_summary], "final_beta_l", "Seed robustness of proxy precision", "Proxy precision βₗ")
    push!(audit, figure_file_status(f, false, ""))

    heat = tables[:heatmap]
    rho_vals, rh_vals, proxy_mat = robust_grid_matrix(heat, 12)
    f = joinpath(figs, "robustness_heatmap_rho_rh_proxy.png")
    plot_heatmap(f, [@sprintf("%.2f", x) for x in rho_vals], [@sprintf("%.0f", y) for y in rh_vals], proxy_mat;
        title="Proxy use over correlation and target value", xlabel="Correlation ρ", ylabel="Target value rₕ", colorbar_title="Proxy share")
    push!(audit, figure_file_status(f, false, ""))

    _, _, viol_mat = robust_grid_matrix(heat, 13)
    f = joinpath(figs, "robustness_heatmap_rho_rh_violation.png")
    plot_heatmap(f, [@sprintf("%.2f", x) for x in rho_vals], [@sprintf("%.0f", y) for y in rh_vals], viol_mat;
        title="Total KL violation magnitude", xlabel="Correlation ρ", ylabel="Target value rₕ", colorbar_title="max(Kₕ - κₕ, 0)")
    push!(audit, figure_file_status(f, false, ""))

    audit_path = joinpath(outdirs[:logs], "robustness_figure_audit.txt")
    open(audit_path, "w") do io
        println(io, "png_path,png_exists,png_size_kb,pdf_path,pdf_exists,pdf_size_kb,empty_data,q_values_cap_state_valid,manual_axis_label_annotations")
        for row in audit
            println(io, join(row, ","))
        end
    end
    return audit_path
end

function write_robustness_tables(outdirs, tables)
    tdir = outdirs[:tables]
    write_table(joinpath(tdir, "robustness_all_long.csv"), ROBUSTNESS_LONG_HEADERS, tables[:all_long])
    write_table(joinpath(tdir, "robustness_correlation.csv"), ROBUSTNESS_LONG_HEADERS, tables[:correlation])
    write_table(joinpath(tdir, "robustness_target_value.csv"), ROBUSTNESS_LONG_HEADERS, tables[:target_value])
    write_table(joinpath(tdir, "robustness_proxy_cost.csv"), ROBUSTNESS_LONG_HEADERS, tables[:proxy_cost])
    write_table(joinpath(tdir, "robustness_privacy_cap.csv"), ROBUSTNESS_LONG_HEADERS, tables[:privacy_cap])
    write_table(joinpath(tdir, "robustness_discount_factor.csv"), ROBUSTNESS_LONG_HEADERS, tables[:discount_factor])
    write_table(joinpath(tdir, "robustness_horizon.csv"), ROBUSTNESS_LONG_HEADERS, tables[:horizon])
    write_table(joinpath(tdir, "robustness_regime_comparison.csv"), [
        "scenario", "seed", "episodes", "rho", "r_h", "cost_l", "kappa_h",
        "evaluation_reward_direct", "evaluation_reward_total_kl", "reward_loss",
        "K_h_direct", "K_h_total_kl", "KL_reduction",
        "beta_l_direct", "beta_l_total_kl", "proxy_reduction",
        "proxy_share_direct", "proxy_share_total_kl", "K_own_h_direct", "K_own_h_total_kl",
    ], tables[:regime_comparison])
    write_table(joinpath(tdir, "robustness_q_vs_vi.csv"), [
        "scenario", "rho", "r_h", "cost_l", "kappa_h", "gamma", "T", "regulation",
        "seed", "episodes", "all_state_match_rate", "path_match_rate",
        "avg_exact_regret", "max_exact_regret",
    ], tables[:q_vs_vi])
    write_table(joinpath(tdir, "robustness_seed_summary.csv"), [
        "row_type", "seed", "statistic", "final_beta_l", "K_h", "proxy_share_after_cap", "evaluation_reward",
    ], tables[:seed_summary])
    write_table(joinpath(tdir, "robustness_heatmap_rho_rh.csv"), [
        "rho", "r_h", "cost_l", "kappa_h", "gamma", "T", "regulation", "seed",
        "episodes", "final_beta_l", "K_h", "proxy_share_after_cap", "violation_magnitude",
    ], tables[:heatmap])
end

function robustness_quality_checks(output_dir, outdirs, tables, q_audit_path::Union{Nothing,String}=nothing)
    checks = Dict{Symbol,Bool}()
    required_tables = [
        "robustness_all_long.csv",
        "robustness_correlation.csv",
        "robustness_target_value.csv",
        "robustness_proxy_cost.csv",
        "robustness_privacy_cap.csv",
        "robustness_discount_factor.csv",
        "robustness_horizon.csv",
        "robustness_regime_comparison.csv",
        "robustness_q_vs_vi.csv",
        "robustness_seed_summary.csv",
    ]
    checks[:tables_have_rows] = all(begin
        path = joinpath(outdirs[:tables], file)
        isfile(path) && length(readlines(path)) > 1
    end for file in required_tables)

    expected_figs = [
        "robustness_rho_proxy_share",
        "robustness_rho_final_beta_l",
        "robustness_rh_proxy_share",
        "robustness_rh_total_KL",
        "robustness_cost_proxy",
        "robustness_cost_proxy_share",
        "robustness_kappa_total_KL",
        "robustness_gamma_proxy",
        "robustness_horizon_proxy",
        "robustness_regime_reward_loss",
        "robustness_regime_KL_reduction",
        "robustness_seed_distribution_K_h",
        "robustness_seed_distribution_final_beta_l",
        "robustness_heatmap_rho_rh_proxy",
        "robustness_heatmap_rho_rh_violation",
    ]
    checks[:figures_exist] = all(begin
        png = joinpath(outdirs[:figures], name * ".png")
        pdf = joinpath(outdirs[:figures], name * ".pdf")
        isfile(png) && stat(png).size > 0 && isfile(pdf) && stat(pdf).size > 0
    end for name in expected_figs)

    checks[:q_cap_binding_valid] = q_audit_path === nothing ? true :
        (isfile(q_audit_path) && occursin("q_values_when_cap_binds.png,true", read(q_audit_path, String)) &&
         occursin("q_values_when_cap_binds.pdf,true", read(q_audit_path, String)) &&
         occursin("false,true,false", read(q_audit_path, String)))

    checks[:total_kl_cap_respects_cap] = all(begin
        string(row[13]) == "total_kl_cap" ? Float64(row[19]) <= Float64(row[8]) + 1.0e-8 : true
    end for row in tables[:all_long])

    corr = tables[:correlation]
    low_corr = first(row for row in corr if isapprox(Float64(row[4]), 0.0; atol=1.0e-12))
    high_corr = first(row for row in corr if isapprox(Float64(row[4]), 0.95; atol=1.0e-12))
    checks[:high_correlation_more_proxy_learning] =
        Float64(high_corr[17]) > Float64(low_corr[17]) &&
        Float64(high_corr[21]) >= Float64(low_corr[21])

    target = tables[:target_value]
    low_rh = first(row for row in target if isapprox(Float64(row[4]), 1.0; atol=1.0e-12))
    high_rh = first(row for row in target if isapprox(Float64(row[4]), 40.0; atol=1.0e-12))
    checks[:high_target_value_more_proxy_learning] =
        Float64(high_rh[17]) >= Float64(low_rh[17]) &&
        Float64(high_rh[21]) >= Float64(low_rh[21])

    checks[:no_empty_or_nan_data] = all(row -> all(x -> !(x isa Float64 && !isfinite(x)), row), tables[:all_long])
    return checks
end

function write_robustness_log(outdirs, tables, checks, figure_audit_path)
    base_row = first(row for row in tables[:regime_comparison] if row[1] == "baseline")
    qvi_base = first(row for row in tables[:q_vs_vi] if row[1] == "baseline_direct_cap")
    seed_rows = tables[:seed_summary]
    seed_mean = first(row for row in seed_rows if row[1] == "summary" && row[3] == "mean")
    path = joinpath(outdirs[:logs], "robustness_run_log.txt")
    open(path, "w") do io
        println(io, "Robustness simulation run complete.")
        println(io, "Number of long-format runs: ", length(tables[:all_long]))
        println(io, "Correlation grid: ", ROBUSTNESS_RHO_GRID)
        println(io, "Target-value grid: ", ROBUSTNESS_RH_GRID)
        println(io, "Proxy-cost grid: ", ROBUSTNESS_COST_L_GRID)
        println(io, "Privacy-cap grid: ", ROBUSTNESS_KAPPA_GRID)
        println(io, "Discount-factor grid: ", ROBUSTNESS_GAMMA_GRID)
        println(io, "Horizon grid: ", ROBUSTNESS_T_GRID)
        println(io, "Tables: ", outdirs[:tables])
        println(io, "Figures: ", outdirs[:figures])
        println(io, "Logs: ", outdirs[:logs])
        println(io, @sprintf("Baseline direct-cap: K_own_h=%.6f K_h=%.6f beta_l=%.6f reward=%.6f",
            Float64(base_row[19]), Float64(base_row[11]), Float64(base_row[14]), Float64(base_row[8])))
        println(io, "Direct cap violates total KL: ", Float64(base_row[11]) > Float64(base_row[7]) + 1.0e-8)
        println(io, @sprintf("Robust total-KL: K_h=%.6f beta_l=%.6f reward=%.6f",
            Float64(base_row[12]), Float64(base_row[15]), Float64(base_row[9])))
        println(io, @sprintf("Robust regulation reward_loss=%.6f KL_reduction=%.6f proxy_reduction=%.6f",
            Float64(base_row[10]), Float64(base_row[13]), Float64(base_row[16])))
        println(io, @sprintf("Value-iteration validation baseline direct: all_state_match=%.4f path_match=%.4f avg_regret=%.6f max_regret=%.6f",
            Float64(qvi_base[11]), Float64(qvi_base[12]), Float64(qvi_base[13]), Float64(qvi_base[14])))
        println(io, @sprintf("Seed robustness mean: beta_l=%.6f K_h=%.6f proxy_share=%.6f reward=%.6f",
            Float64(seed_mean[4]), Float64(seed_mean[5]), Float64(seed_mean[6]), Float64(seed_mean[7])))
        println(io, "Figure audit: ", figure_audit_path)
        for key in sort(collect(keys(checks)); by=string)
            println(io, string("check_", key, "=", checks[key]))
        end
    end
    return path
end

"""
    run_robustness_pipeline(output_dir; episodes=10000, grid_episodes=3000, vi_episodes=episodes)

Run thesis robustness simulations and write CSV tables, figures, and logs under
`output_dir/robustness`. Use `episodes=2000, grid_episodes=500` for a fast mode
and `episodes=50000, grid_episodes=5000` for a paper mode.
"""
function run_robustness_pipeline(output_dir::AbstractString=DEFAULT_OUTPUT_DIR;
    episodes::Int=10_000,
    grid_episodes::Int=3_000,
    vi_episodes::Int=episodes,
    q_audit_path::Union{Nothing,String}=nothing,
)
    run_formula_tests()
    outdirs = ensure_robustness_dirs(output_dir)
    base = ModelParams()
    all_long = Any[]

    corr_rows, _ = run_parameter_sweep("correlation", :rho, ROBUSTNESS_RHO_GRID, base, episodes, 20260701)
    append!(all_long, corr_rows)
    target_rows, _ = run_parameter_sweep("target_value", :r_h, ROBUSTNESS_RH_GRID, base, episodes, 20260751)
    append!(all_long, target_rows)
    cost_rows, _ = run_parameter_sweep("proxy_cost", :cost_l, ROBUSTNESS_COST_L_GRID, base, episodes, 20260801)
    append!(all_long, cost_rows)
    kappa_rows, _ = run_parameter_sweep("privacy_cap", :kappa_h, ROBUSTNESS_KAPPA_GRID, base, episodes, 20260851)
    append!(all_long, kappa_rows)
    gamma_rows, _ = run_parameter_sweep("discount_factor", :gamma, ROBUSTNESS_GAMMA_GRID, base, episodes, 20260901)
    append!(all_long, gamma_rows)
    horizon_rows, _ = run_parameter_sweep("horizon", :T, ROBUSTNESS_T_GRID, base, episodes, 20260951)
    append!(all_long, horizon_rows)

    regime_rows, regime_long, _ = run_regime_comparison(base, episodes)
    append!(all_long, regime_long)
    qvi_rows, qvi_long, _ = run_q_vi_robustness(base, vi_episodes)
    append!(all_long, qvi_long)
    seed_rows, seed_long, _ = run_seed_robustness(base, episodes)
    append!(all_long, seed_long)
    heatmap_rows, heatmap_long, _ = run_heatmap_robustness(base, grid_episodes)
    append!(all_long, heatmap_long)

    tables = Dict{Symbol,Any}(
        :all_long => all_long,
        :correlation => corr_rows,
        :target_value => target_rows,
        :proxy_cost => cost_rows,
        :privacy_cap => kappa_rows,
        :discount_factor => gamma_rows,
        :horizon => horizon_rows,
        :regime_comparison => regime_rows,
        :q_vs_vi => qvi_rows,
        :seed_summary => seed_rows,
        :heatmap => heatmap_rows,
    )
    write_robustness_tables(outdirs, tables)
    figure_audit_path = build_robustness_figures(outdirs, tables)
    checks = robustness_quality_checks(output_dir, outdirs, tables, q_audit_path)
    log_path = write_robustness_log(outdirs, tables, checks, figure_audit_path)

    println("Robustness simulation complete.")
    println("Robustness tables saved in: ", outdirs[:tables])
    println("Robustness figures saved in: ", outdirs[:figures])
    println("Robustness log saved in: ", log_path)
    for key in sort(collect(keys(checks)); by=string)
        println(string("  check_", key, " = ", checks[key]))
    end
    return Dict(:outdirs => outdirs, :tables => tables, :checks => checks, :figure_audit => figure_audit_path, :log => log_path)
end
