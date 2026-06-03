using Plots
using Plots.PlotMeasures

gr()

const FIG_SIZE = (1200, 800)
const FIG_DPI = 300
const FIG_FONT = "DejaVu Sans"

const BLUE = "#1f77b4"
const GREEN = "#2ca02c"
const RED = "#d62728"
const ORANGE = "#ff7f0e"
const PURPLE = "#9467bd"
const GRAY = "#7f7f7f"

function thesis_plot_kwargs(;
    legend=:best,
    left_margin=16mm,
    right_margin=22mm,
    bottom_margin=16mm,
    top_margin=10mm,
)
    return (
        size = FIG_SIZE,
        dpi = FIG_DPI,
        framestyle = :box,
        grid = true,
        legend = legend,
        fontfamily = FIG_FONT,
        titlefontsize = 16,
        guidefontsize = 13,
        tickfontsize = 11,
        legendfontsize = 11,
        linewidth = 2.5,
        left_margin = left_margin,
        right_margin = right_margin,
        bottom_margin = bottom_margin,
        top_margin = top_margin,
    )
end

function pdf_sibling(path::AbstractString)::String
    root, _ = splitext(path)
    return root * ".pdf"
end

function save_plot_pair(p, path::AbstractString)
    mkpath(dirname(path))
    savefig(p, path)
    savefig(p, pdf_sibling(path))
    return path
end

function finite_float_vector(values)::Vector{Float64}
    out = Float64[]
    for value in collect(values)
        x = Float64(value)
        isfinite(x) && push!(out, x)
    end
    return out
end

function padded_limits(values::Vector{Float64}; include_zero::Bool=false, upper_pad::Float64=0.10)
    isempty(values) && return (0.0, 1.0)
    lo = minimum(values)
    hi = maximum(values)
    if include_zero
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    end
    span = max(hi - lo, max(abs(hi), abs(lo), 1.0) * 0.05)
    return (lo - 0.05 * span, hi + upper_pad * span)
end

function plot_line_series(
    path::AbstractString,
    series;
    title::AbstractString,
    ylabel::AbstractString,
    xlabel::AbstractString="Episode/time",
    capline=nothing,
    caplabel::AbstractString="κ_h cap",
    legend=:best,
    ylims=nothing,
    yzero::Bool=false,
)
    p = plot(;
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        thesis_plot_kwargs(legend=legend)...,
    )
    all_y = Float64[]
    for (label, values, color) in series
        y = finite_float_vector(values)
        x = collect(0:(length(y) - 1))
        append!(all_y, y)
        plot!(p, x, y; label=label, color=color)
    end
    if capline !== nothing
        c = Float64(capline)
        push!(all_y, c)
        hline!(p, [c]; label=caplabel, linestyle=:dash, color=GRAY, linewidth=2.2)
    end
    if ylims !== nothing
        ylims!(p, ylims)
    elseif yzero
        _, hi = padded_limits(all_y; include_zero=true)
        ylims!(p, (0.0, max(hi, 1.0e-6)))
    end
    return save_plot_pair(p, path)
end

function plot_bar_chart(
    path::AbstractString,
    labels,
    values;
    title::AbstractString,
    ylabel::AbstractString,
    xlabel::AbstractString="Action",
    colors=[BLUE, RED, GREEN],
    annotations=String[],
    yzero::Bool=false,
)
    y = finite_float_vector(values)
    p = bar(
        labels,
        y;
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        label="",
        color=colors,
        bar_width=0.62,
        thesis_plot_kwargs(legend=false)...,
    )
    if yzero
        lo, hi = padded_limits(y; include_zero=true)
        ylims!(p, (min(lo, 0.0), hi))
    end
    return save_plot_pair(p, path)
end

function plot_grouped_bars(
    path::AbstractString,
    groups,
    labels,
    values;
    title::AbstractString,
    ylabel::AbstractString,
    xlabel::AbstractString="",
)
    y = Matrix{Float64}(values)
    centers = collect(1:length(groups))
    nseries = size(y, 2)
    offsets = nseries == 1 ? [0.0] : collect(range(-0.18, 0.18; length=nseries))
    colors = [GREEN, RED, BLUE, ORANGE, PURPLE]
    p = plot(;
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xticks=(centers, string.(groups)),
        thesis_plot_kwargs(legend=:outerright, right_margin=25mm, bottom_margin=18mm)...,
    )
    for j in 1:nseries
        bar!(
            p,
            centers .+ offsets[j],
            y[:, j];
            label=string(labels[j]),
            color=colors[mod1(j, length(colors))],
            bar_width=0.32,
        )
    end
    yvals = vec(y)
    _, hi = padded_limits(yvals; include_zero=true)
    ylims!(p, (0.0, hi))
    xlims!(p, (0.4, length(groups) + 0.6))
    return save_plot_pair(p, path)
end

function plot_heatmap(
    path::AbstractString,
    xlabels,
    ylabels,
    values;
    title::AbstractString,
    xlabel::AbstractString,
    ylabel::AbstractString,
    colorbar_title::AbstractString,
)
    z = Matrix{Float64}(values)
    xs = collect(1:length(xlabels))
    ys = collect(1:length(ylabels))
    p = heatmap(
        xs,
        ys,
        z;
        xticks=(xs, string.(xlabels)),
        yticks=(ys, string.(ylabels)),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        colorbar=true,
        colorbar_title=colorbar_title,
        color=:viridis,
        thesis_plot_kwargs(legend=false, right_margin=32mm, bottom_margin=18mm)...,
    )
    return save_plot_pair(p, path)
end
