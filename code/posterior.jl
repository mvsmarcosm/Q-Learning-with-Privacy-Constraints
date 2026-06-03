using LinearAlgebra

"""
    ModelParams(; kwargs...)

Container for the two-consumer proxy-learning environment. The platform values
MSE leakage reductions with weights `r_h` and `r_l`, while privacy regulation is
encoded by `regulation = :direct_cap` or `:total_kl_cap`.
"""
Base.@kwdef struct ModelParams
    rho::Float64 = 0.85
    r_h::Float64 = 10.0
    r_l::Float64 = 1.0
    theta::Float64 = 1.0
    cost_h::Float64 = 0.02
    cost_l::Float64 = 0.02
    cost_exploit::Float64 = 0.0
    gamma::Float64 = 0.95
    kappa_h::Float64 = 0.05
    terminal_multiplier::Float64 = 5.0
    dbeta::Float64 = 0.05
    beta_max::Float64 = 5.0
    T::Int = 60
    regulation::Symbol = :direct_cap
end

"""
    sigma_matrix(rho)

Return the two-consumer Gaussian prior covariance matrix with unit variances and
correlation `rho`.
"""
function sigma_matrix(rho::Float64)::Matrix{Float64}
    @assert abs(rho) < 1.0 "rho must be strictly between -1 and 1"
    return [1.0 rho; rho 1.0]
end

"""
    posterior_covariance(beta_h, beta_l, rho)

Compute `V(beta) = inv(inv(Sigma) + Diagonal(beta))`, the posterior covariance
after collecting direct precision `beta_h` and proxy precision `beta_l`.
"""
function posterior_covariance(beta_h::Float64, beta_l::Float64, rho::Float64)::Matrix{Float64}
    Sigma = sigma_matrix(rho)
    P = inv(Sigma) + Diagonal([beta_h, beta_l])
    V = inv(P)
    V = (V + V') / 2
    @assert isapprox(V, V'; atol=1.0e-10)
    @assert minimum(eigvals(Symmetric(V))) > 0.0
    return V
end

"""
    leakage(beta_h, beta_l, rho)

Return MSE leakage `(L_h, L_l)`, where `L_i = 1 - V[i,i]`.
"""
function leakage(beta_h::Float64, beta_l::Float64, rho::Float64)::Tuple{Float64,Float64}
    V = posterior_covariance(beta_h, beta_l, rho)
    return (1.0 - V[1, 1], 1.0 - V[2, 2])
end

"""
    target_leakage(beta_h, beta_l, rho)

Return MSE leakage about the high-value target consumer.
"""
target_leakage(beta_h::Float64, beta_l::Float64, rho::Float64)::Float64 =
    leakage(beta_h, beta_l, rho)[1]

"""
    total_kl_target(beta_h, beta_l, rho)

Return total Gaussian KL inference about the high-value target, equal to
`0.5 * log(1 / V[1,1])`.
"""
function total_kl_target(beta_h::Float64, beta_l::Float64, rho::Float64)::Float64
    V = posterior_covariance(beta_h, beta_l, rho)
    return 0.5 * log(1.0 / V[1, 1])
end

"""
    own_kl(beta)

Return own-data KL from direct precision about a consumer:
`0.5 * log(1 + beta)`.
"""
own_kl(beta::Float64)::Float64 = 0.5 * log1p(beta)

"""
    direct_cap_beta(kappa_h)

Convert the target own-data KL cap into an equivalent precision cap.
"""
direct_cap_beta(kappa_h::Float64)::Float64 = exp(2.0 * kappa_h) - 1.0

"""
    weighted_leakage(beta_h, beta_l, params)

Return the platform's weighted MSE leakage objective before costs.
"""
function weighted_leakage(beta_h::Float64, beta_l::Float64, params::ModelParams)::Float64
    Lh, Ll = leakage(beta_h, beta_l, params.rho)
    return params.theta * (params.r_h * Lh + params.r_l * Ll)
end

"""
    metrics_at(beta_h, beta_l, params)

Return a dictionary of posterior, leakage, and KL metrics at a precision pair.
"""
function metrics_at(beta_h::Float64, beta_l::Float64, params::ModelParams)::Dict{Symbol,Float64}
    Lh, Ll = leakage(beta_h, beta_l, params.rho)
    return Dict{Symbol,Float64}(
        :beta_h => beta_h,
        :beta_l => beta_l,
        :K_own_h => own_kl(beta_h),
        :K_h => total_kl_target(beta_h, beta_l, params.rho),
        :L_h => Lh,
        :L_l => Ll,
        :weighted_leakage => params.theta * (params.r_h * Lh + params.r_l * Ll),
    )
end

"""
    run_formula_tests()

Assert the key analytical formulas used in the thesis chapter.
"""
function run_formula_tests()
    rho = 0.85
    b = 1.35
    V = posterior_covariance(0.0, b, rho)
    @assert isapprox(V, V'; atol=1.0e-10)
    @assert minimum(eigvals(Symmetric(V))) > 0.0

    lhs_L = target_leakage(0.0, b, rho)
    rhs_L = rho^2 * b / (1.0 + b)
    @assert isapprox(lhs_L, rhs_L; atol=1.0e-10)

    lhs_K = total_kl_target(0.0, b, rho)
    rhs_K = 0.5 * log((1.0 + b) / (1.0 + b * (1.0 - rho^2)))
    @assert isapprox(lhs_K, rhs_K; atol=1.0e-10)

    kappa_h = 0.05
    cap = direct_cap_beta(kappa_h)
    @assert own_kl(cap) <= kappa_h + 1.0e-12
    @assert own_kl(cap + 1.0e-4) > kappa_h
    return true
end
