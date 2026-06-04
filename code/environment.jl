const ACTION_EXPLOIT = 0
const ACTION_DIRECT_H = 1
const ACTION_PROXY_L = 2
const ACTIONS = (ACTION_EXPLOIT, ACTION_DIRECT_H, ACTION_PROXY_L)

"""
    ProxyEnv(params)

Finite MDP environment with precision-grid states `(beta_h_index, beta_l_index, t)`.
The environment knows the posterior formulas and computes rewards, while the
Q-learning agent only observes states, actions, rewards, and next states.
"""
struct ProxyEnv
    params::ModelParams
    beta_grid::Vector{Float64}
    cap_beta_h::Float64
    nbeta::Int
    nstates::Int
end

"""
    make_env(params)

Construct a proxy-learning environment and its precision grid.
"""
function make_env(params::ModelParams)::ProxyEnv
    beta_grid = collect(0.0:params.dbeta:params.beta_max)
    nb = length(beta_grid)
    return ProxyEnv(params, beta_grid, direct_cap_beta(params.kappa_h), nb, nb * nb * (params.T + 1))
end

"""
    action_name(action)

Human-readable label for an action code.
"""
function action_name(action::Int)::String
    action == ACTION_EXPLOIT && return "exploit"
    action == ACTION_DIRECT_H && return "direct_h"
    action == ACTION_PROXY_L && return "proxy_l"
    return "unknown"
end

"""
    beta_to_index(env, beta)

Map a precision value to the nearest grid index.
"""
function beta_to_index(env::ProxyEnv, beta::Float64)::Int
    idx = Int(round(beta / env.params.dbeta)) + 1
    return clamp(idx, 1, env.nbeta)
end

"""
    state_index(env, ih, il, t)

Encode `(beta_h_index, beta_l_index, t)` as a one-based integer state index.
"""
function state_index(env::ProxyEnv, ih::Int, il::Int, t::Int)::Int
    @assert 1 <= ih <= env.nbeta
    @assert 1 <= il <= env.nbeta
    @assert 0 <= t <= env.params.T
    return ih + (il - 1) * env.nbeta + t * env.nbeta * env.nbeta
end

"""
    decode_state(env, s)

Decode an integer state into `(beta_h_index, beta_l_index, t)`.
"""
function decode_state(env::ProxyEnv, s::Int)::Tuple{Int,Int,Int}
    @assert 1 <= s <= env.nstates
    z = s - 1
    per_t = env.nbeta * env.nbeta
    t = div(z, per_t)
    rem1 = z - t * per_t
    il = div(rem1, env.nbeta) + 1
    ih = rem1 - (il - 1) * env.nbeta + 1
    return ih, il, t
end

"""
    beta_values(env, ih, il)

Return precision values from precision-grid indices.
"""
beta_values(env::ProxyEnv, ih::Int, il::Int)::Tuple{Float64,Float64} =
    (env.beta_grid[ih], env.beta_grid[il])

"""
    direct_cap_binds(env, ih)

Return true once the next direct target signal would violate the written own-data cap.
"""
function direct_cap_binds(env::ProxyEnv, ih::Int)::Bool
    next_beta_h = env.beta_grid[min(ih + 1, env.nbeta)]
    return next_beta_h > env.cap_beta_h + 1.0e-12
end

"""
    would_satisfy_regulation(env, beta_h, beta_l)

Return whether a candidate next precision pair satisfies the active privacy rule.
"""
function would_satisfy_regulation(env::ProxyEnv, beta_h::Float64, beta_l::Float64)::Bool
    p = env.params
    if p.regulation == :direct_cap
        return beta_h <= env.cap_beta_h + 1.0e-12
    elseif p.regulation == :total_kl_cap
        return total_kl_target(beta_h, beta_l, p.rho) <= p.kappa_h + 1.0e-12
    else
        error("Unknown regulation $(p.regulation)")
    end
end

"""
    is_allowed_action(env, state, action)

Return true if an action is feasible from the current state under the privacy rule.
"""
function is_allowed_action(env::ProxyEnv, state::Int, action::Int)::Bool
    ih, il, t = decode_state(env, state)
    t >= env.params.T && return action == ACTION_EXPLOIT
    beta_h, beta_l = beta_values(env, ih, il)
    if action == ACTION_EXPLOIT
        return true
    elseif action == ACTION_DIRECT_H
        ih >= env.nbeta && return false
        beta_h_next = env.beta_grid[ih + 1]
        return would_satisfy_regulation(env, beta_h_next, beta_l)
    elseif action == ACTION_PROXY_L
        il >= env.nbeta && return false
        beta_l_next = env.beta_grid[il + 1]
        return would_satisfy_regulation(env, beta_h, beta_l_next)
    else
        return false
    end
end

"""
    allowed_actions(env, state)

Return feasible actions as action codes `0`, `1`, and/or `2`.
"""
function allowed_actions(env::ProxyEnv, state::Int)::Vector{Int}
    out = Int[]
    for a in ACTIONS
        is_allowed_action(env, state, a) && push!(out, a)
    end
    return out
end

"""
    transition_state(env, state, action)

Return the next state after applying a feasible action.
"""
function transition_state(env::ProxyEnv, state::Int, action::Int)::Int
    @assert is_allowed_action(env, state, action)
    ih, il, t = decode_state(env, state)
    t >= env.params.T && return state
    ih2, il2 = ih, il
    if action == ACTION_DIRECT_H
        ih2 += 1
    elseif action == ACTION_PROXY_L
        il2 += 1
    end
    return state_index(env, ih2, il2, t + 1)
end

"""
    action_cost(env, action)

Return the experimentation cost of an action.
"""
function action_cost(env::ProxyEnv, action::Int)::Float64
    p = env.params
    action == ACTION_DIRECT_H && return p.cost_h
    action == ACTION_PROXY_L && return p.cost_l
    return p.cost_exploit
end

"""
    step_reward(env, state, action, next_state)

Compute incremental platform reward plus terminal exploitation payoff if the
transition reaches `T`.
"""
function step_reward(env::ProxyEnv, state::Int, action::Int, next_state::Int)::Float64
    ih, il, _ = decode_state(env, state)
    ih2, il2, t2 = decode_state(env, next_state)
    bh, bl = beta_values(env, ih, il)
    bh2, bl2 = beta_values(env, ih2, il2)
    p = env.params
    reward = weighted_leakage(bh2, bl2, p) - weighted_leakage(bh, bl, p) - action_cost(env, action)
    if t2 == p.T
        reward += p.terminal_multiplier * weighted_leakage(bh2, bl2, p)
    end
    return reward
end

"""
    environment_step(env, state, action)

Apply one feasible MDP action and return `(next_state, reward, done)`.
"""
function environment_step(env::ProxyEnv, state::Int, action::Int)::Tuple{Int,Float64,Bool}
    ns = transition_state(env, state, action)
    r = step_reward(env, state, action, ns)
    _, _, t2 = decode_state(env, ns)
    return ns, r, t2 >= env.params.T
end

"""
    initial_state(env)

Return the starting state with zero precision and time zero.
"""
initial_state(env::ProxyEnv)::Int = state_index(env, 1, 1, 0)

"""
    terminal_metrics(env, final_state)

Return posterior and privacy metrics at an episode's final precision pair.
"""
function terminal_metrics(env::ProxyEnv, final_state::Int)::Dict{Symbol,Float64}
    ih, il, _ = decode_state(env, final_state)
    bh, bl = beta_values(env, ih, il)
    return metrics_at(bh, bl, env.params)
end

"""
    simulate_policy(env, choose_action)

Evaluate a deterministic policy from the initial state and return trajectory
metrics, actions, rewards, and proxy-use summaries.
"""
function simulate_policy(env::ProxyEnv, choose_action::Function)::Dict{Symbol,Any}
    state = initial_state(env)
    states = Int[state]
    actions = Int[]
    rewards = Float64[]
    beta_h_path = Float64[]
    beta_l_path = Float64[]
    K_own_path = Float64[]
    K_total_path = Float64[]
    Lh_path = Float64[]
    after_cap_actions = 0
    after_cap_proxy = 0

    for _ in 1:env.params.T
        ih, il, _ = decode_state(env, state)
        bh, bl = beta_values(env, ih, il)
        push!(beta_h_path, bh)
        push!(beta_l_path, bl)
        push!(K_own_path, own_kl(bh))
        push!(K_total_path, total_kl_target(bh, bl, env.params.rho))
        push!(Lh_path, target_leakage(bh, bl, env.params.rho))

        cap_bound = direct_cap_binds(env, ih)
        action = choose_action(state)
        @assert is_allowed_action(env, state, action)
        if cap_bound
            after_cap_actions += 1
            action == ACTION_PROXY_L && (after_cap_proxy += 1)
        end
        next_state, reward, done = environment_step(env, state, action)
        push!(actions, action)
        push!(rewards, reward)
        push!(states, next_state)
        state = next_state
        done && break
    end

    ih, il, _ = decode_state(env, state)
    bh, bl = beta_values(env, ih, il)
    push!(beta_h_path, bh)
    push!(beta_l_path, bl)
    push!(K_own_path, own_kl(bh))
    push!(K_total_path, total_kl_target(bh, bl, env.params.rho))
    push!(Lh_path, target_leakage(bh, bl, env.params.rho))

    final = terminal_metrics(env, state)
    final[:proxy_share_after_cap] = after_cap_actions == 0 ? 0.0 : after_cap_proxy / after_cap_actions
    final[:total_reward] = sum(rewards)
    final[:total_KL_violation] = final[:K_h] > env.params.kappa_h + 1.0e-10 ? 1.0 : 0.0
    return Dict{Symbol,Any}(
        :states => states,
        :actions => actions,
        :rewards => rewards,
        :final_state => state,
        :final_metrics => final,
        :beta_h_path => beta_h_path,
        :beta_l_path => beta_l_path,
        :K_own_path => K_own_path,
        :K_total_path => K_total_path,
        :Lh_path => Lh_path,
        :proxy_share_after_cap => final[:proxy_share_after_cap],
    )
end
