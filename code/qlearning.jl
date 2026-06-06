using Random
using Statistics
using Printf

"""
    LearningParams(; kwargs...)

Hyperparameters for tabular Q-learning.
"""
Base.@kwdef struct LearningParams
    episodes::Int = 50_000
    alpha0::Float64 = 0.20
    alpha_min::Float64 = 0.02
    epsilon0::Float64 = 0.30
    epsilon_min::Float64 = 0.02
    epsilon_decay::Float64 = 0.9995
    seed::Int = 1234
end

"""
    action_col(action)

Convert action code `0`, `1`, or `2` into a one-based Q-table column.
"""
action_col(action::Int)::Int = action + 1

"""
    learning_rate(lp, episode)

Episode-level learning rate schedule bounded below by `alpha_min`.
"""
learning_rate(lp::LearningParams, episode::Int)::Float64 =
    max(lp.alpha_min, lp.alpha0 / sqrt(1.0 + (episode - 1) / 1000.0))

"""
    exploration_rate(lp, episode)

Episode-level epsilon schedule for epsilon-greedy exploration.
"""
exploration_rate(lp::LearningParams, episode::Int)::Float64 =
    max(lp.epsilon_min, lp.epsilon0 * lp.epsilon_decay^(episode - 1))

"""
    greedy_action(Q, env, state; rng=nothing)

Return the feasible action with the largest Q-value. Random tie-breaking is
used only when an RNG is supplied.
"""
function greedy_action(Q::Matrix{Float64}, env::ProxyEnv, state::Int; rng::Union{Nothing,AbstractRNG}=nothing)::Int
    acts = allowed_actions(env, state)
    vals = [Q[state, action_col(a)] for a in acts]
    best = maximum(vals)
    idxs = findall(v -> isapprox(v, best; atol=1.0e-12), vals)
    chosen = isnothing(rng) ? idxs[1] : rand(rng, idxs)
    return acts[chosen]
end

"""
    choose_epsilon_greedy(Q, env, state, epsilon, rng)

Sample an epsilon-greedy feasible action.
"""
function choose_epsilon_greedy(Q::Matrix{Float64}, env::ProxyEnv, state::Int, epsilon::Float64, rng::AbstractRNG)::Int
    acts = allowed_actions(env, state)
    if rand(rng) < epsilon
        return rand(rng, acts)
    end
    return greedy_action(Q, env, state; rng=rng)
end

"""
    max_next_q(Q, env, state)

Return the maximum feasible Q-value in a next state.
"""
function max_next_q(Q::Matrix{Float64}, env::ProxyEnv, state::Int)::Float64
    acts = allowed_actions(env, state)
    isempty(acts) && return 0.0
    return maximum(Q[state, action_col(a)] for a in acts)
end

"""
    train_q_learning(env, lp; verbose=false)

Train a tabular Q-learning agent on the finite proxy-learning MDP.
"""
function train_q_learning(env::ProxyEnv, lp::LearningParams; verbose::Bool=false)::Dict{Symbol,Any}
    rng = MersenneTwister(lp.seed)
    Q = zeros(Float64, env.nstates, length(ACTIONS))
    episode_rewards = zeros(Float64, lp.episodes)
    proxy_after_cap = zeros(Float64, lp.episodes)
    final_beta_h = zeros(Float64, lp.episodes)
    final_beta_l = zeros(Float64, lp.episodes)

    for ep in 1:lp.episodes
        alpha = learning_rate(lp, ep)
        epsilon = exploration_rate(lp, ep)
        state = initial_state(env)
        total_reward = 0.0
        after_cap_actions = 0
        after_cap_proxy = 0

        for _ in 1:env.params.T
            ih, _, _ = decode_state(env, state)
            cap_bound = direct_cap_binds(env, ih)
            action = choose_epsilon_greedy(Q, env, state, epsilon, rng)
            next_state, reward, done = environment_step(env, state, action)

            target = reward
            if !done
                target += env.params.gamma * max_next_q(Q, env, next_state)
            end
            col = action_col(action)
            Q[state, col] = (1.0 - alpha) * Q[state, col] + alpha * target
            total_reward += reward

            if cap_bound
                after_cap_actions += 1
                action == ACTION_PROXY_L && (after_cap_proxy += 1)
            end
            state = next_state
            done && break
        end

        ih, il, _ = decode_state(env, state)
        episode_rewards[ep] = total_reward
        proxy_after_cap[ep] = after_cap_actions == 0 ? 0.0 : after_cap_proxy / after_cap_actions
        final_beta_h[ep], final_beta_l[ep] = beta_values(env, ih, il)

        if verbose && (ep == 1 || ep % max(1, div(lp.episodes, 5)) == 0)
            window = max(1, ep - min(ep - 1, 999)):ep
            @printf("    episode %d/%d, avg reward %.4f\n", ep, lp.episodes, mean(episode_rewards[window]))
        end
    end

    return Dict{Symbol,Any}(
        :Q => Q,
        :episode_rewards => episode_rewards,
        :proxy_after_cap => proxy_after_cap,
        :final_beta_h => final_beta_h,
        :final_beta_l => final_beta_l,
        :learning_params => lp,
    )
end

"""
    evaluate_q_policy(env, Q)

Evaluate the greedy policy induced by a learned Q-table.
"""
function evaluate_q_policy(env::ProxyEnv, Q::Matrix{Float64})::Dict{Symbol,Any}
    return simulate_policy(env, s -> greedy_action(Q, env, s))
end

"""
    q_values_at_state(Q, env, state)

Return Q-values for exploit, direct, and proxy actions, using `NaN` for
infeasible actions so figures do not treat them as available.
"""
function q_values_at_state(Q::Matrix{Float64}, env::ProxyEnv, state::Int)::Vector{Float64}
    vals = fill(NaN, length(ACTIONS))
    for a in ACTIONS
        if is_allowed_action(env, state, a)
            vals[action_col(a)] = Q[state, action_col(a)]
        end
    end
    return vals
end

"""
    best_learned_cap_binding_state(env, Q)

Find a non-terminal state where the target direct cap binds, the proxy action is
feasible, and the learned Q-table gives the proxy action the largest advantage
over the other displayed actions. This avoids plotting an unvisited all-zero
state when a learned cap-binding state is available.
"""
function best_learned_cap_binding_state(env::ProxyEnv, Q::Matrix{Float64})::Int
    fallback = representative_cap_state(env)
    best_state = fallback
    best_gap = -Inf
    for t in 0:(env.params.T - 1)
        for il in 1:env.nbeta
            beta_l = env.beta_grid[il]
            beta_l > min(2.5, env.params.beta_max) && continue
            for ih in 1:env.nbeta
                s = state_index(env, ih, il, t)
                direct_cap_binds(env, ih) || continue
                is_allowed_action(env, s, ACTION_PROXY_L) || continue
                q_proxy = Q[s, action_col(ACTION_PROXY_L)]
                q_exploit = Q[s, action_col(ACTION_EXPLOIT)]
                q_direct = Q[s, action_col(ACTION_DIRECT_H)]
                gap = q_proxy - max(q_exploit, q_direct)
                if gap > best_gap
                    best_gap = gap
                    best_state = s
                end
            end
        end
    end
    return best_state
end
