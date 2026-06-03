"""
    solve_value_iteration(env)

Solve the exact finite-horizon dynamic program on the same precision-grid MDP
used by Q-learning. This is backward induction because time is part of the state.
"""
function solve_value_iteration(env::ProxyEnv)::Dict{Symbol,Any}
    V = zeros(Float64, env.nstates)
    Q = fill(-Inf, env.nstates, length(ACTIONS))
    policy = fill(ACTION_EXPLOIT, env.nstates)

    for t in (env.params.T - 1):-1:0
        for il in 1:env.nbeta
            for ih in 1:env.nbeta
                s = state_index(env, ih, il, t)
                best_val = -Inf
                best_action = ACTION_EXPLOIT
                for a in ACTIONS
                    if is_allowed_action(env, s, a)
                        ns, r, done = environment_step(env, s, a)
                        val = done ? r : r + env.params.gamma * V[ns]
                        Q[s, action_col(a)] = val
                        if val > best_val + 1.0e-12
                            best_val = val
                            best_action = a
                        end
                    end
                end
                V[s] = best_val
                policy[s] = best_action
            end
        end
    end
    return Dict{Symbol,Any}(:V => V, :Q => Q, :policy => policy)
end

"""
    evaluate_vi_policy(env, vi)

Evaluate the exact value-iteration policy from the initial state.
"""
function evaluate_vi_policy(env::ProxyEnv, vi::Dict{Symbol,Any})::Dict{Symbol,Any}
    policy = vi[:policy]
    return simulate_policy(env, s -> policy[s])
end

"""
    compare_q_to_value_iteration(env, Q, vi)

Compare Q-learning's greedy policy with exact value iteration over all
finite-horizon states and along the Q-greedy evaluation path.
"""
function compare_q_to_value_iteration(env::ProxyEnv, Q::Matrix{Float64}, vi::Dict{Symbol,Any})::Dict{Symbol,Float64}
    policy = vi[:policy]
    total = 0
    matches = 0
    regrets = Float64[]
    for t in 0:(env.params.T - 1)
        for il in 1:env.nbeta
            for ih in 1:env.nbeta
                s = state_index(env, ih, il, t)
                aq = greedy_action(Q, env, s)
                av = policy[s]
                total += 1
                aq == av && (matches += 1)
                push!(regrets, maximum(vi[:Q][s, action_col(a)] for a in allowed_actions(env, s)) - vi[:Q][s, action_col(aq)])
            end
        end
    end

    traj = evaluate_q_policy(env, Q)
    path_total = 0
    path_matches = 0
    for s in traj[:states][1:end-1]
        aq = greedy_action(Q, env, s)
        av = policy[s]
        path_total += 1
        aq == av && (path_matches += 1)
    end

    return Dict{Symbol,Float64}(
        :all_state_match_rate => matches / total,
        :path_match_rate => path_total == 0 ? 0.0 : path_matches / path_total,
        :avg_exact_regret => isempty(regrets) ? 0.0 : mean(regrets),
        :max_exact_regret => isempty(regrets) ? 0.0 : maximum(regrets),
    )
end

"""
    representative_cap_state(env)

Return a state where the target direct cap binds and proxy precision is still low.
"""
function representative_cap_state(env::ProxyEnv)::Int
    ih = beta_to_index(env, floor(env.cap_beta_h / env.params.dbeta) * env.params.dbeta)
    ih = max(1, min(ih, env.nbeta))
    while ih < env.nbeta && env.beta_grid[ih + 1] <= env.cap_beta_h + 1.0e-12
        ih += 1
    end
    il = 1
    t = min(10, env.params.T - 1)
    return state_index(env, ih, il, t)
end
