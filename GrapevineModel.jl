module GrapevineModel

export Grapevine, run_experiment, influence, run_influence_experiment, process_data, check_greedy, run_greedy_experiment

using StatsBase, StaticArrays, Graphs, Parameters, SNAPDatasets, JLD2, DataFrames

"""
    A type that represents an instance of the problem. 
    Basically packages all the parameters together.
    Properties:
        g::SimpleGraph{Int64}            #underlying graph
        sources::Vector{Int64}           #array of sources
        source_msg::Dict{Int64, Int64}   #original messages by sources
        mu10::Float64                    #probability of mutation from 1 to 0
        mu01::Float64                    #probability of mutation from 0 to 1
        p::Float64                       #probability of propagating the message
        theta::Float64                   #prior probability of agents that ω=1
"""
@with_kw struct Grapevine
    g::SimpleGraph{Int64} 
    sources::Vector{Int64} 
    source_msg::Dict{Int64, Int64} 
    mu10::Float64 
    mu01::Float64 
    p::Float64 
    theta::Float64 
end

"""
    M_trans(depth, p, mu10, mu01)

Compute transition probability between the message states (1, 0, and dropped) after `depth` mutations.

Returns M[i, j] = probability the message goes from i to j.
""" 
function M_trans(depth, p, mu10, mu01)
    d = depth
    
    M = SA[ p*(1-mu10) p*mu10     1-p
            p*mu01     p*(1-mu01) 1-p
            0          0          1  ]
    
    return M^d 
end

"Return the number of messages of type i∈{0, 1, -1} in I that came from each depth in levels."
count_msg(i, I, levels) = Dict(unique(levels) .=> [count(x -> x == i, I[levels .== d]) for d in unique(levels)])

"""
    posterior(theta::Float64, I::Array{Int}, levels::Array{Int}, mu10::Float64, mu01::Float64)
    posterior(inst::Grapevine, I::Array{Int}, levels::Array{Int})

Calculate the posterior probability of 1, given I, using Jack's derivation. 
"""
function posterior(theta::Float64, I::Array{Int}, levels::Array{Int}, mu10::Float64, mu01::Float64)
    mu = mu10 + mu01
    # @assert mu < 1 "μ has to be less than 1"
    
    x1 = count_msg(1, I, levels)
    x0 = count_msg(0, I, levels)
    
    chi = BigFloat((1-theta)/theta)
    for d in unique(levels)
        chi *= ( (mu10+mu01*(1-mu)^d)/(mu10*(1-(1-mu)^d)) )^x0[d] * 
               ( (mu01*(1-(1-mu)^d))/(mu01+mu10*(1-mu)^d) )^x1[d]
    end
    
    return 1/(1+chi)
end

posterior(inst::Grapevine, I::Array{Int}, levels::Array{Int}) = posterior(inst.theta, I, levels, inst.mu10, inst.mu01)

"""
    expected_learned_prob(theta::Float64, levels::Array{Int}, mu10::Float64, mu01::Float64, p::Float64, original_msg::Array{Int})
    expected_learned_prob(instance::Grapevine, levels::Array{Int})

Calculate the expected learned probability of an agent given the set of original_msg propagated by the sources.
"""
function expected_learned_prob(theta::Float64, levels::Array{Int}, mu10::Float64, mu01::Float64, p::Float64, original_msg::Array{Int})
    
    original_msg = map(x -> x == 0 ? 2 : x == -1 ? 3 : 1, original_msg)
    
    M = M_trans(1, p, mu10, mu01)
    
    s = length(levels) # number of sources
    
    total_prob = 0
    
    for I in Base.product(repeat([[-1, 0, 1]], s)...) # iterate over possible messages
        post = posterior(theta, collect(I), levels, mu10, mu01) # learned probabiltiy after seeing I
        
        # probability of receiving I
        I = map(x -> x == 0 ? 2 : x == -1 ? 3 : 1, I) # convert 0 -> 2, 1 -> 1, Nothing -> 3
        probI = prod([BigFloat((M^levels[k])[original_msg[k], I[k]]) for k in 1:s])
        
        total_prob += post*probI
    end
    
    return total_prob
end 

expected_learned_prob(instance::Grapevine, levels::Array{Int}) = expected_learned_prob(instance.theta, levels, instance.mu10, instance.mu01, 
    instance.p, [instance.source_msg[s] for s in instance.sources])

"""
    influence(instance::Grapevine, S_corr::Vector{Int}=Int[]; method=:analytic, verbose=true, dists_from_sources = Dict(instance.sources .=> [gdistances(instance.g, s) for s in instance.sources]))

Compute the influence of the set S_corr in an instance of Grapevine model.
"""
function influence(instance::Grapevine, S_corr::Vector{Int}=Int[]; method=:analytic, verbose=true, 
    dists_from_sources = Dict(instance.sources .=> [gdistances(instance.g, s) for s in instance.sources]))
    
    @unpack_Grapevine instance #get the parameters from the instance
    
    dists_from_sources = Dict(sources .=> [gdistances(g, s) for s in sources])

    learners = setdiff(vertices(g), sources)
    
    # create the corrupted instance
    new_msg = Dict(sources .=> map(x -> x in S_corr ? 1 - source_msg[x] : source_msg[x], sources)) 
    inst_corr = Grapevine(instance; source_msg = new_msg) # corrupted instance
    
    if method == :analytic

        # compute the original and corrupted score
        orig_score = 0
        corr_score = 0
        for i in learners
            levels = [dists_from_sources[s][i] for s in sources]
            orig_score += expected_learned_prob(instance, levels)
            corr_score += expected_learned_prob(inst_corr, levels)
        end
        avg_orig_score = Float64(orig_score/length(learners))
        avg_corr_score = Float64(corr_score/length(learners))

    elseif  method == :empirical
        N = 40


        exp_orig = run_experiment(instance, N; dists_from_sources=dists_from_sources)
        exp_corr = run_experiment(inst_corr, N; dists_from_sources=dists_from_sources)

        avg_orig_score = sum([mean(values(x)) for x in values(exp_orig)])/N
        avg_corr_score = sum([mean(values(x)) for x in values(exp_corr)])/N

    else
        @error "Method must be :analytic or :empirical"
        return
    end

    prob_decrease = avg_orig_score - avg_corr_score

    if isempty(S_corr)
        if verbose
            println("method = ", method)
            println("Average learned probability = $(round(avg_orig_score; digits=4))")
        end
        return avg_orig_score
    end

    if verbose
        println("method: ", method)
        println("Original average learned probability = $(round(avg_orig_score; digits=4))")
        println("After corrupting $(S_corr), the learned probability drops by  $(round(prob_decrease; digits=4))")
        println("Corrupted average learned probability = $(round(avg_corr_score; digits=4))")
    end
    return prob_decrease

end

"""
    run_experiment(instance::Grapevine, N=1; levels_all=nothing, dists_from_sources=nothing)

Run the experiment to simulate the information propagation in the Grapevine model for `N` iterations.
"""
function run_experiment(instance::Grapevine, N=1; levels_all=nothing, dists_from_sources=nothing)
    
    @unpack_Grapevine instance #get the parameters from the instance
    S = length(sources)
    learners = setdiff(vertices(g), sources)
    n = length(learners)
    
    #cache the distances from the sources
    if isnothing(dists_from_sources)
        dists_from_sources = Dict(sources .=> [gdistances(g, s) for s in sources])
    end
    
    data = Dict()

    for k in 1:N
        # print("\rIteration $k")
        
        learned_probs = Dict{Int, Float64}()
        for v in learners
            if levels_all == nothing
                levels = [dists_from_sources[s][v] for s in sources]
            else
                levels = levels_all[v]
            end

            #sample I
            I = zeros(Int, S)
            for i in 1:S
                M = M_trans(levels[i], p, mu10, mu01)
                j = source_msg[sources[i]]
                j = j == 0 ? 2 : j == -1 ? 3 : 1
                weights = Weights(M[j, :])
                I[i] = sample(SA[1, 0, -1], weights)
            end
            
            learned_probs[v] = posterior(instance, I, levels)
        end
        
        data[k] = learned_probs
    end
    # println()
    
    return data
end

"""
    powerset(x::Vector{T}) where T

Return all subsets of an array x.
"""
function powerset(x::Vector{T}) where T
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    return [Set(arr) for arr in result]
end

"""
    run_influence_experiment(; g = loadsnap(:facebook_combined), params = nothing, sources = 7, 
    n_exp = 1, data_path="data/new_data$(rand(1:1000000)).jld2", graph_name=nothing, method=:empirical)

Run the experiment which measures the influence of the corrupted sources. 
"""
function run_influence_experiment(; g = loadsnap(:facebook_combined), params = nothing, sources = 7, n_exp = 1, data_path="data/new_data$(rand(1:1000000)).jld2", graph_name=nothing, method=:empirical)

    if isfile(data_path)
        all_data = load_object(data_path)
    else
        all_data = Dict()
    end

    for i in 1:n_exp
        print("\rExperiment $i")

        if isnothing(params)
            theta = 0.5 + 0.5*rand() # prior probability the true state is 1
            p = 0.5 + 0.5*rand() # dropout rate
            mu10 = 0.5*rand() # mutation from 1 to 0
            mu01 = 0.5*rand() # mutation from 0 to 1
        else
            mu10, mu01, p, theta = params
        end
        
        if typeof(sources) == Int
            csources = sample(vertices(g), sources; replace=false)
        else
            csources = sources
        end
        
        original_msg = Dict(csources .=> ones(Int, length(csources)))
        instance = Grapevine(g, csources, original_msg, mu10, mu01, p, theta)

        S_pset = collect.(powerset(csources))
        data = Dict(S_pset .=> zeros(length(S_pset)))

        for S_corr in S_pset
            data[S_corr] = influence(instance, S_corr; method=method, verbose=false)
        end

        key = (mu10, mu01, p, theta, csources)
        if !isnothing(graph_name)
            key = (graph_name, key...)
        end
        
        all_data[key] = data

    end
    save_object(data_path, all_data)

    println()
    return all_data
end

"""
    greedy(data, k)
    greedy(data, k, centralities)

Run the greedy algorithm on the data for the budget of `k`. Optionally use centrailities instead of data. 
"""
function greedy(data, k)
    sources = sort(collect(keys(data)), by = x -> length(x))[end]
    current = Set(Int[])
    for i in 1:k
        best = [0, 0] # (source, influence)
        for s in sources
            new_set = push!(copy(current), s)
            if data[new_set] > best[2]
                best = [s, data[new_set]]
            end
        end
        push!(current, best[1])
    end
    return current
end

function greedy(data, k, centralities)
    sources = sort(collect(keys(data)), by = x -> length(x))[end]
    sources = collect(sources)
    best_sources = sources[sortperm(centralities[sources], rev=true)][1:k]
    return Set(best_sources)
end

"""
    check_greedy(data, tol=0.01, log=nothing, centralities=nothing)

Given data, check if Greedy outputs the optimal set. Difference of less than tol is considered non-significant. 

If centralities are provided, Greedy is using them instead of data.
Log the optimal and greedy sets and their scores in `log`.
"""
function check_greedy(data, tol=0.01, log=nothing, centralities=nothing)
    greedy_is_opt = true
    
    all_sets = Set.(sort(collect(keys(data)), by = x -> length(x)))
    # print(all_sets[end])
    if data[all_sets[end]] < 0.01 # if total influence is very small, ignore this data
        return greedy_is_opt
    end
    
    for set_size in 2:length(all_sets[end])-1
        sets = [x for x in all_sets if length(x) == set_size]
        sets_opt = sort(sets, by = set -> data[set])[end]
        if centralities == nothing
            sets_greedy = greedy(data, set_size)
        else
            sets_greedy = greedy(data, set_size, centralities)
        end
        
        if !issetequal(sets_opt, sets_greedy)
            if (data[sets_opt] - data[sets_greedy])/data[sets_greedy] > tol
            #     println("    Greedy is different from optimal
            # Greedy: $sets_greedy: $(data[sets_greedy])
            # Optimal: $sets_opt: $(data[sets_opt])\n")
                greedy_is_opt = false
                if !isnothing(log)
                    push!(log, (sets_greedy, data[sets_greedy], sets_opt, data[sets_opt]))
                end
            end
        end
    end
#     println("Check complete.")
    return greedy_is_opt
end

harmonic_centrality(g) = [sum(1 ./ gdistances(g,i)[gdistances(g, i) .> 0]) for i in vertices(g)]

function run_greedy_experiment(; heuristics=[], n_exp=1, data_path="data/greedy_test.jld2", params=nothing, m=30, g = loadsnap(:facebook_combined))
    if isfile(data_path)
        all_data = load_object(data_path)
    else
        all_data = Dict()
    end

    println("Experiment starting")
    for k in 2:m-1
        println("k = $k")

        for i in 1:n_exp
            # print("\rExperiment $i")

            # if isfile(data_path)
            #     all_data = load_object(data_path)
            # else
            #     all_data = Dict()
            # end

            if isnothing(params)
                theta = 0.5 + 0.5*rand() # prior probability the true state is 1
                p = 0.5 + 0.5*rand() # dropout rate
                mu10 = 0.5*rand() # mutation from 1 to 0
                mu01 = 0.5*rand() # mutation from 0 to 1
            else
                mu10, mu01, p, theta = params
            end
            
            csources = sample(vertices(g), m; replace=false)
            
            original_msg = Dict(csources .=> ones(Int, length(csources)))
            instance = Grapevine(g, csources, original_msg, mu10, mu01, p, theta)

            data = Dict()
            for x in [:greedy, :random, :closeness, :pagerank, :degree, :harmonic, :eigenvector]
                data[x] = 0.0
            end
            # run greedy - slow
            # println("Running greedy")
            sources = csources
            dists_from_sources = Dict(sources .=> [gdistances(g, s) for s in sources])

            greedy_set = Set(Int[])
            cbest = 0
            for i in 1:k
                # println("Iteration $i")
                best = [0, 0] # (source, influence)
                for s in sources
                    new_set = push!(copy(greedy_set), s)
                    new_set_inf = influence(instance, collect(new_set); method=:empirical, verbose=false, 
                        dists_from_sources = dists_from_sources)
                    if new_set_inf > best[2]
                        best = [s, new_set_inf]
                        cbest = best
                    end
                end
                push!(greedy_set, best[1])
            end
            data[:greedy] = cbest[2]

            # get random set
            random_set = sample(sources, k; replace=false)
            data[:random] = influence(instance, collect(random_set); method=:empirical, verbose=false, 
            dists_from_sources = dists_from_sources)

            # closeness centrality
            centralities = closeness_centrality(g)
            closeness_set = sources[sortperm(centralities[sources], rev=true)][1:k]
            data[:closeness] = influence(instance, collect(closeness_set); method=:empirical, verbose=false, 
            dists_from_sources = dists_from_sources)

            # pagerank centrality
            centralities = pagerank(g)
            pagerank_set = sources[sortperm(centralities[sources], rev=true)][1:k]
            data[:pagerank] = influence(instance, collect(pagerank_set); method=:empirical, verbose=false, 
            dists_from_sources = dists_from_sources)

            # degree centrality
            centralities = degree_centrality(g)
            degree_set = sources[sortperm(centralities[sources], rev=true)][1:k]
            data[:degree] = influence(instance, collect(degree_set); method=:empirical, verbose=false, 
            dists_from_sources = dists_from_sources)

            # harmonic centrality
            centralities = harmonic_centrality(g)
            harmonic_set = sources[sortperm(centralities[sources], rev=true)][1:k]
            data[:harmonic] = influence(instance, collect(harmonic_set); method=:empirical, verbose=false, 
            dists_from_sources = dists_from_sources)

            # eigenvector centrality
            centralities = eigenvector_centrality(g)
            eigenvector_set = sources[sortperm(centralities[sources], rev=true)][1:k]
            data[:eigenvector] = influence(instance, collect(eigenvector_set); method=:empirical, verbose=false, 
            dists_from_sources = dists_from_sources)

            key = (mu10, mu01, p, theta, sources, k)
            all_data[key] = data
            
        end

    end

    save_object(data_path, all_data)
    return all_data
end


function process_data(all_data, closeness, pagerank, degree, harmonic, eigenvector)
    algs_sets = [:greedy_set, :closeness_set, :pagerank_set, :degree_set, :harmonic_set, :eigenvector_set]
    algs_scores = [:greedy_score, :closeness_score, :pagerank_score, :degree_score, :harmonic_score, :eigenvector_score]

    col_names = [:mu10, :mu01, :p, :theta, :sources, :k, :opt_set, :opt, :avg, :worst, algs_sets..., algs_scores...]
    col_types = [fill(Float64, 4); Vector{Int}; Int; Set{Int}; fill(Float64, 3); fill(Set{Int}, 6); fill(Float64, 6)]
    log_df = DataFrame(col_names .=> [type[] for type in col_types])

    for x in keys(all_data)
        # convert to sets
        # println(x)
        rawdata = all_data[x]
        # println(rawdata)
        data = Dict()
        for key in keys(rawdata)
            data[Set(key)] = rawdata[key]
        end
        # println(data)

        all_sets = Set.(sort(collect(keys(data)), by = x -> length(x)))
        if data[all_sets[end]] < 0.05 # if total influence is very small, ignore this data
            continue
        end

        for k in 2:length(all_sets[end])-1
            sets = [x for x in all_sets if length(x) == k]
            sets = sort(sets, by = set -> data[set])

            opt_set = sets[end]
            opt = data[opt_set]
            avg = mean([data[set] for set in sets])
            worst = data[sets[1]]

            greedy_set = greedy(data, k)
            greedy_score = data[greedy_set]

            closeness_set = greedy(data, k, closeness)
            closeness_score = data[closeness_set]

            pagerank_set = greedy(data, k, pagerank)
            pagerank_score = data[pagerank_set]

            degree_set = greedy(data, k, degree)
            degree_score = data[degree_set]

            harmonic_set = greedy(data, k, harmonic)
            harmonic_score = data[harmonic_set]

            eigenvector_set = greedy(data, k, eigenvector)
            eigenvector_score = data[eigenvector_set]

            push!(log_df, (x..., k, opt_set, opt, avg, worst, 
                [greedy_set, closeness_set, pagerank_set, degree_set, harmonic_set, eigenvector_set]...,
                [greedy_score, closeness_score, pagerank_score, degree_score, harmonic_score, eigenvector_score]...))
    
        end
    end

    return log_df
end

# MODULE END
end