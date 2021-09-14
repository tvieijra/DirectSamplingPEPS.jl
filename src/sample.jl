function perfect_sample_boundary!(mpo::MPO{T}, norm::T, sz_zero::Bool, num_up::Int64, num_down::Int64, num_max::Int64, marginal::Array{Array{T,4},1}) where T
    row_sample = zeros(Int64, mpo.Lx)
    boundary_sample = zeros(Int64, 2)

    edge_dim = size(mpo.tensors[mpo.Lx].data, 3)
    right_edge = zeros(T, 1, edge_dim, edge_dim)
    right_samp = rand(1:edge_dim)
    right_edge[1,right_samp,right_samp] = 1
    boundary_sample[2] = right_samp

    left_edges = Array{Array{T,3},1}()
    push!(left_edges, ones(T,1,1,1))
    for i in 1:mpo.Lx
        push!(left_edges, @tensor ten[-1,-2,-3] := left_edges[i][1,2,3]*mpo.tensors[i].data[2,4,-2,5]*conj(mpo.tensors[i].data)[3,4,-3,6]*marginal[i][1,5,-1,6])
    end
    norm_con = left_edges[end][1,1,1]
    cond_ind = mpo.Lx+1
    for i in mpo.Lx:-1:1
        if sz_zero && num_up == num_max
            samp = 1
        elseif sz_zero && num_down == num_max
            samp = 2
        else
            @tensor begin
                rho[-1,-2] := left_edges[i][1,2,3]*(mpo.tensors[i].data)[2,-1,7,4]*conj(mpo.tensors[i].data)[3,-2,8,5]*marginal[i][1,4,6,5]*right_edge[6,7,8]
            end
            rho /= tr(rho)
            samp = sample(1:mpo.in_dim, Weights(diag(real(rho))))
            if samp == 1
                num_down += 1
            else
                num_up += 1
            end
        end
        row_sample[i] = samp
        if sz_zero && (num_up == num_max || num_down == num_max)
            if i == mpo.Lx
	            right_edge = right_edge #*norm
            else
		        right_edge = right_edge
	        end
        else
            cond_ind -= 1
            @tensor begin
                right_edge[-1,-2,-3] := mpo.tensors[i].data[:,samp,:,:][-2,2,4]*conj(mpo.tensors[i].data)[:,samp,:,:][-3,3,5]*right_edge[1,2,3]*marginal[i][-1,4,1,5]
            end
        end
        left_dim, top_dim, right_dim, bot_dim = size(mpo.tensors[i].data)
        mpo.tensors[i].data = reshape(mpo.tensors[i].data[:,samp,:,:], (left_dim, 1, right_dim, bot_dim))
    end

    left_samp = 1
    boundary_sample[1] = left_samp

    mpo.tensors[1].data = reshape(mpo.tensors[1].data[left_samp,:,:,:], (1, 1, size(mpo.tensors[1].data)[3], size(mpo.tensors[1].data)[4]))
    mpo.tensors[mpo.Lx].data = reshape(mpo.tensors[mpo.Lx].data[:,:,right_samp,:], (size(mpo.tensors[mpo.Lx].data)[1],1,1,size(mpo.tensors[mpo.Lx].data)[4]))

    @tensor unnormalised_prob[] := left_edges[cond_ind][1,2,3]*right_edge[1,2,3]
    prob_con = (unnormalised_prob[1]/norm_con) ^ (1/(2*mpo.Lx))
    for i in 1:mpo.Lx
        mpo.tensors[i].data /= prob_con
    end
    mpo.in_dim = 1

    return row_sample, boundary_sample, num_up, num_down
end


function get_sample(peps::PEPS{T}, mps::MPO{T}, num_rows::Int64, max_dim::Int64, sz_zero::Bool, marginals::Array{Array{Array{T,4},1},1}) where T
    edge_dim = peps.edge_dim
    boundary_samples_vert = zeros(Int64, (2,num_rows))
    boundary_samples_hor = zeros(Int64, (peps.Lx))

    sample_peps = zeros(Int64, (num_rows,peps.Lx))
    num_up = 0
    num_down = 0
    if peps.Lx*peps.Ly % 2 == 0
        num_max = Int64(peps.Lx*peps.Ly/2)
    elseif sz_zero == true
        @warn "number of degrees of freedom is not divisible by two: switching off sz_zero constraint" maxlog=1
        num_max = peps.Lx*peps.Ly
        sz_zero=false
    else
        num_max = peps.Lx*peps.Ly
    end

    for i in 1:num_rows
        multiply_mps_pepsrow!(mps, peps, i)
        if max_dim >= 1
            norm = truncate_boundary_mpo_svd!(mps, max_dim)
        else
            norm = canonicalize_boundary_mpo_left!(mps)
        end
        row_sample, boundary_sample, num_up, num_down = perfect_sample_boundary!(mps, norm, sz_zero, num_up, num_down, num_max, marginals[i])
        sample_peps[i,:] = row_sample
        boundary_samples_vert[:,i] = boundary_sample[:]
    end

    if num_rows == peps.Ly
        boundary_samples_hor = rand(1:edge_dim, peps.Lx)
        mpo = construct_boundary_mps(2, edge_dim, boundary_samples_hor, (1/sqrt(edge_dim))*ones(Float64, peps.Lx), T)
        multiply_mps_mpo!(mps, mpo)
    end
    return sample_peps, boundary_samples_vert, boundary_samples_hor, mps
end


function get_samples!(peps::PEPS{T}, psi_sqs::Array{T,2}, samples::Array{Int64,4}, sample_cut::Int64, sz_zero::Bool, flip_symm::Bool, marginals::Array{Array{Array{T,4},1},1}) where T
    num_threads = Threads.nthreads()
    samples_per_thread = size(psi_sqs,2)

    @Threads.threads for i in 1:num_threads
        for j in 1:samples_per_thread
            edge_dim = peps.edge_dim
            boundary_samples_hor = rand(1:edge_dim, peps.Lx)
            mps = construct_boundary_mps(4, edge_dim, boundary_samples_hor, (1/sqrt(edge_dim))*ones(peps.Lx), T)
            sample_peps, boundary_samples_vert, boundary_samples_hor, mps = get_sample(peps, mps, peps.Ly, sample_cut, sz_zero, marginals)
            norm_peps = mps_overlap(mps,mps)
            psi_sqs[i,j] = norm_peps
            samples[i,j,:,:] = sample_peps
        end
    end
end


function get_samples_wolff!(peps::PEPS{T}, psi_sqs::Array{T,2}, samples::Array{Int64,4}, burn_in_samples::Int64, begin_samples::Array{Int64,3}, beta::Float64) where T
    mean_cluster_size = zeros(Float64, size(psi_sqs, 1))
    @Threads.threads for i in 1:size(psi_sqs,1)
        state = begin_samples[i,:,:]
        for j in 1:burn_in_samples
            ind = CartesianIndex(rand(1:peps.Lx), rand(1:peps.Ly))
            cluster_size = transition_wolff!(peps, state, ind, beta)
        end
        mean_cluster_size[i] = 0
        for j in 1:size(samples,2)
            ind = CartesianIndex(rand(1:peps.Lx), rand(1:peps.Ly))
            cluster_size = transition_wolff!(peps, state, ind, beta)
            psi_sqs[i,j] = 1
            samples[i,j,:,:] = state
            mean_cluster_size[i] += cluster_size/(peps.Lx*peps.Ly*size(psi_sqs, 1)*size(psi_sqs, 2))
        end
    end
    return sum(mean_cluster_size)
end


function get_samples_mcmc!(peps::PEPS{T}, psi_sqs::Array{T,2}, samples::Array{Int64,4}, sample_cut::Int64, burn_in_samples::Int64, props_between_samples::Int64, begin_samples::Array{Int64,3}, fast::Bool, transition) where T
    accept_rates = zeros(Float64, size(psi_sqs,1))
    if transition == transition_single_flip! && fast
        trans_string = "single_flip"
    elseif transition == transition_exchange! && fast
        trans_string = "exchange"
    elseif fast
        throw(ArgumentError("transition of markov chain not implemented with fast argument"))
    end

    @Threads.threads for i in 1:size(psi_sqs,1)
        state = begin_samples[i,:,:]
        for i in 1:burn_in_samples
            if !fast
                state, accept = get_sample_mcmc(peps, state, sample_cut, props_between_samples, transition)
            else
                state, accept = get_sample_mcmc_fast(peps, state, sample_cut, props_between_samples, trans_string)
            end
        end
        accept_rates[i] = 0
        for j in 1:size(psi_sqs,2)
            if !fast
                state, accept = get_sample_mcmc(peps, state, sample_cut, props_between_samples, transition)
            else
                state, accept = get_sample_mcmc_fast(peps, state, sample_cut, props_between_samples, trans_string)
            end
            psi_sqs[i,j] = 1
            samples[i,j,:,:] = state
            accept_rates[i] += accept/(size(psi_sqs, 1)*size(psi_sqs, 2))
        end
    end
    return sum(accept_rates)
end


function get_samples_mcmc_direct!(peps::PEPS{T}, psi_sqs::Array{T,2}, norms::Array{T,2}, samples::Array{Int64,4}, sample_cut::Int64, burn_in_samples::Int64, props_between_samples::Int64, sz_zero::Bool, marginals::Array{Array{Array{T,4},1},1}) where T
    num_threads = Threads.nthreads()
    samples_per_thread = size(psi_sqs,2)

    edge_dim = peps.edge_dim
    boundary_samples_hor = rand(1:edge_dim, peps.Lx)
    mps = construct_boundary_mps(4, edge_dim, boundary_samples_hor, (1/sqrt(edge_dim))*ones(peps.Lx), T)
    sample_peps, boundary_samples_vert, boundary_samples_hor, mps = get_sample(peps, mps, peps.Ly, sample_cut, sz_zero, marginals)
    norm_peps = mps_overlap(mps,mps)
    for i in 1:burn_in_samples
        for j in 1:props_between_samples
            sample_peps, norm_peps, accept = get_sample_mcmc_direct(peps, sample_peps, norm_peps, sample_cut, sz_zero,marginals)
        end
    end
    begin_sample = sample_peps
    begin_norm = norm_peps
    accept_rate = 0
    @Threads.threads for i in 1:num_threads
        for j in 1:samples_per_thread
            for k in 1:props_between_samples
	            if j > 1
                    sample_peps, norm_peps, accept = get_sample_mcmc_direct(peps, sample_peps, norm_peps, sample_cut, sz_zero, marginals)
                else
                    sample_peps, norm_peps, accept = get_sample_mcmc_direct(peps, begin_sample, begin_norm, sample_cut, sz_zero, marginals)
                end
                norms[i,j] = norm_peps
                samples[i,j,:,:] = sample_peps
                accept_rate += accept/(num_threads*samples_per_thread*props_between_samples)
            end
            psi_sqs[i,j] = 1
        end
    end
    return accept_rate
end
