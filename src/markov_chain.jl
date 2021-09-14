function transition_exchange!(peps::PEPS{T}, state::Array{Int64,2}, psi::T, ind1::CartesianIndex, ind2::CartesianIndex, sample_cut::Int64) where T
    temp = state[ind2]
    state[ind2] = state[ind1]
    state[ind1] = temp

    sample_data = [peps.tensors[(i,j)].data[state[i,j],:,:,:,:] for i in 1:peps.Ly, j in 1:peps.Lx]
    psi_new_state = contract_sampled_single_layer_svd_conv(sample_data, false, sample_cut, false)
    prob = min(1, abs2(psi_new_state/psi))
    if rand(Float64) > prob
        temp = state[ind2]
        state[ind2] = state[ind1]
        state[ind1] = temp
        accept = 0
    else
        psi = psi_new_state
        accept = 1
    end
    return psi, accept
end


function transition_single_flip!(peps::PEPS{T}, state::Array{Int64,2}, psi::T, ind1::CartesianIndex, ind2::CartesianIndex, sample_cut::Int64) where T
    state[ind1] = state[ind1]%2 + 1

    sample_data = [peps.tensors[(i,j)].data[state[i,j],:,:,:,:] for i in 1:peps.Ly, j in 1:peps.Lx]
    psi_new_state = contract_sampled_single_layer_svd_conv(sample_data, false, sample_cut, false)
    prob = min(1, abs2(psi_new_state)/abs2(psi))
    if rand(Float64) > prob
        state[ind1] = state[ind1]%2 + 1
        accept = 0
    else
        psi = psi_new_state
        accept = 1
    end
    return psi, accept
end


function transition_energydiff!(peps::PEPS{T}, state::Array{Int64,2}, psi::T, ind1::CartesianIndex, ind2::CartesianIndex, sample_cut::Int64, beta::Float64, energy_diff) where T
    energy_diff = energy_diff(state, ind1)
    prob = min(1, exp(-beta*energy_diff))
    if rand(Float64) > prob
        accept = 0
    else
        psi = psi*exp(-beta*energy_diff)
        state[ind1] = state[ind1]%2 + 1
        accept = 1
    end
    return psi, accept
end


function energy_diff_ising(state1::Array{Int64,2}, ind1::CartesianIndex)
    energy_diff = 0
    if ind1[1] > 1; energy_diff += 2*(2*state1[ind1]-3)*(2*state1[ind1[1]-1, ind1[2]]-3) end
    if ind1[1] < size(state1,1); energy_diff += 2*(2*state1[ind1]-3)*(2*state1[ind1[1]+1, ind1[2]]-3) end
    if ind1[2] > 1; energy_diff += 2*(2*state1[ind1]-3)*(2*state1[ind1[1], ind1[2]-1]-3) end
    if ind1[2] < size(state1,2); energy_diff += 2*(2*state1[ind1]-3)*(2*state1[ind1[1], ind1[2]+1]-3) end
    return energy_diff
end


function transition_wolff!(peps::PEPS{T}, state::Array{Int64,2}, ind::CartesianIndex, beta::Float64) where T
    cluster = Array{CartesianIndex, 1}()
    check_list = Array{CartesianIndex, 1}()

    push!(cluster, ind)
    push!(check_list, ind)
    while !isempty(check_list)
        ch_spin = pop!(check_list)
        neighs = get_neighbours!(ch_spin, peps.Lx, peps.Ly)
        for i in 1:size(neighs,1)
            if !(neighs[i] in cluster) && state[ch_spin] == state[neighs[i]] && rand(Float64) < (1 - exp(-2*beta))
                push!(cluster, neighs[i])
                push!(check_list, neighs[i])
            end
        end
    end
    for i in 1:size(cluster, 1)
        state[cluster[i]] = state[cluster[i]]%2 + 1
    end
    return size(cluster, 1)
end


function get_neighbours!(ind::CartesianIndex, Lx::Int64, Ly::Int64)
    neigh_list = Array{CartesianIndex,1}()
    up = CartesianIndex(ind[1],ind[2]-1)
    down = CartesianIndex(ind[1],ind[2]+1)
    left = CartesianIndex(ind[1]-1,ind[2])
    right = CartesianIndex(ind[1]+1,ind[2])
    if ind[1] > 1; push!(neigh_list, left) end
    if ind[1] < Lx; push!(neigh_list, right) end
    if ind[2] > 1; push!(neigh_list, up) end
    if ind[2] < Ly; push!(neigh_list, down) end
    return neigh_list
end


function get_sample_mcmc_direct(peps::PEPS{T}, state_old::Array{Int64,2}, psi_old::T, sample_cut::Int64, sz_zero::Bool, marginals::Array{Array{Array{T,4},1},1}) where T
    edge_dim = peps.edge_dim
    boundary_samples_hor = rand(1:edge_dim, peps.Lx)
    mps = construct_boundary_mps(4, edge_dim, boundary_samples_hor, (1/sqrt(edge_dim))*ones(peps.Lx), T)
    state_new, boundary_samples_vert, boundary_samples_hor, mps = get_sample(peps, mps, peps.Ly, sample_cut, sz_zero, marginals)
    psi_new = mps_overlap(mps,mps)
    prob = min(1, real(psi_new/psi_old))
    if rand(Float64) > prob
        return state_old, psi_old, 0
    else
        return state_new, psi_new, 1
    end
end


function get_sample_mcmc(peps::PEPS{T}, state::Array{Int64,2}, sample_cut::Int64, sweeps_between_samples::Int64, transition) where T
    edge_dim = peps.edge_dim
    if edge_dim == 1
        boundary_samples_vert = ones(Int64, (2,peps.Ly))
        boundary_samples_hor = ones(Int64, (2,peps.Lx))
    else
        throw(DomainError(edge_dim, "markov chain only implemented for edge_dim=1"))
    end

    sample_data = [peps.tensors[(i,j)].data[state[i,j],:,:,:,:] for i in 1:peps.Ly, j in 1:peps.Lx]
    psi = contract_sampled_single_layer_svd_conv(sample_data, false, sample_cut, false)

    accept_rate = 0
    for i in 1:sweeps_between_samples
        for x in 1:peps.Lx
            for y in 1:peps.Ly
                ind1 = CartesianIndex(y,x)

                direction = rand([1,2])
	            if direction == 1
                    ind2 = CartesianIndex((ind1[1]+rand([-1,1])-1+peps.Ly)%peps.Ly+1, ind1[2])
                else
                    ind2 = CartesianIndex(ind1[1], (ind1[2]+rand([-1,1])-1+peps.Lx)%peps.Lx+1)
                end
                psi, accept = transition(peps, state, psi, ind1, ind2, sample_cut)
                accept_rate += accept/(sweeps_between_samples*peps.Lx*peps.Ly)
            end
        end
    end
    return state, accept_rate
end


function get_sample_mcmc_fast(peps::PEPS{T}, state::Array{Int64,2}, sample_cut::Int64, sweeps_between_samples::Int64, transition::String) where T
    sample_data = [peps.tensors[(i,j)].data[state[i,j],:,:,:,:] for i in 1:peps.Ly, j in 1:peps.Lx]
    accept_rate = 0
    for i in 1:sweeps_between_samples
        top_mps_boundaries, down_mps_boundaries = get_all_horizontal_boundaries(sample_data, sample_cut)
        push!(down_mps_boundaries, [ones(T, 1,1,1,1) for x in 1:peps.Lx])
        for y in 1:peps.Ly
            mpo1 = sample_data[y,:]
            if y == peps.Ly
                mpo2 = [ones(T, 1,1,1,1) for x in 1:peps.Lx]
            else
                mpo2 = sample_data[y+1,:]
            end
            four_sb_left = get_fourind_dense_vertical_boundaries(top_mps_boundaries[y], mpo1, mpo2, down_mps_boundaries[y+1])

            four_sb_right = Array{Array{T,4},1}()
            push!(four_sb_right, ones(T, 1, 1, 1, 1))
            psi = contract_four_leftright(four_sb_left[end],four_sb_right[1])
            for x in peps.Lx:-1:1
                if x > 1 && y < peps.Ly
                    rel_inds = rand([(1,0), (0,-1)])
                elseif x == 1 && y < peps.Ly
                    rel_inds = (1,0)
                elseif x > 1 && y == peps.Ly
                    rel_inds = (0,-1)
                else
                    continue
                end
                if state[y,x] != state[y+rel_inds[1],x+rel_inds[2]] && transition == "exchange"
                    sample_data[y,x] = peps.tensors[(y,x)].data[flip_spin(state[y,x]),:,:,:,:]
                    sample_data[y+rel_inds[1],x+rel_inds[2]] = peps.tensors[(y+rel_inds[1],x+rel_inds[2])].data[flip_spin(state[y+rel_inds[1],x+rel_inds[2]]),:,:,:,:]
                    if y != peps.Ly
                        if rel_inds[2] == -1
                            mpo_four1 = [top_mps_boundaries[y][x-1], sample_data[y,x-1], sample_data[y+1,x-1], down_mps_boundaries[y+1][x-1]]
                        end
                        mpo_four2 = [top_mps_boundaries[y][x], sample_data[y,x], sample_data[y+1,x], down_mps_boundaries[y+1][x]]
                    elseif y == peps.Ly
                        if rel_inds[2] == -1
                            mpo_four1 = [top_mps_boundaries[y][x-1], sample_data[y,x-1], ones(T, 1,1,1,1), down_mps_boundaries[y+1][x-1]]
                        end
                        mpo_four2 = [top_mps_boundaries[y][x], sample_data[y,x], ones(T, 1,1,1,1), down_mps_boundaries[y+1][x]]
                    end
                    new_right = contract_boundary_four_right(four_sb_right[peps.Lx-x+1], mpo_four2)
                    if rel_inds[2] == -1
                        new_left = contract_boundary_four_left(four_sb_left[x-1], mpo_four1)
                    else
                        new_left = four_sb_left[x]
                    end
                    psi_new = contract_four_leftright(new_left, new_right)
                    if rand(Float64) < min(1, psi_new^2/psi^2)
                        psi = psi_new
                        state[y,x] = flip_spin(state[y,x])
                        state[y+rel_inds[1],x+rel_inds[2]] = flip_spin(state[y+rel_inds[1], x+rel_inds[2]])
                        push!(four_sb_right, new_right)
                        accept_rate += 1/(sweeps_between_samples*peps.Lx*peps.Ly)
                    else
                        sample_data[y,x] = peps.tensors[(y,x)].data[state[y,x],:,:,:,:]
                        sample_data[y+rel_inds[1],x+rel_inds[2]] = peps.tensors[(y+rel_inds[1],x+rel_inds[2])].data[state[y+rel_inds[1],x+rel_inds[2]],:,:,:,:]
                        if y != peps.Lx
                            mpo_four2 = [top_mps_boundaries[y][x], sample_data[y,x], sample_data[y+1,x], down_mps_boundaries[y+1][x]]
                        else
                            mpo_four2 = [top_mps_boundaries[y][x], sample_data[y,x], ones(T, 1,1,1,1), down_mps_boundaries[y+1][x]]
                        end
                        new_right = contract_boundary_four_right(four_sb_right[peps.Lx-x+1], mpo_four2)
                        push!(four_sb_right, new_right)
                    end
                elseif transition == "single_flip"
                    sample_data[y,x] = peps.tensors[(y,x)].data[flip_spin(state[y,x]),:,:,:,:]
                    if y != peps.Ly
                        mpo_four = [top_mps_boundaries[y][x], sample_data[y,x], sample_data[y+1,x], down_mps_boundaries[y+1][x]]
                    elseif y == peps.Ly
                        mpo_four = [top_mps_boundaries[y][x], sample_data[y,x], ones(T, 1,1,1,1), down_mps_boundaries[y+1][x]]
                    end
                    new_right = contract_boundary_four_right(four_sb_right[peps.Lx-x+1], mpo_four)
                    new_left = four_sb_left[x]
                    psi_new = contract_four_leftright(new_left, new_right)
                    if rand(Float64) < min(1, psi_new^2/psi^2)
                        psi = psi_new
                        state[y,x] = flip_spin(state[y,x])
                        push!(four_sb_right, new_right)
                        accept_rate += 1/(sweeps_between_samples*peps.Lx*peps.Ly)
                    else
                        sample_data[y,x] = peps.tensors[(y,x)].data[state[y,x],:,:,:,:]
                        if y != peps.Lx
                            mpo_four = [top_mps_boundaries[y][x], sample_data[y,x], sample_data[y+1,x], down_mps_boundaries[y+1][x]]
                        else
                            mpo_four = [top_mps_boundaries[y][x], sample_data[y,x], ones(T, 1,1,1,1), down_mps_boundaries[y+1][x]]
                        end
                        new_right = contract_boundary_four_right(four_sb_right[peps.Lx-x+1], mpo_four)
                        push!(four_sb_right, new_right)
                    end
                else
                    if y != peps.Ly
                        mpo_four2 = [top_mps_boundaries[y][x], sample_data[y,x], sample_data[y+1,x], down_mps_boundaries[y+1][x]]
                    else
                        mpo_four2 = [top_mps_boundaries[y][x], sample_data[y,x], ones(T, 1,1,1,1), down_mps_boundaries[y+1][x]]
                    end
                    new_right = contract_boundary_four_right(four_sb_right[peps.Lx-x+1], mpo_four2)
                    push!(four_sb_right, new_right)
                end
            end
            if y != peps.Ly
                mpo = [peps.tensors[(y,x)].data[state[y,x],:,:,:,:] for x in 1:peps.Lx]
                mps_up = multiply_mps_mpo(top_mps_boundaries[y], mpo)
                if sample_cut >= 1
                    mps_up, norm_mps = truncate_boundary_mpo_svd(mps_up, sample_cut)
                end
                top_mps_boundaries[y+1] = mps_up
            end
        end
    end
    return state, accept_rate
end


function get_sample_wolff(peps::PEPS{T}, state::Array{Int64,2}, transition) where T
    accept_rate = 0
    for i in 1:sweeps_between_samples
        for x in 1:peps.Lx
            for y in 1:peps.Ly
                ind1 = CartesianIndex(x,y)

                direction = rand([1,2])
	            if direction == 1
                    ind2 = CartesianIndex((ind1[1]+rand([-1,1])-1+peps.Ly)%peps.Ly+1, ind1[2])
                else
                    ind2 = CartesianIndex(ind1[1], (ind1[2]+rand([-1,1])-1+peps.Lx)%peps.Lx+1)
                end
                psi, accept = transition(peps, state, psi, ind1, ind2, sample_cut)
                accept_rate += accept/(sweeps_between_samples*peps.Lx*peps.Ly)
            end
        end
    end
    return state, accept_rate
end
