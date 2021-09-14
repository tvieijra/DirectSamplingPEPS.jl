function get_local_energy(peps::PEPS{T}, sample::Array{Int64,2}, boundary_sample_vert::Array{Int64,2}, boundary_sample_hor::Array{Int64,2}, contract_cut::Int64, rot_symm::Bool, hamiltonian) where T
    conn_samples, mels, non_zero_conns, non_zero_mels = hamiltonian(sample)
    conn_psis = zeros(T, size(mels))

    sampled_peps = [peps.tensors[(i,j)].data[sample[i,j],:,:,:,:] for i in 1:peps.Ly, j in 1:peps.Lx]
    psi = contract_sampled_single_layer_svd(sampled_peps, true, contract_cut)
    for i in 1:size(mels, 1)
        if mels[i] != 0
            sampled_peps = [peps.tensors[(j,k)].data[conn_samples[i,j,k],:,:,:,:] for j in 1:peps.Ly, k in 1:peps.Lx]
            conn_psis[i] = contract_sampled_single_layer_svd(sampled_peps, true, contract_cut)
        end
    end
    conn_psis /= psi
    loc_energy = dot(conn_psis, mels)
    return loc_energy
end


function get_fast_local_energy(peps::PEPS{T}, sample::Array{Int64,2}, contract_cut::Int64, rot_symm::Bool, hamiltonian) where T
    conn_samples, mels, non_zero_conn_inds, non_zero_mels = hamiltonian(sample)
    loc_energy = mels[1]
    sample_data = [peps.tensors[(i,j)].data[sample[i,j],:,:,:,:] for i in 1:peps.Ly, j in 1:peps.Lx]

    top_mps_boundaries, down_mps_boundaries = get_all_horizontal_boundaries(sample_data, contract_cut)

    for i in 1:peps.Ly
        three_sb_left = get_threeind_dense_vertical_boundaries(top_mps_boundaries[i], sample_data[i,:], down_mps_boundaries[i])
        if i != peps.Ly
            four_sb_left = get_fourind_dense_vertical_boundaries(top_mps_boundaries[i], sample_data[i,:], sample_data[i+1,:], down_mps_boundaries[i+1])
        end
        three_sb_right = Array{Array{T,3},1}()
        four_sb_right = Array{Array{T,4},1}()
        push!(three_sb_right, ones(T, 1, 1, 1))
        push!(four_sb_right, ones(T, 1, 1, 1, 1))

        pbc_nn = false
        pbc_nnn = false
        pbc_three = ones(T, 1, 1, 1)
        pbc_four = ones(T, 1, 1, 1, 1)
        mel_pbc_nn = 0
        mel_pbc_nnn = 0
        for k in 1:size(non_zero_conn_inds[i,end],1)
            if non_zero_conn_inds[i,end][k] == (i,1)
                pbc_nn = true
                mel_pbc_nn = non_zero_mels[i,end][k]
            end
            if non_zero_conn_inds[i,end][k] == (i+1,1)
                pbc_nnn = true
                mel_pbc_nnn = non_zero_mels[i,end][k]
            end
        end
        for j in peps.Lx:-1:1
            mpo_three = [top_mps_boundaries[i][j], sample_data[i,j], down_mps_boundaries[i][j]]
            push!(three_sb_right, contract_boundary_three_right(three_sb_right[peps.Lx-j+1], mpo_three))
            if i != peps.Ly
                mpo_four = [top_mps_boundaries[i][j], sample_data[i,j], sample_data[i+1,j], down_mps_boundaries[i+1][j]]
                push!(four_sb_right, contract_boundary_four_right(four_sb_right[peps.Lx-j+1], mpo_four))
            end
            #PBC only
            if pbc_nn
                ind = sample[i,j]
                if j == peps.Lx || j == 1; ind = flip_spin(sample[i,j]) end
                mpo_three = [top_mps_boundaries[i][j], peps.tensors[i,j].data[ind,:,:,:,:], down_mps_boundaries[i][j]]
                pbc_three = contract_boundary_three_right(pbc_three, mpo_three)
                if j == 1
                    loc_energy += mel_pbc_nn*contract_three_leftright(ones(T,1,1,1), pbc_three)/contract_three_leftright(ones(T,1,1,1), three_sb_right[end])
                end
            end
            if pbc_nnn
                ind1 = sample[i,j]
                ind2 = sample[i+1,j]
                if j == peps.Lx; ind1 = flip_spin(sample[i,j]) end
                if j == 1; ind2 = flip_spin(sample[i+1,j]) end
                mpo_four = [top_mps_boundaries[i][j], peps.tensors[i,j].data[ind1,:,:,:,:], peps.tensors[i,j].data[ind2,:,:,:,:], down_mps_boundaries[i+1][j]]
                pbc_four = contract_boundary_four_right(pbc_four, mpo_four)
                if j == 1
                    loc_energy += mel_pbc_nnn*contract_four_leftright(ones(T,1,1,1,1), pbc_four)/contract_four_leftright(ones(T,1,1,1,1), four_sb_right[end])
                end
            end
            for k in 1:size(non_zero_conn_inds[i,j],1)
                if non_zero_conn_inds[i,j][k][1] == i+1 && non_zero_conn_inds[i,j][k][2] == j
                    mpo_four = [top_mps_boundaries[i][j], peps.tensors[i,j].data[flip_spin(sample[i,j]),:,:,:,:], peps.tensors[i+1,j].data[flip_spin(sample[i+1,j]),:,:,:,:], down_mps_boundaries[i+1][j]]
                    loc_energy += non_zero_mels[i,j][k]*contract_four_leftright(four_sb_left[j], contract_boundary_four_right(four_sb_right[peps.Lx-j+1], mpo_four))/contract_four_leftright(four_sb_left[j], four_sb_right[peps.Lx-j+2])
                end
                if non_zero_conn_inds[i,j][k][1] == i && non_zero_conn_inds[i,j][k][2] == j+1
                    mpo_three_l = [top_mps_boundaries[i][j], peps.tensors[i,j].data[flip_spin(sample[i,j]),:,:,:,:], down_mps_boundaries[i][j]]
                    mpo_three_r = [top_mps_boundaries[i][j+1], peps.tensors[i,j+1].data[flip_spin(sample[i,j+1]),:,:,:,:], down_mps_boundaries[i][j+1]]
                    loc_energy += non_zero_mels[i,j][k]*contract_three_leftright(contract_boundary_three_left(three_sb_left[j], mpo_three_l), contract_boundary_three_right(three_sb_right[peps.Lx-j], mpo_three_r))/contract_three_leftright(three_sb_left[j+1], three_sb_right[peps.Lx-j+1])
                end
                if non_zero_conn_inds[i,j][k][1] == i+1 && non_zero_conn_inds[i,j][k][2] == j+1
                    mpo_four_l = [top_mps_boundaries[i][j], peps.tensors[i,j].data[flip_spin(sample[i,j]),:,:,:,:], peps.tensors[i+1,j].data[sample[i+1,j],:,:,:,:], down_mps_boundaries[i+1][j]]
                    mpo_four_r = [top_mps_boundaries[i][j+1], peps.tensors[i,j+1].data[sample[i,j+1],:,:,:,:], peps.tensors[i+1,j+1].data[flip_spin(sample[i+1,j+1]),:,:,:,:], down_mps_boundaries[i+1][j+1]]
                    loc_energy += non_zero_mels[i,j][k]*contract_four_leftright(contract_boundary_four_left(four_sb_left[j], mpo_four_l), contract_boundary_four_right(four_sb_right[peps.Lx-j], mpo_four_r))/contract_four_leftright(four_sb_left[j+1], four_sb_right[peps.Lx-j+1])
                end
                if non_zero_conn_inds[i,j][k][1] == i+1 && non_zero_conn_inds[i,j][k][2] == j-1
                    mpo_four_l = [top_mps_boundaries[i][j-1], peps.tensors[i,j-1].data[sample[i,j-1],:,:,:,:], peps.tensors[i+1,j-1].data[flip_spin(sample[i+1,j-1]),:,:,:,:], down_mps_boundaries[i+1][j-1]]
                    mpo_four_r = [top_mps_boundaries[i][j], peps.tensors[i,j].data[flip_spin(sample[i,j]),:,:,:,:], peps.tensors[i+1,j].data[sample[i+1,j],:,:,:,:], down_mps_boundaries[i+1][j]]
                    loc_energy += non_zero_mels[i,j][k]*contract_four_leftright(contract_boundary_four_left(four_sb_left[j-1], mpo_four_l), contract_boundary_four_right(four_sb_right[peps.Lx-j+1], mpo_four_r))/contract_four_leftright(four_sb_left[j], four_sb_right[peps.Lx-j+2])
                end
                if non_zero_conn_inds[i,j][k] == (i,j)
                    mpo_three = [top_mps_boundaries[i][j], peps.tensors[i,j].data[flip_spin(sample[i,j]),:,:,:,:], down_mps_boundaries[i][j]]
                    loc_energy += non_zero_mels[i,j][k]*contract_three_leftright(three_sb_left[j], contract_boundary_three_right(three_sb_right[peps.Lx-j+1], mpo_three))/contract_three_leftright(three_sb_left[j+1], three_sb_right[peps.Lx-j+1])
                end
            end
        end
    end
    return loc_energy
end


function get_energy!(peps::PEPS{T}, hamiltonian, psi_sqs::Array{T,2}, samples::Array{Int64,4}, elocs::Array{T,2}, contract_cut::Int64, rot_symm::Bool, fast::Bool) where T
    num_threads = Threads.nthreads()
    samples_per_thread = size(psi_sqs,2)

    @Threads.threads for i in 1:num_threads
        for j in 1:samples_per_thread
            if fast
                elocs[i,j] = get_fast_local_energy(peps, samples[i,j,:,:], contract_cut, rot_symm, hamiltonian)

            else
                elocs[i,j] = get_local_energy(peps, samples[i,j,:,:], zeros(Int64,2,peps.Lx), zeros(Int64,2,peps.Ly), contract_cut, rot_symm, hamiltonian)
            end
        end
    end
    norm_peps = sum(psi_sqs)
    psi_times_loc_energies = psi_sqs .* elocs
    energy = sum(psi_times_loc_energies)/norm_peps
    elocs .-= energy
    return real(energy), norm_peps
end


function get_heisenberg_mels(sample::Array{Int64,2}, J::Number, sign_rule::Bool, pbc::Bool)
    if sign_rule
        sign = -1
    else
        sign = 1
    end

    sample = (sample .- 1)*2 .- 1
    Lx, Ly = size(sample)

    if J == 0
        conn_samples = reshape(sample, (1, size(sample)...))
    elseif !pbc
        conn_samples = permutedims(repeat(sample, outer = [1, 1, 1+2*(Lx-1)*Ly]), (3,1,2))
    else
        conn_samples = permutedims(repeat(sample, outer = [1, 1, 1+2*(Lx-1)*Ly + Ly]), (3,1,2))
    end
    mels = zeros(Complex{Float64}, size(conn_samples, 1))
    non_zero_conns = [Tuple[] for i=1:size(sample,1), j=1:size(sample,2)]
    non_zero_mels = [Float64[] for i=1:size(sample,1), j=1:size(sample,2)]


    #szsz part
    shift_up = sample[2:end,:]
    shift_down = sample[1:end-1,:]
    shift_left = sample[:,2:end]
    shift_right = sample[:,1:end-1]

    energy = sum(shift_up .* shift_down) + sum(shift_left .* shift_right)
    if pbc
        energy += sum(sample[:,1] .* sample[:,end])
    end
    mels[1] = energy

    if J != 0
        index = 2
        for i in 1:Ly
            for j in 1:Lx
                if j != Lx
                    mels[index] = sign*abs(sample[i,j] - sample[i,j+1])
                    conn_samples[index,i,j] *= -1
                    conn_samples[index,i,j+1] *= -1
                    if mels[index] != 0
                        push!(non_zero_conns[i,j], (i,j+1))
                        push!(non_zero_mels[i,j], mels[index])
                    end
                    index += 1
                end
                if i != Ly
                    mels[index] = sign*abs(sample[i,j] - sample[i+1,j])
                    conn_samples[index,i,j] *= -1
                    conn_samples[index,i+1,j] *= -1
                    if mels[index] != 0
                        push!(non_zero_conns[i,j], (i+1,j))
                        push!(non_zero_mels[i,j], mels[index])
                    end
                    index += 1
                end
                if pbc && j == Lx
                    mels[index] = sign*abs(sample[i,1] - sample[i,end])
                    conn_samples[index,i,1] *= -1
                    conn_samples[index,i,end] *= -1
                    if mels[index] != 0
                        push!(non_zero_conns[i,j], (i,1))
                        push!(non_zero_mels[i,j], mels[index])
                    end
                    index += 1
                end
            end
        end
    end
    return (conn_samples .+ 1).÷2 .+1, mels, non_zero_conns, non_zero_mels
end


function get_J1J2_mels(sample::Array{Int64,2}, J2::Number, sign_rule::Bool, triangular::Bool)
    if sign_rule
        sign = -1
    else
        sign = 1
    end

    sample = (sample .- 1)*2 .- 1
    Lx, Ly = size(sample)

    if triangular
        conn_samples = permutedims(repeat(sample, outer = [1, 1, 1 + 2*(Lx-1)*Ly + (Lx-1)*(Ly-1)]), (3,1,2))
    else
        conn_samples = permutedims(repeat(sample, outer = [1, 1, 1 + 2*(Lx-1)*Ly + 2*(Lx-1)*(Ly-1)]), (3,1,2))
    end
    mels = zeros(Complex{Float64}, size(conn_samples, 1))
    non_zero_conns = [Tuple[] for i=1:size(sample,1), j=1:size(sample,2)]
    non_zero_mels = [Float64[] for i=1:size(sample,1), j=1:size(sample,2)]


    #szsz part
    shift_up = sample[2:end,:]
    shift_down = sample[1:end-1,:]
    shift_left = sample[:,2:end]
    shift_right = sample[:,1:end-1]

    energy = sum(shift_up .* shift_down) + sum(shift_left .* shift_right)

    shift_upleft = sample[2:end,2:end]
    shift_downright = sample[1:end-1,1:end-1]
    energy += J2*sum(shift_upleft .* shift_downright)
    if !triangular
        shift_upright = sample[2:end,1:end-1]
        shift_downleft = sample[1:end-1,2:end]
        energy += J2*sum(shift_upright .* shift_downleft)
    end
    mels[1] = energy


    index = 2
    for i in 1:Ly
        for j in 1:Lx
            if j != Lx
                mels[index] = sign*abs(sample[i,j] - sample[i,j+1])
                conn_samples[index,i,j] *= -1
                conn_samples[index,i,j+1] *= -1
                if mels[index] != 0
                    push!(non_zero_conns[i,j], (i,j+1))
                    push!(non_zero_mels[i,j], mels[index])
                end
                index += 1
                if i != Ly
                    mels[index] = J2*sign*abs(sample[i,j] - sample[i+1,j+1])
                    conn_samples[index,i,j] *= -1
                    conn_samples[index,i+1,j+1] *= -1
                    if mels[index] != 0
                        push!(non_zero_conns[i,j], (i+1,j+1))
                        push!(non_zero_mels[i,j], mels[index])
                    end
                    index += 1
                end
            end
            if i != Ly
                mels[index] = sign*abs(sample[i,j] - sample[i+1,j])
                conn_samples[index,i,j] *= -1
                conn_samples[index,i+1,j] *= -1
                if mels[index] != 0
                    push!(non_zero_conns[i,j], (i+1,j))
                    push!(non_zero_mels[i,j], mels[index])
                end
                index += 1
                if !triangular && j != 1
                    mels[index] = J2*sign*abs(sample[i,j] - sample[i+1,j-1])
                    conn_samples[index,i,j] *= -1
                    conn_samples[index,i+1,j-1] *= -1
                    if mels[index] != 0
                        push!(non_zero_conns[i,j], (i+1,j-1))
                        push!(non_zero_mels[i,j], mels[index])
                    end
                    index += 1
                end
            end
        end
    end
    return (conn_samples .+ 1).÷2 .+1, mels, non_zero_conns, non_zero_mels
end


function get_ising_mels(sample::Array{Int64,2}, g::Number, pbc::Bool)
    sample = (2*sample .- 3)
    Lx, Ly = size(sample)

    conn_samples = permutedims(repeat(sample, outer = [1, 1, Lx*Ly+1]), (3,1,2))

    mels = zeros(Complex{Float64}, size(conn_samples, 1))
    non_zero_conns = [Tuple[] for i=1:size(sample,1), j=1:size(sample,2)]
    non_zero_mels = [Float64[] for i=1:size(sample,1), j=1:size(sample,2)]


    #szsz part
    shift_up = sample[2:end,:]
    shift_down = sample[1:end-1,:]
    shift_left = sample[:,2:end]
    shift_right = sample[:,1:end-1]

    energy = - sum(shift_up .* shift_down) - sum(shift_left .* shift_right)
    if pbc
        energy -= sum(sample[:,1] .* sample[:,end])
    end
    mels[1] = energy

    #off-diagonal part
    index = 2
    for i in 1:Ly
        for j in 1:Lx
            mels[index] = g
            conn_samples[index,i,j] = -conn_samples[index,i,j]
            push!(non_zero_conns[i,j], (i,j))
            push!(non_zero_mels[i,j], mels[index])
            index += 1
        end
    end
    return Int64.((conn_samples .+ 3)/2), mels, non_zero_conns, non_zero_mels
end


function get_sx_op(sample::Array{Int64,2}, posx::Int64, posy::Int64)
    sample = (2*sample .- 3)
    Lx, Ly = size(sample)

    conn_samples = permutedims(repeat(sample, outer = [1, 1, 1]), (3,1,2))

    mels = zeros(Complex{Float64}, size(conn_samples, 1))
    non_zero_conns = [Tuple[] for i=1:size(sample,1), j=1:size(sample,2)]
    non_zero_mels = [Float64[] for i=1:size(sample,1), j=1:size(sample,2)]

    #off-diagonal part
    mels[1] = 1
    conn_samples[1,posy,posx] = -conn_samples[1,posy,posx]
    push!(non_zero_conns[posy,posx], (posy,posx))
    push!(non_zero_mels[posy,posx], mels[1])
    return Int64.((conn_samples .+ 3)/2), mels, non_zero_conns, non_zero_mels
end
