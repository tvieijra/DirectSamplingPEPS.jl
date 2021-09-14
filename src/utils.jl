function mps_overlap(mps1::MPO, mps2::MPO)
    @tensor begin
        edge[i,j] := mps1.tensors[1].data[a,b,i,c]*conj(mps2.tensors[1].data)[a,b,j,c]
    end
    for i in 2:mps1.Lx
        @tensor begin
            edge[-1,-2] := mps1.tensors[i].data[1,2,-1,3]*edge[1,4]*conj(mps2.tensors[i].data)[4,2,-2,3]
        end
    end

    return edge[1,1]
end


function mps_overlap(mps1::Array{Array{T,4},1}, mps2::Array{Array{T,4},1}) where T
    edge = ones(T, 1, 1)
    for i in 1:size(mps1,1)
        @tensor begin
            edge[-1,-2] := mps1[i][1,2,-1,3]*edge[1,4]*conj(mps2[i])[4,3,-2,2]
        end
    end

    return edge[1,1]
end


function matrix_product_row(mps::Array{Array{T,4},1}) where T
    edge = ones(T, 1)
    for i in 1:size(mps,1)
        @tensor begin
            edge[-1] := edge[1]*dropdims(mps[i], dims=(2,4))[1,-1]
        end
    end
    return edge[1]
end


function mps_pepsrow_mps_overlap(mps1::MPO, pepsrow::Array{Tensor{T,5},2}, mps2::MPO) where T
    function multiply_tensors(btensor::Array{T,4}, ptensor::Array{T,5})
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o,p] := btensor[k,m,n,a]*ptensor[j,l,a,o,p]
        end
        phys_dim,_,_,top_dim,_,_,bot_dim = size(mps_mpo_tensor)
        new_left_dim = size(mps_mpo_tensor)[2]*size(mps_mpo_tensor)[3]
        new_right_dim = size(mps_mpo_tensor)[5]*size(mps_mpo_tensor)[6]
        return reshape(mps_mpo_tensor, (phys_dim, new_left_dim, top_dim, new_right_dim, bot_dim))
    end
    top = [multiply_tensors(mps1.tensors[i].data, pepsrow[1,i].data) for i in 1:size(pepsrow,2)]

    if size(pepsrow,1) == 1
        botbra = [permutedims(dropdims(multiply_tensors(mps2.tensors[i].data, permutedims(top[i], (1,2,5,4,3))), dims=5), (2,3,4,1)) for i in 1:size(pepsrow,2)]
        botket = [permutedims(botbra[i], (1,4,3,2)) for i in 1:size(pepsrow,2)]
        return mps_overlap(botbra, botket)
    else
        bot = [multiply_tensors(mps2.tensors[i].data, permutedims(pepsrow[2,i], (1,2,5,4,3))) for i in 1:size(pepsrow,2)]
    end
end


function contract_single_layer(peps::PEPS{T}, sample::Array{Int,2}, boundary_sample_vert::Array{Int,2}, boundary_sample_hor::Array{Int,2}, max_dim::Int64) where T
    edge_dim = peps.edge_dim

    mps = construct_boundary_mps(4, edge_dim, boundary_sample_hor[1,:], (1/sqrt(edge_dim))*ones(Float64, peps.Lx), T)
    mpo = MPO{T}(peps.Lx, peps.b_dim, peps.b_dim, peps.b_dim, Dict{Int, Tensor{T,4}}())

    for i in 1:peps.Ly
        for j in 1:peps.Lx
            mpo.tensors[j] = Tensor{T,4}(peps.tensors[(i,j)].data[sample[i,j],:,:,:,:])
        end
        contract_mpo_boundary!(mpo, boundary_sample_vert[1,i], boundary_sample_vert[2,i])
        multiply_mps_mpo!(mps, mpo)
        if max_dim >= 1
            _ = truncate_boundary_mpo_svd!(mps, max_dim)
        end
    end

    edge = ones(T, 1,1)
    for i in 1:mps.Lx
        samp = boundary_sample_hor[2,i]
        boundary_tensor = zeros(T, (edge_dim))
        boundary_tensor[samp] = 1*sqrt(edge_dim)
        @tensor begin
            edge[j,k] := edge[j,a]*dropdims(mps.tensors[i].data, dims=2)[a,k,b]*boundary_tensor[b]
        end
    end

    return edge[1,1]
end


function contract_sampled_single_layer_svd(peps_data::Array{Array{T,4},2}, return_real::Bool, max_dim::Int64) where T
    mps = [ones(T, 1, 1, 1, 1) for i in 1:size(peps_data)[2]]

    for i in 1:size(peps_data)[1]
        mpo = peps_data[i,:]
        mps = multiply_mps_mpo(mps, mpo)
        if max_dim >= 1
            mps, norm_mps = truncate_boundary_mpo_svd(mps, max_dim)
        end
    end

    edge = ones(T, 1)
    for i in 1:size(mps)[1]
        @tensor begin
            edge[k] := edge[a]*dropdims(mps[i], dims=(2,4))[a,k]
        end
    end

    if return_real
        return real(edge[1])
    else
        return imag(edge[1])
    end
end


function contract_sampled_single_layer_svd_conv(peps_data::Array{Array{T,4},2}, return_real::Bool, max_dim::Int64, rot_symm::Bool) where T
    psi = 0

    if rot_symm
        range = 0:3
    else
        range = 0
    end

    for k in range
        rot_peps_data = tensor_rotl90(peps_data, k)
        mps_up = [ones(T, 1, 1, 1, 1) for i in 1:size(rot_peps_data)[2]]
        mps_down = [ones(T, 1, 1, 1, 1) for i in 1:size(rot_peps_data)[2]]

        for i in 1:Int(size(rot_peps_data)[1]/2)
            up_ind = i
            down_ind = size(rot_peps_data, 1) - (i-1)
            mpo_up = rot_peps_data[up_ind,:]
            mpo_down = rot_peps_data[down_ind,:]
            mps_up = multiply_mps_mpo(mps_up, mpo_up)
            mps_down = multiply_mps_mpo(mpo_down, mps_down)
            if max_dim >= 1
                mps_up, norm_mps = truncate_boundary_mpo_svd(mps_up, max_dim)
                mps_down, norm_mps = truncate_boundary_mpo_svd(mps_down, max_dim)
            end
        end
        edge = ones(T, 1, 1)

        for i in 1:size(mps_up,1)
            @tensor begin
                temp[-1,-2,-3,-4] := edge[1,-1]*mps_up[i][1,-2,-3,-4]
            end
            @tensor begin
                edge[-1,-2] := temp[1,2,-1,3]*conj(mps_down[i])[1,3,-2,2]
            end
        end
        if return_real
            psi += real(edge[1])
        else
            psi += edge[1]
        end
    end
    return psi
end


function get_vert_boundary_and_truncate(mps::Array{Array{T,4},1}, mpo::Array{Array{T,4},1}, max_dim::Int64) where T
    mps_mpo = MPO{T}(3, 1, 1, 1, Dict{Int64,Tensor{T,4}}())

    for i in 1:size(mps)
        left_dim = size(mps[i-2].data)[1]
        top_dim = size(mps[i-2].data)[2]*size(mpo[i-2].data)[2]
        right_dim = size(mpo[i-2].data)[3]
        bot_dim = size(mps[i-2].data)[4]*size(mpo[i-2].data)[4]
        @tensor begin
            new_tensor[j,k,l,m,n,o] := mps[i-2].data[o,j,a,m]*mpo[i-2].data[a,k,l,n]
        end
        mps_mpo.tensors[i] = Tensor{T,4}(reshape(new_tensor, (top_dim,right_dim,bot_dim,left_dim)))
    end

    if max_dim >= 1
        _ = truncate_boundary_mpo_svd!(mps_mpo, max_dim)
    end

    new_mps = Dict{Int64,Tensor{T,4}}()
    for i in 1:3
        new_mps[i-2] = Tensor{T,4}(permutedims(mps_mpo.tensors[i].data, (4,1,2,3)))
    end
    return new_mps
end


function contract_sampled_single_layer_svdsolve(peps_data::Array{Array{T,4},2}, real::Bool, max_dim::Int64) where T
    mps = [ones(T, 1, 1, 1, 1) for i in 1:size(peps_data)[2]]

    for i in 1:size(peps_data)[1]
        mpo = peps_data[i,:]
        mps = multiply_mps_mpo(mps, mpo)
        if max_dim >= 1
            mps, norm_mps = truncate_boundary_mpo_svdsolve(mps, max_dim)
        end
    end

    edge = ones(T, 1)
    for i in 1:size(mps)[1]
        @tensor begin
            edge[k] := edge[a]*dropdims(mps[i], dims=(2,4))[a,k]
        end
    end

    if real
        return real(edge[1])
    else
        return imag(edge[1])
    end
end


function tensor_rotl90(tensors::Array{Array{T,4},2}, k::Int64) where T
    Ly, Lx = size(tensors)
    k = mod(k, 4)
    k == 1 ? reshape([permutedims(tensors[ind÷Lx+1, Lx-ind%Lx], circshift([1,2,3,4], -k)) for ind in 0:Lx*Ly-1], (Lx,Ly)) :
    k == 2 ? reshape([permutedims(tensors[Ly-ind%Ly, Lx-ind÷Ly], circshift([1,2,3,4], -k)) for ind in 0:Lx*Ly-1], (Ly,Lx)) :
    k == 3 ? reshape([permutedims(tensors[Ly-ind÷Lx,ind%Lx+1], circshift([1,2,3,4], -k)) for ind in 0:Lx*Ly-1], (Lx,Ly)) : tensors
end


function flip_spin(ind::Int64)
    if ind == 1
        return 2
    else
        return 1
    end
end


function get_all_horizontal_boundaries(sample_peps::Array{Array{T,4},2}, contract_cut::Int64) where T
    top_mps_boundaries = Array{Array{Array{T,4},1},1}()
    down_mps_boundaries = Array{Array{Array{T,4},1},1}()

    push!(top_mps_boundaries, [ones(T, 1, 1, 1, 1) for i in 1:size(sample_peps)[2]])
    push!(down_mps_boundaries, [ones(T, 1, 1, 1, 1) for i in 1:size(sample_peps)[2]])

    for i in 1:size(sample_peps,1)-1
        mpo_up = sample_peps[i,:]
        mpo_down = sample_peps[size(sample_peps,1)-i+1,:]
        mps_up = multiply_mps_mpo(top_mps_boundaries[i], mpo_up)
        mps_down = multiply_mps_mpo(mpo_down, down_mps_boundaries[i])
        if contract_cut >= 1
            mps_up, norm_mps = truncate_boundary_mpo_svd(mps_up, contract_cut)
            mps_down, norm_mps = truncate_boundary_mpo_svd(mps_down, contract_cut)
        end
        push!(top_mps_boundaries, mps_up)
        push!(down_mps_boundaries, mps_down)
    end
    down_mps_boundaries = reverse(down_mps_boundaries)
    return top_mps_boundaries, down_mps_boundaries
end


function get_fourind_dense_vertical_boundaries(top_mps_boundary::Array{Array{T,4},1}, pepsrow1::Array{Array{T,4},1}, pepsrow2::Array{Array{T,4},1}, down_mps_boundary::Array{Array{T,4},1}) where T
    four_sb_left = Array{Array{T,4},1}()
    push!(four_sb_left, ones(T, 1, 1, 1, 1))
    for j in 1:size(top_mps_boundary,1)
        mpo = [top_mps_boundary[j], pepsrow1[j], pepsrow2[j], down_mps_boundary[j]]
        push!(four_sb_left, contract_boundary_four_left(four_sb_left[j], mpo))
    end
    return four_sb_left
end


function get_threeind_dense_vertical_boundaries(top_mps_boundary::Array{Array{T,4},1}, pepsrow1::Array{Array{T,4},1}, down_mps_boundary::Array{Array{T,4},1}) where T
    three_sb_left = Array{Array{T,3},1}()
    push!(three_sb_left, ones(T, 1, 1, 1))
    for j in 1:size(top_mps_boundary,1)
        mpo = [top_mps_boundary[j], pepsrow1[j], down_mps_boundary[j]]
        push!(three_sb_left, contract_boundary_three_left(three_sb_left[j], mpo))
    end
    return three_sb_left
end


function exact_contract_peps(peps)
    boundary_tensors = Dict{Int,Array}()
    edge_bdim = peps.tensors[(1,1)].b_dims[2]
    unit_edge = Matrix(1.0I, edge_bdim, edge_bdim)
    boundary_tensors[1] = reshape(unit_edge, (size(unit_edge)...,1))
    for i in 2:peps.Lx-1
        boundary_tensors[i] = reshape(unit_edge, (size(unit_edge)...,1,1))
    end
    boundary_tensors[peps.Lx] = reshape(unit_edge, (size(unit_edge)...,1))

    for i in 1:peps.Ly
        @tensor begin
            temp[k,l,p,q,r] := peps.tensors[(i,1)].data[a,d,b,p,k]*boundary_tensors[1][b,c,q]*conj(peps.tensors[(i,1)].data)[a,d,c,r,l]
        end
        d1,d2,d3,d4,d5 = size(temp)
        top_d = d1
        bot_d = d2
        right_d = d3*d4*d5
        boundary_tensors[1] = reshape(temp, (top_d,bot_d,right_d))
        for j in 2:peps.Lx-1
            @tensor begin
                temp[k,l,m,n,o,p,q,r] := peps.tensors[(i,j)].data[a,m,b,p,k]*boundary_tensors[j][b,c,n,q]*conj(peps.tensors[(i,j)].data)[a,o,c,r,l]
            end
            d1,d2,d3,d4,d5,d6,d7,d8 = size(temp)
            top_d = d1
            bot_d = d2
            left_d = d3*d4*d5
            right_d = d6*d7*d8
            boundary_tensors[j] = reshape(temp, (top_d,bot_d,left_d,right_d))
        end
        @tensor begin
            temp[k,l,m,n,o] := peps.tensors[(i,peps.Lx)].data[a,m,b,d,k]*boundary_tensors[peps.Lx][b,c,n]*conj(peps.tensors[(i,peps.Lx)].data)[a,o,c,d,l]
        end
        d1,d2,d3,d4,d5 = size(temp)
        top_d = d1
        bot_d = d2
        left_d = d3*d4*d5
        boundary_tensors[peps.Lx] = reshape(temp, (top_d,bot_d,left_d))
    end
    @tensor begin
        edge[i] := boundary_tensors[1][a,a,i]
    end
    for i in 2:peps.Lx-1
        @tensor begin
            edge[j] := edge[a]*boundary_tensors[i][b,b,a,j]
        end
    end
    @tensor begin
        norm[] := edge[a]*boundary_tensors[peps.Lx][b,b,a]
    end

    return norm
end


function get_marginals(peps::PEPS{T}, marginal_bdim::Int64; operators=nothing, last_row=2) where T
    marginals = Array{Array{Array{T,4},1},1}()
    push!(marginals, [ones(T, 1,1,1,1) for j in 1:peps.Lx])
    function get_mpo_tensor(boundary_tensor, top_tensor, bot_tensor, op_tensor)
        @tensor tensor[-1,-2,-3,-4,-5,-6,-7,-8] := boundary_tensor[-1,1,-5,2]*top_tensor[3,-2,-4,-6,1]*bot_tensor[4,-3,-8,-7,2]*op_tensor[3,4]
        left1,left2,left3,top,right1,right2,right3,bot = size(tensor)
        return reshape(tensor, (left1*left2*left3,top,right1*right2*right3,bot))
    end
    for i in peps.Ly:-1:last_row
        if operators == nothing
            mpo = [get_mpo_tensor(marginals[peps.Ly-(i-1)][j], peps.tensors[(i,j)].data, conj(peps.tensors[(i,j)].data), Matrix{T}(I, peps.phys_dim, peps.phys_dim)) for j in 1:peps.Lx]
        else
            mpo = [get_mpo_tensor(marginals[peps.Ly-(i-1)][j], peps.tensors[(i,j)].data, conj(peps.tensors[(i,j)].data), operators[i,j]) for j in 1:peps.Lx]
        end
        mpo, norm = truncate_boundary_mpo_svd(mpo, marginal_bdim)
        push!(marginals, mpo)
    end

    return reverse(marginals)
end


function trivial_marginals(peps::PEPS{T}) where T
    function make_marginal_tensor(i, j)
        bdim = size(peps.tensors[(i,j)].data,3)
        return reshape(Matrix{T}(I,bdim, bdim), (1,bdim,1,bdim))
    end

    marginals = Array{Array{Array{T,4},1},1}()
    push!(marginals, [ones(T, 1,1,1,1) for j in 1:peps.Lx])
    for i in peps.Ly:-1:2
        mpo = [make_marginal_tensor(i,j) for j in 1:peps.Lx]
        push!(marginals, mpo)
    end
    return reverse(marginals)
end
