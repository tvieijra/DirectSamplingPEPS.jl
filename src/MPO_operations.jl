function multiply_mps_pepsrow!(mps::MPO{T}, peps::PEPS{T}, row_ind::Int64) where T
    for i in 1:mps.Lx
        @tensor begin
            mps_mpo_tensor[k,l,j,m,n,o,p] := mps.tensors[i].data[k,m,n,a]*peps.tensors[(row_ind, i)].data[j,l,a,o,p]
        end
        _,_,phys_dim,top_dim,_,_,bot_dim = size(mps_mpo_tensor)
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[5]*size(mps_mpo_tensor)[6]
        mps.tensors[i].data = reshape(mps_mpo_tensor, (new_left_dim, phys_dim, new_right_dim, bot_dim))
        mps.in_dim = phys_dim
    end
end


function multiply_mps_mpo!(mps::MPO{T}, mpo::MPO{T}) where T
    for i in 1:mps.Lx
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o] := mps.tensors[i].data[j,l,m,a]*mpo.tensors[i].data[k,a,n,o]
        end
        bot_dim = size(mps_mpo_tensor)[end]
        top_dim = size(mps_mpo_tensor)[3]
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[4]*size(mps_mpo_tensor)[5]
        mps.tensors[i].data = reshape(mps_mpo_tensor, (new_left_dim,top_dim,new_right_dim,bot_dim))
    end
end


function multiply_mps_mpo(mps::Array{Array{T, 4},1}, mpo::Array{Array{T, 4},1}) where T
    function multiply_tensors(mps_ten, mpo_ten)
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o] := mps_ten[j,l,m,a]*mpo_ten[k,a,n,o]
        end
        bot_dim = size(mps_mpo_tensor)[end]
        top_dim = size(mps_mpo_tensor)[3]
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[4]*size(mps_mpo_tensor)[5]
        return reshape(mps_mpo_tensor, (new_left_dim,top_dim,new_right_dim,bot_dim))
    end
    return [multiply_tensors(mps[i], mpo[i]) for i in 1:size(mps)[1]]
end


function multiply_mps_mpo_vert(mps::Array{Array{T, 4},1}, mpo::Array{Array{T, 4},1}) where T
    rotmps = map(x->permutedims(x, [2,3,4,1]), mps)
    rotmpo = map(x->permutedims(x, [2,3,4,1]), mpo)
    return map(x->permutedims(x, [4,1,2,3]), multiply_mps_mpo(rotmps, rotmpo))
end


function multiply_mpo_mps(mpo::Array{Array{T, 4},1}, mps::Array{Array{T, 4},1}) where T
    function multiply_tensors(mpo_ten, mps_ten)
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o] := mpo_ten[j,l,m,a]*mps_ten[k,a,n,o]
        end
        bot_dim = size(mps_mpo_tensor)[end]
        top_dim = size(mps_mpo_tensor)[3]
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[4]*size(mps_mpo_tensor)[5]
        return reshape(mps_mpo_tensor, (new_left_dim,top_dim,new_right_dim,bot_dim))
    end
    return [multiply_tensors(mpo[i], mps[i]) for i in 1:size(mps)[1]]
end


function multiply_mpo_mps!(mpo::MPO{T}, mps::MPO{T}) where T
    for i in 1:mps.Lx
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o] := mpo.tensors[i].data[j,l,m,a]*mps.tensors[i].data[k,a,n,o]
        end
        bot_dim = size(mps_mpo_tensor)[end]
        top_dim = size(mps_mpo_tensor)[3]
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[4]*size(mps_mpo_tensor)[5]
        mps.tensors[i].data = reshape(mps_mpo_tensor, (new_left_dim,top_dim,new_right_dim,bot_dim))
    end
end


function multiply_mps_mpo(mps::MPO{T}, mpo::MPO{T}) where T
    next_mps = MPO(mps.Lx, 1, mpo.out_dim, mps.b_dim*mpo.b_dim, Dict{Int64,Tensor{T,4}}())
    for i in 1:mps.Lx
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o] := mps.tensors[i].data[j,l,m,a]*mpo.tensors[i].data[k,a,n,o]
        end
        bot_dim = size(mps_mpo_tensor)[end]
        top_dim = size(mps_mpo_tensor)[3]
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[4]*size(mps_mpo_tensor)[5]
        next_mps.tensors[i] = Tensor{T,4}(reshape(mps_mpo_tensor, (new_left_dim,top_dim,new_right_dim,bot_dim)))
    end
    return next_mps
end


function multiply_mpo_mps(mpo::MPO{T}, mps::MPO{T}) where T
    next_mps = MPO(mps.Lx, 1, mpo.out_dim, mps.b_dim*mpo.b_dim, Dict{Int64,Tensor{T,4}}())
    for i in 1:mps.Lx
        @tensor begin
            mps_mpo_tensor[j,k,l,m,n,o] := mpo.tensors[i].data[j,l,m,a]*mps.tensors[i].data[k,a,n,o]
        end
        bot_dim = size(mps_mpo_tensor)[end]
        top_dim = size(mps_mpo_tensor)[3]
        new_left_dim = size(mps_mpo_tensor)[1]*size(mps_mpo_tensor)[2]
        new_right_dim = size(mps_mpo_tensor)[4]*size(mps_mpo_tensor)[5]
        next_mps.tensors[i] = Tensor{T,4}(reshape(mps_mpo_tensor, (new_left_dim,top_dim,new_right_dim,bot_dim)))
    end
    return next_mps
end


function canonicalize_boundary_mpo_left!(mpo::MPO{T}) where T
    for i in 1:mpo.Lx-1
        left_dim,top_dim,right_dim,bot_dim = size(mpo.tensors[i].data)
        tensor = reshape(permutedims(mpo.tensors[i].data, (1,2,4,3)), (top_dim*left_dim*bot_dim,right_dim))
        F = try
            svd(tensor)
        catch e
            npzwrite(pwd()*"/bad_svd.npz", tensor)
            svd(tensor; alg=LinearAlgebra.QRIteration())
        end
        mpo.tensors[i].data = permutedims(reshape(F.U, (left_dim,top_dim,bot_dim,size(F.U)[2])), (1,2,4,3))
        @tensor begin
            mpo.tensors[i+1].data[-1,-2,-3,-4] := Diagonal(F.S)[-1,1]*F.Vt[1,2]*mpo.tensors[i+1].data[2,-2,-3,-4]
        end

    end
    @tensor begin
        norm[] := mpo.tensors[mpo.Lx].data[a,b,c,d]*conj(mpo.tensors[mpo.Lx].data)[a,b,c,d]
    end
    return norm[]
end


function canonicalize_boundary_mpo_right!(mpo::MPO{T}) where T
    for i in mpo.Lx:-1:2
        left_dim,top_dim,right_dim,bot_dim = size(mpo.tensors[i].data)
        tensor = reshape(mpo.tensors[i].data, (left_dim,top_dim*bot_dim*right_dim))
        F = try
            svd(tensor)
        catch e
            npzwrite(pwd()*"/bad_svd.npz", tensor)
            svd(tensor; alg=LinearAlgebra.QRIteration())
        end
        mpo.tensors[i].data = reshape(F.Vt, (size(F.Vt)[1],top_dim,right_dim,bot_dim))
        @tensor begin
            mpo.tensors[i-1].data[-1,-2,-3,-4] := mpo.tensors[i-1].data[-1,-2,2,-4]*F.U[2,1]*Diagonal(F.S)[1,-3]
        end

    end
    @tensor begin
        norm[] := mpo.tensors[1].data[a,b,c,d]*conj(mpo.tensors[1].data)[a,b,c,d]
    end

    return norm[]
end


function canonicalize_boundary_mpo_right_svd(mpo::Array{Array{T,4},1}) where T
    new_mpo = copy(mpo)

    function canonicalize_tensor(tensor, left_tensor, index)
        left_dim,top_dim,right_dim,bot_dim = size(tensor)
        tensorr = reshape(tensor, (left_dim,top_dim*bot_dim*right_dim))
        link_dim = min(size(tensorr, 1), size(tensorr, 2))
        U, S, V = try
            svd(tensorr)
        catch e
            npzwrite(pwd()*"/bad_svd.npz", tensorr)
            svd(tensorr, alg=LinearAlgebra.QRIteration())
        end
        new_mpo[index] = reshape(V', (size(V, 2),top_dim,right_dim,bot_dim))
        @tensor begin
            next_tensor[-1,-2,-3,-4] := left_tensor[-1,-2,2,-4]*U[2,1]*Diagonal(S)[1,-3]
        end
        return next_tensor
    end

    tensor = mpo[size(mpo, 1)]
    for i in size(mpo, 1):-1:2
        tensor = canonicalize_tensor(tensor, mpo[i-1], i)
    end
    new_mpo[1] = tensor
    @tensor begin
        norm[] := new_mpo[1][a,b,c,d]*conj(new_mpo[1])[a,b,c,d]
    end

    return copy(new_mpo), norm[]
end


function truncate_boundary_mpo_svd!(mpo::MPO{T}, max_dim::Int64) where T
    norm_mpo = canonicalize_boundary_mpo_right!(mpo)

    for i in 1:mpo.Lx-1
        left_dim,top_dim,right_dim,bot_dim = size(mpo.tensors[i].data)
        tensor = reshape(permutedims(mpo.tensors[i].data, (1,2,4,3)), (top_dim*left_dim*bot_dim,right_dim))
        F = try
            svd(tensor)
        catch e
            npzwrite(pwd()*"/bad_svd.npz", tensor)
            svd(tensor; alg=LinearAlgebra.QRIteration())
        end
        # F = svd(tensor)

        norm_before = norm(F.S)
        max_dim >= 1 ? right_dim = min(max_dim,size(F.U)[2]) : right_dim = size(F.U)[2]
        norm_after = norm(F.S[1:right_dim])
        new_S = F.S[1:right_dim] * (norm_before/norm_after)

        mpo.tensors[i].data = permutedims(reshape(F.U[:,1:right_dim], (left_dim,top_dim,bot_dim,right_dim)), (1,2,4,3))

        @tensor begin
            mpo.tensors[i+1].data[-1,-2,-3,-4] := Diagonal(new_S)[-1,1]*F.Vt[1:right_dim,:][1,2]*mpo.tensors[i+1].data[2,-2,-3,-4]
        end
    end
    return norm_mpo
end


function truncate_boundary_mpo_svd(mpo::Array{Array{T,4},1}, max_dim::Int64) where T
    can_mpo, norm_mpo = canonicalize_boundary_mpo_right_svd(mpo)
    new_mpo = copy(can_mpo)

    function truncate_tensor(tensor, right_tensor, index)
        left_dim,top_dim,right_dim,bot_dim = size(tensor)
        tensorr = reshape(permutedims(tensor, (1,2,4,3)), (top_dim*left_dim*bot_dim,right_dim))
        max_dim >= 1 ? link_dim = min(max_dim, size(tensorr, 1), size(tensorr, 2)) : link_dim = min(size(tensorr, 1), size(tensorr, 2))
        U, S, V = try
            svd(tensorr)
        catch e
            npzwrite(pwd()*"/bad_svd.npz", tensorr)
            svd(tensorr, alg=LinearAlgebra.QRIteration())
        end
        # U, S, V = svd(tensorr; alg=LinearAlgebra.QRIteration())
        # U, S, V = svd(tensorr)
        norm_before = norm(S)
        right_dim = min(link_dim,size(U, 2))
        norm_after = norm(S[1:right_dim])
        new_S = S[1:right_dim] * (norm_before/norm_after)

        new_mpo[index] = permutedims(reshape(U[:,1:right_dim], (left_dim,top_dim,bot_dim,right_dim)), (1,2,4,3))
        @tensor begin
            # next_tensor[-1,-2,-3,-4] := Diagonal(new_S)[-1,1]*V'[1:right_dim,:][1,2]*right_tensor[2,-2,-3,-4]
            next_tensor[-1,-2,-3,-4] := V'[1:right_dim,:][-1,2]*right_tensor[2,-2,-3,-4]
        end
        mat_S = repeat(new_S, outer = [1,size(next_tensor,2)])
        return mat_S .* next_tensor
    end

    tensor = mpo[1]
    for i in 1:size(mpo,1)-1
        tensor = truncate_tensor(tensor, mpo[i+1], i)
    end
    new_mpo[size(mpo,1)] = tensor
    return copy(new_mpo), norm_mpo
end


function truncate_boundary_mpo_svd_vert(mpo::Array{Array{T,4},1}, max_dim::Int64) where T
    rotmpo = map(x->permutedims(x, [2,3,4,1]), mpo)
    trunc_mpo, norm_mpo = truncate_boundary_mpo_svd(rotmpo, max_dim)
    return map(x->permutedims(x, [4,1,2,3]), trunc_mpo), norm_mpo
end


function contract_boundary_three_left(boundary::Array{T,3}, new_tensors::Array{Array{T,4},1}) where T
    @tensor begin
        new_boundary[-1,-2,-3] := boundary[1,2,3]*dropdims(new_tensors[1], dims=2)[1,-1,4]*new_tensors[2][2,4,-2,5]*dropdims(new_tensors[3], dims=4)[3,5,-3]
    end
    return new_boundary
end


function contract_boundary_four_left(boundary::Array{T,4}, new_tensors::Array{Array{T,4},1}) where T
    @tensor begin
        new_boundary[-1,-2,-3,-4] := boundary[1,2,3,4]*dropdims(new_tensors[1], dims=2)[1,-1,5]*new_tensors[2][2,5,-2,6]*new_tensors[3][3,6,-3,7]*dropdims(new_tensors[4], dims=4)[4,7,-4]
    end
    return new_boundary
end


function contract_boundary_three_right(boundary::Array{T,3}, new_tensors::Array{Array{T,4},1}) where T
    @tensor begin
        new_boundary[-1,-2,-3] := boundary[1,2,3]*dropdims(new_tensors[1], dims=2)[-1,1,4]*new_tensors[2][-2,4,2,5]*dropdims(new_tensors[3], dims=4)[-3,5,3]
    end
    return new_boundary
end


function contract_boundary_four_right(boundary::Array{T,4}, new_tensors::Array{Array{T,4},1}) where T
    @tensor begin
        new_boundary[-1,-2,-3,-4] := boundary[1,2,3,4]*dropdims(new_tensors[1], dims=2)[-1,1,5]*new_tensors[2][-2,5,2,6]*new_tensors[3][-3,6,3,7]*dropdims(new_tensors[4], dims=4)[-4,7,4]
    end
    return new_boundary
end


function contract_three_leftright(left::Array{T,3}, right::Array{T,3}) where T
    @tensor begin
        psi[] := left[1,2,3]*right[1,2,3]
    end
    return psi[]
end


function contract_four_leftright(left::Array{T,4}, right::Array{T,4}) where T
    @tensor begin
        psi[] := left[1,2,3,4]*right[1,2,3,4]
    end
    return psi[]
end
