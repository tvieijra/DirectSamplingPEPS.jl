function construct_boundary_mps(index::Int64, b_dim::Int64, boundary_samples::Array{Int64,1}, norm::Array{Float64,1}, T::Type)
    function construct_boundary_tensor(index::Int64, b_dim::Int64, sample::Int64, norm::Float64)
        shape = ones(Int64, 4)
        shape[index] = b_dim
        boundary_vector = zeros(T, b_dim)
        boundary_vector[sample] = 1/norm
        boundary_tensor = Tensor{T,4}(reshape(boundary_vector, Tuple(shape)))
        return boundary_tensor
    end

    boundary_tensors = Dict{Int, Tensor{T,4}}()
    L = size(boundary_samples)[1]

    for i in 1:L
        boundary_tensors[i] = construct_boundary_tensor(index, b_dim, boundary_samples[i], norm[i])
    end
    mps = MPO{T}(L, 1, b_dim, b_dim, boundary_tensors)

    return mps
end


function contract_mpo_boundary(mpo::MPO, edge_dim::Int64, left_boundary::Int64, right_boundary::Int64)
    left_edge = zeros(Complex{Float64},(1,edge_dim))
    left_edge[1,left_boundary] = 1
    right_edge = zeros(Complex{Float64},(edge_dim,1))
    right_edge[right_boundary,1] = 1
    @tensor begin
        mpo.tensors[1].data[j,k,l,m] := left_edge[j,a]*mpo.tensors[1].data[a,k,l,m]
    end
    @tensor begin
        mpo.tensors[mpo.Lx].data[j,k,l,m] := mpo.tensors[mpo.Lx].data[j,k,a,m]*right_edge[a,l]
    end

    return mpo
end


function contract_mpo_boundary!(mpo::MPO, left_boundary::Int64, right_boundary::Int64)
    mpo.tensors[1].data = reshape(mpo.tensors[1].data[left_boundary,:,:,:], (1,size(mpo.tensors[1].data)[2:end]...))
    mpo.tensors[mpo.Lx].data = reshape(mpo.tensors[mpo.Lx].data[:,:,right_boundary,:], (size(mpo.tensors[mpo.Lx].data)[1:2]...,1,size(mpo.tensors[mpo.Lx].data, 4)))
end
