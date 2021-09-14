function init_PEPS(Lx::Int64, Ly::Int64, phys_dim::Int64, b_dim::Int64, edge_dim::Int64, positive::Bool, T::Type)
    tensors = Dict{Tuple,Tensor{T,5}}()
    for i in 1:Ly
        for j in 1:Lx
            if edge_dim == b_dim
                if positive
                    peps_tensor = Tensor{T,5}(rand(T, phys_dim,b_dim,b_dim,b_dim,b_dim))
                else
                    peps_tensor = Tensor{T,5}(randn(T, phys_dim,b_dim,b_dim,b_dim,b_dim))
                end
                tensors[(i,j)] = peps_tensor
            else
                if j==1; l_D = edge_dim; else l_D = b_dim end
                if i==1; t_D = edge_dim; else t_D = b_dim end
                if j==Lx; r_D = edge_dim; else r_D = b_dim end
                if i==Ly; b_D = edge_dim; else b_D = b_dim end
                if positive
                    peps_tensor = Tensor{T,5}(0.1.*rand(T, phys_dim,l_D,t_D,r_D,b_D))
                else
                    peps_tensor = Tensor{T,5}(randn(T, phys_dim,l_D,t_D,r_D,b_D))
                end
                tensors[(i,j)] = peps_tensor
            end
        end
    end
    peps = PEPS{T}(Lx, Ly, phys_dim, b_dim, edge_dim, tensors)
    return peps
end


function init_ising_PEPS(Lx::Int64, Ly::Int64, beta::Float64, glassy::Bool)
    tensors = Dict{Tuple,Tensor{Complex{Float64},5}}()
    for i in 1:Ly
        for j in 1:Lx
            if j==1; l_D = 1; else l_D = 2 end
            if i==1; t_D = 1; else t_D = 2 end
            if j==Lx; r_D = 1; else r_D = 2 end
            if i==Ly; b_D = 1; else b_D = 2 end

            tensor = zeros(Complex{Float64}, (2, l_D, t_D, r_D, b_D))
            tensor[1,1,1,1,1] = 1
            tensor[end,end,end,end,end] = 1
            tensors[(i,j)] = Tensor{Complex{Float64},5}(tensor)
        end
    end
    #horizontal bonds
    for i in 1:Ly
        for j in 1:Lx-1
            if glassy
                c = randn()
            else
                c = 1
            end

            ising_mat = sqrt([exp(c*beta/2) exp(-c*beta/2); exp(-c*beta/2) exp(c*beta/2)])
            @tensor new_tensor[-1,-2,-3,-4,-5] := tensors[(i,j)].data[-1,-2,-3,1,-5]*ising_mat[1,-4]
            tensors[(i,j)].data = new_tensor
            @tensor new_tensor[-1,-2,-3,-4,-5] := ising_mat[-2,1]*tensors[(i,j+1)].data[-1,1,-3,-4,-5]
            tensors[(i,j+1)].data = new_tensor
        end
    end

    #vertical bonds
    for i in 1:Ly-1
        for j in 1:Lx
            if glassy
                c = randn()
            else
                c = 1
            end

            ising_mat = sqrt([exp(c*beta/2) exp(-c*beta/2); exp(-c*beta/2) exp(c*beta/2)])
            @tensor new_tensor[-1,-2,-3,-4,-5] := tensors[(i,j)].data[-1,-2,-3,-4,1]*ising_mat[1,-5]
            tensors[(i,j)].data = new_tensor
            @tensor new_tensor[-1,-2,-3,-4,-5] := ising_mat[-3,1]*tensors[(i+1,j)].data[-1,-2,1,-4,-5]
            tensors[(i+1,j)].data = new_tensor
        end
    end
    peps = PEPS{Complex{Float64}}(Lx, Ly, 2, 2, 1, tensors)
    return peps
end


function read_PEPS_npy(directory::String, Lx::Int64, Ly::Int64, phys_dim::Int64, b_dim::Int64, T::Type)
    tensors = Dict{Tuple,Tensor{T,5}}()
    for i in 1:Ly
        for j in 1:Lx
            if j==1; l_D = 1; else l_D = b_dim end
            if i==1; t_D = 1; else t_D = b_dim end
            if j==Lx; r_D = 1; else r_D = b_dim end
            if i==Ly; b_D = 1; else b_D = b_dim end
            data = npzread(directory * string(i-1) * "_" * string(j-1) * ".npy")
            peps_tensor = Tensor{T,5}(data)
            tensors[(i,j)] = peps_tensor
        end
    end
    peps = PEPS(Lx, Ly, phys_dim, b_dim, 1, tensors)
    return peps
end


function write_PEPS_npy(peps::PEPS{T}, directory::String) where T
    for i in 1:peps.Ly
        for j in 1:peps.Lx
            npzwrite(directory * string(i-1) * "_" * string(j-1) * ".npy", peps.tensors[(i,j)].data)
        end
    end
end


function rotate_PEPS_90(peps::PEPS)
    rot_tensors = Dict{Tuple,Tensor{5}}()
    for i in 1:peps.Ly
        for j in 1:peps.Lx
            tensor = peps.tensors[(i,j)]
            tensor.data = permutedims(tensor.data, (1,5,2,3,4))
            r_ind = j
            c_ind = peps.Ly+1 - i
            rot_tensors[(r_ind,c_ind)] = tensor
        end
    end
    return PEPS(peps.Ly, peps.Lx, peps.phys_dim, peps.b_dim, peps.edge_dim, rot_tensors)
end


function normalize_peps_stochastically!(peps::PEPS{T}, sample_cut::Int64, marginals::Array{Array{Array{T,4},1},1}) where T
    for i in 1:peps.Ly
        part_norm = 0
        for j in 1:100
            edge_dim = peps.edge_dim
            boundary_samples_hor = rand(1:edge_dim, peps.Lx)
            mps = construct_boundary_mps(4, edge_dim, boundary_samples_hor, (1/sqrt(edge_dim))*ones(peps.Lx), T)
            sample_peps, boundary_samples_vert, boundary_samples_hor, mps = get_sample(peps, mps, i, sample_cut, false, marginals)
            norm_peps = mps_overlap(mps,mps)
            # sample_peps, boundary_samples_vert, boundary_samples_hor, norm_peps = get_sample(peps, i, sample_cut, false)
            part_norm += norm_peps/100
        end
        for j in 1:peps.Lx
            peps.tensors[(i,j)].data /= part_norm^(1/(2*peps.Lx))
        end
    end
end
