function get_local_log_gradients(peps::PEPS{T}, samples::Array{Int64,4}, norm_peps::T, elocs::Array{T,2}, gradients::Array{T,9}, max_dim::Int64, rot_symm::Bool) where T
    num_threads = Threads.nthreads()
    samples_per_thread = size(samples,2)
    num_samples = num_threads*samples_per_thread
    @Threads.threads for i in 1:num_threads
        for j in 1:samples_per_thread

            sample_data = [peps.tensors[(k,l)].data[samples[i,j,k,l],:,:,:,:] for k in 1:peps.Ly, l in 1:peps.Lx]
            top_mps_boundaries = Array{Array{Array{T,4},1},1}()
            down_mps_boundaries = Array{Array{Array{T,4},1},1}()

            push!(top_mps_boundaries, [ones(T, 1, 1, 1, 1) for i in 1:size(sample_data)[2]])
            push!(down_mps_boundaries, [ones(T, 1, 1, 1, 1) for i in 1:size(sample_data)[2]])

            for k in 1:peps.Ly-1
                mpo_up = sample_data[k,:]
                mpo_down = sample_data[peps.Ly-k+1,:]
                mps_up = multiply_mps_mpo(top_mps_boundaries[k], mpo_up)
                mps_down = multiply_mps_mpo(mpo_down, down_mps_boundaries[k])
                if max_dim >= 1
                    mps_up, norm_mps = truncate_boundary_mpo_svd(mps_up, max_dim)
                    mps_down, norm_mps = truncate_boundary_mpo_svd(mps_down, max_dim)
                end
                push!(top_mps_boundaries, mps_up)
                push!(down_mps_boundaries, mps_down)
            end
            psi = mps_overlap(top_mps_boundaries[end], sample_data[peps.Ly,:])
            down_mps_boundaries = reverse(down_mps_boundaries)

            for k in 1:peps.Ly
                three_sb_left = Array{Array{T,3},1}()
                three_sb_right = Array{Array{T,3},1}()
                push!(three_sb_left, ones(T, 1, 1, 1))
                push!(three_sb_right, ones(T, 1, 1, 1))
                for l in 1:peps.Lx-1
                    mpo_three = [top_mps_boundaries[k][l], sample_data[k,l], down_mps_boundaries[k][l]]
                    push!(three_sb_left, contract_boundary_three_left(three_sb_left[l], mpo_three))
                end
                for l in peps.Lx:-1:1
                    mpo_three = [top_mps_boundaries[k][l], sample_data[k,l], down_mps_boundaries[k][l]]
                    push!(three_sb_right, contract_boundary_three_right(three_sb_right[peps.Lx-l+1], mpo_three))

                    one_hot_vec = zeros(T, peps.phys_dim)
                    one_hot_vec[samples[i,j,k,l]] = 1
                    gradient = get_gradient_tensor(three_sb_left[l], top_mps_boundaries[k][l], three_sb_right[peps.Lx-l+1], down_mps_boundaries[k][l], one_hot_vec)
                    Dp, Dl, Dt, Dr, Dd = size(gradient)
                    gradients[i,j,k,l,1:Dp,1:Dl,1:Dt,1:Dr,1:Dd] = gradient ./ psi
                end
            end
        end
    end
end


function get_gradient_tensor(left_b::Array{T,3}, top_b::Array{T,4}, right_b::Array{T,3}, down_b::Array{T,4}, one_hot_vec::Array{T,1}) where T
    @tensor begin
        gradient[-1,-2,-3,-4,-5] := one_hot_vec[-1]*left_b[1,-2,4]*dropdims(top_b, dims=2)[1,2,-3]*right_b[2,-4,3]*dropdims(down_b, dims=4)[4,-5,3]
    end
    return gradient
end


function get_gradient_tensor(left_b::Dict{Int64,Tensor{T,4}}, top_b::Tensor{T,4}, right_b::Dict{Int64,Tensor{T,4}}, down_b::Tensor{T,4}) where T
    up_left = dropdims(left_b[-1].data, dims=(1,2))
    up_center = dropdims(top_b.data, dims=2)
    up_right = dropdims(right_b[-1].data, dims=(2,3))
    center_left = dropdims(left_b[0].data, dims=1)
    center_right = dropdims(right_b[0].data, dims=3)
    down_left = dropdims(left_b[1].data, dims=(1,4))
    down_center = dropdims(down_b.data, dims=4)
    down_right = dropdims(right_b[1].data, dims=(3,4))

    @tensor begin
        gradient[-1,-2,-3,-4] := center_left[1,-1,8]*up_left[2,1]*up_center[2,3,-2]*
                                up_right[3,4]*center_right[-3,4,5]*down_right[6,5]*
                                down_center[7,-4,6]*down_left[8,7]
    end
    return gradient
end


function stochastic_reconfiguration(local_gradients::Array{T,9}, psi_sqs::Array{T,2}, elocs::Array{T,2}, diag_shift::Float64, tol::Float64) where T
    num_threads = Threads.nthreads()
    samples_per_thread = size(psi_sqs,2)
    num_samples = num_threads*samples_per_thread

    gradient_shape = size(local_gradients)[3:end]
    local_gradients = reshape(local_gradients, (size(local_gradients, 1), size(local_gradients, 2), :))
    norm_peps = sum(psi_sqs)

    gradients = dropdims(sum((psi_sqs.*conj(elocs)).*local_gradients, dims=(1,2)), dims=(1,2))/norm_peps

    centered_local_gradients = local_gradients .- sum(psi_sqs.*local_gradients, dims=(1,2))/norm_peps

    function S_times_v(v)
        thread_data = zeros(num_threads, size(v, 1))
        @Threads.threads for i in 1:num_threads
            thread_data[i,:] = (transpose(centered_local_gradients[i,:,:]) * ((psi_sqs[i,:] .* centered_local_gradients[i,:,:]) * v)) ./ norm_peps
        end
        return dropdims(sum(thread_data, dims=1), dims=1)
    end

    if tol > 0
        update_vec, info = linsolve(S_times_v, gradients, diag_shift; tol=tol, isposdef=true, krylovdim=200)
    else
        update_vec, info = linsolve(S_times_v, gradients, diag_shift; isposdef=true, krylovdim=200)
    end
    print(info)
    update_vec = reshape(update_vec, gradient_shape)
    gradients = reshape(gradients, gradient_shape)
    return update_vec, gradients
end


function accumulate_gradient(local_gradients::Array{T,9}, psi_sqs::Array{T,2}, elocs::Array{T,2}) where T
    num_threads = Threads.nthreads()
    samples_per_thread = size(psi_sqs,2)
    num_samples = num_threads*samples_per_thread

    gradient_shape = size(local_gradients)[3:end]
    local_gradients = reshape(local_gradients, (prod(size(local_gradients)[1:2]), :))
    elocs = reshape(elocs, :)
    psi_sqs = reshape(psi_sqs, :)
    norm_peps = sum(psi_sqs)

    gradients = dropdims(sum((psi_sqs.*conj(elocs)).*local_gradients, dims=1), dims=1)/norm_peps
    gradients = reshape(gradients, gradient_shape)
    return gradients
end
