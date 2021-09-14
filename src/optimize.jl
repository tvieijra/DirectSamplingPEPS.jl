abstract type Optimizer end

@with_kw mutable struct SGD <: Optimizer
    step_size::Float64
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
end

function update!(opt::SGD, peps::PEPS{T}, gradients::Array{T,7}) where T
    if opt.use_clipping
        clamp!(gradients, -opt.clip_val, opt.clip_val)
        norm(gradients) > opt.clip_norm ? gradients = (gradients ./ norm(gradients)) .* opt.clip_norm : nothing
    end
    for i in 1:peps.Ly
        for j in 1:peps.Lx
            Dp, Dl, Dt, Dr, Dd = size(peps.tensors[(i,j)].data)
            peps.tensors[(i,j)].data -= opt.step_size*gradients[i,j,1:Dp,1:Dl,1:Dt,1:Dr,1:Dd]
        end
    end
end


@with_kw mutable struct Adabelief{T} <: Optimizer
    step_size::Float64
    m::Array{T,7}
    s::Array{T,7}
    step::Int64 = 0
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
    epsilon::Float64 = 1e-8
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
end


function update!(opt::Adabelief{T}, peps::PEPS{T}, gradients::Array{T,7}) where T
    if opt.use_clipping
        clamp!(gradients, -opt.clip_val, opt.clip_val)
        norm(gradients) > opt.clip_norm ? gradients = (gradients ./ norm(gradients)) .* opt.clip_norm : nothing
    end
    opt.step += 1
    opt.m = opt.beta1 .* opt.m + (1-opt.beta1) .* gradients
    opt.s = opt.beta2 .* opt.s + (1-opt.beta2) .* ((gradients .- opt.m) .^ 2)
    m = opt.m ./ (1 - opt.beta1^opt.step)
    s = opt.s ./ (1 - opt.beta2^opt.step)
    for i in 1:peps.Ly
        for j in 1:peps.Lx
            Dp, Dl, Dt, Dr, Dd = size(peps.tensors[(i,j)].data)
            peps.tensors[(i,j)].data -= opt.step_size * m[i,j,1:Dp,1:Dl,1:Dt,1:Dr,1:Dd] ./ (sqrt.(s[i,j,1:Dp,1:Dl,1:Dt,1:Dr,1:Dd]) .+ opt.epsilon)
        end
    end
end


@with_kw mutable struct Adam{T} <: Optimizer
    step_size::Float64
    m::Array{T,7}
    v::Array{T,7}
    step::Int64 = 0
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
    epsilon::Float64 = 1e-8
    use_clipping::Bool = false
    clip_norm::Float64 = 5
    clip_val::Float64 = 1
end


function update!(opt::Adam{T}, peps::PEPS{T}, gradients::Array{T,7}) where T
    if opt.use_clipping
        clamp!(gradients, -opt.clip_val, opt.clip_val)
        norm(gradients) > opt.clip_norm ? gradients = (gradients ./ norm(gradients)) .* opt.clip_norm : nothing
    end
    opt.step += 1
    opt.m = opt.beta1 .* opt.m + (1-opt.beta1) .* gradients
    opt.v = opt.beta2 .* opt.v + (1-opt.beta2) .* (gradients .^ 2)
    m = opt.m ./ (1 - opt.beta1^opt.step)
    v = opt.v ./ (1 - opt.beta2^opt.step)
    for i in 1:peps.Ly
        for j in 1:peps.Lx
            Dp, Dl, Dt, Dr, Dd = size(peps.tensors[(i,j)].data)
            peps.tensors[(i,j)].data -= opt.step_size * m[i,j,1:Dp,1:Dl,1:Dt,1:Dr,1:Dd] ./ (sqrt.(v[i,j,1:Dp,1:Dl,1:Dt,1:Dr,1:Dd]) .+ opt.epsilon)
        end
    end
end
