mutable struct Tensor{T,N}
    data::Array{T,N}
end

function Base.getproperty(t::Tensor,s::Symbol)
    if s == :phys_dim
        return size(t.data,1)
    elseif s == :b_dims
        return Base.tail(size(t.data))
    else
        return getfield(t,s)
    end
end

mutable struct PEPS{T}
    Lx::Int64
    Ly::Int64
    phys_dim::Int64
    b_dim::Int64
    edge_dim::Int64
    tensors::Dict{Tuple,Tensor{T,5}}
end

mutable struct MPO{T}
    Lx::Int64
    in_dim::Int64
    out_dim::Int64
    b_dim::Int64
    tensors::Dict{Int64, Tensor{T,4}}
end
