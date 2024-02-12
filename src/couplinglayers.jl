using Flux

## Coupling layers which are stacked in neural importance sampling


abstract type AbstractLayer end

"""
A coupling layer takes a batch x of random points ϵ [0,1]^d as input and returns a batch of sampled boints ϵ [0,1]^d
First bins W and heights V are generated by the neural network m contained in the coupling layer which are then used by an invertible seperable map C to generate the sample points
The first n values of the input points stay unchanged while the remaining values get converted by NN and C
"""
struct CouplingLayer{M,N} <: AbstractLayer where {M <: Chain, N <: InvSepMap}
    d::Int64      # dimension of input and output, split into A = [1,...,n] and B = [n,...,d]
    dimA::Int64   # size of A; size of B is thus d-n
    dimB::Int64
    C::N          # invertible seperable map, eg. PiecewiseQuadratic or IdentityMapping
    m::M          # neural network
    bins::Int64   # required to reshaping the output of m
end

struct CouplingLayerCPU{M,N} <: AbstractLayer where {M <: Chain, N <: InvSepMap}
    d::Int64      # dimension of input and output, split into A = [1,...,n] and B = [n,...,d]
    dimA::Int64   # size of A; size of B is thus d-n
    dimB::Int64
    C::N          # invertible seperable map, eg. PiecewiseQuadratic or IdentityMapping
    m::M          # neural network
    bins::Int64   # required to reshaping the output of m
end

function (cl::CouplingLayer)(x::T) where {T <: AbstractArray}
    xA = x[1:cl.dimA,:]
    xB = x[cl.dimA+1:cl.d,:]
    Vh, Wh = cl.m(xA)
    V, W = normalizeVW(Vh, Wh, cl.bins, cl.dimB)
    yA = xA
    yB = cl.C(xB, V, W, cl.dimB)
    return vcat(yA, yB)
end

function (cl::CouplingLayerCPU)(x::T) where {T <: AbstractArray}
    xA = x[1:cl.dimA,:]
    xB = x[cl.dimA+1:cl.d,:]
    Vh, Wh = cl.m(xA)
    V, W = normalizeVW(Vh, Wh, cl.bins, cl.dimB)
    yA = xA
    yB = cl.C(xB, V, W, cl.dimB)
    return vcat(yA, yB)
end

# This makes the coupling layers trainable in Flux
Flux.@functor CouplingLayer
Flux.@functor CouplingLayerCPU
Flux.trainable(cl::CouplingLayer) = (;cl.m)
Flux.trainable(cl::CouplingLayerCPU) = (;cl.m)

"""
Swaps the two partitions of the input x.
x = [1 2 3 4
     5 6 7 8]
dimA = 1
y = [5 6 7 8
     1 2 3 4]
To be inserted inbetween coupling layers.
"""
struct SwapLayer <: AbstractLayer
    d::Int64      # dimension of input and output, split into A = [1,...,n] and B = [n,...,d]
    dimA::Int64   # size of A; size of B is thus d-n
end

function(sl::SwapLayer)(x::T) where {T <: AbstractArray}
    xA = x[1:sl.dimA,:]
    xB = x[sl.dimA+1:sl.d,:]
    yA = xB
    yB = xA
    return vcat(yA, yB)
end

"""
Similar to Swaplayer but the exact masking can be controlled
"""
struct MaskLayer <: AbstractLayer
    mask::AbstractArray{Bool,1} 
end

struct MaskLayerCPU <: AbstractLayer
    mask::AbstractArray{Bool,1} 
end

function(ml::MaskLayer)(x::T) where {T <: AbstractArray}
    batchsize = size(x, 2)
    dimtot = size(x, 1)
    dima = sum(ml.mask)
    dimb = dimtot - dima
    fullmask = repeat(ml.mask, 1, batchsize) |> gpu
    yA = reshape(x[fullmask], dima, batchsize)
    yB = reshape(x[.!fullmask], dimb, batchsize)
    return vcat(yA, yB)
end

function(ml::MaskLayerCPU)(x::T) where {T <: AbstractArray}
    batchsize = size(x, 2)
    dimtot = size(x, 1)
    dima = sum(ml.mask)
    dimb = dimtot - dima
    fullmask = repeat(ml.mask, 1, batchsize)
    yA = reshape(x[fullmask], dima, batchsize)
    yB = reshape(x[.!fullmask], dimb, batchsize)
    return vcat(yA, yB)
end


"""
Using split, multiple paths for the same input can be created within a chain resulting in multiple outputs
"""
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x) = map(f -> f(x), m.paths)


"""
Creates a neural network that outputs 2 matrices, for neural importance sampling those are:
Vh = value at each vertex of some function (which??) used by piecewise quadratic  
Wh = bin widths
(h means unnormalized)
"""


function goetz_NN(dimA::Signed, dimB::Signed, bins::Signed, width=16)
    return Chain(
        Split(
            Chain(
                BatchNorm(dimA),
                Dense(dimA => width, relu),
                BatchNorm(width),
                Dense(width => width, relu),
                BatchNorm(width),
                Dense(width => width, relu),
                BatchNorm(width),
                Dense(width => dimB*(bins+1))  
                ), 
            Chain(
                BatchNorm(dimA),
                Dense(dimA => width, relu),
                BatchNorm(width),
                Dense(width => width, relu),
                BatchNorm(width),
                Dense(width => width, relu),
                BatchNorm(width),
                Dense(width => dimB*bins)
                )
            ) 
        ) |> gpu
end

function NN9(dimA::Signed, dimB::Signed, bins::Signed, width=16)
    return Chain(
        Split(
            Chain(
                BatchNorm(dimA),
                Dense(dimA => width, relu),
                Dense(width => width, relu),
                Dense(width => width, relu),
                Dense(width => dimB*(bins+1))  
                ), 
            Chain(
                BatchNorm(dimA),
                Dense(dimA => width, relu),
                Dense(width => width, relu),
                Dense(width => width, relu),
                Dense(width => dimB*bins)
                )
            ) 
        ) |> gpu
end

CouplingLayer(d::Int64, dimA::Int64, nbins::Int64, nn_cons) = CouplingLayer(d, dimA, d-dimA, PiecewiseQuadratic(), nn_cons(dimA, d-dimA, nbins), nbins)
CouplingLayerCPU(d::Int64, dimA::Int64, nbins::Int64, nn_cons) = CouplingLayer(d, dimA, d-dimA, PiecewiseQuadraticCPU(), nn_cons(dimA, d-dimA, nbins), nbins)