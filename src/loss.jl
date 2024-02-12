
## Different loss functions for neural importance sampling
using Flux


"""
Calculates the determinant of a coupling layer for the given x
"""
function cldet(cl::CouplingLayer, x::S, Vh::T, Wh::U) where {S<:AbstractArray{F} , T<:AbstractArray{F}, U<:AbstractArray{F}} where F<:Real
    V, W = normalizeVW(Vh, Wh, cl.bins, cl.dimB)
    a, b = calculate_ab(reshape(x, 1, length(x)), W)
    Vb = sum(V.*onehotorzerobatch(b, size(V,1)), dims=1)
    Vb1 = sum(V.*onehotorzerobatch(b.+1, size(V,1)), dims=1)
    res = reshape((Vb + a .* (Vb1 .- Vb)), cl.dimB, Int64(size(a, 2)/cl.dimB))
    return prod(res, dims=1)
end

function cldet_cpu(cl::CouplingLayer, x::S, Vh::T, Wh::U) where {S<:AbstractArray{F} , T<:AbstractArray{F}, U<:AbstractArray{F}} where F<:Real
    V, W = normalizeVW(Vh, Wh, cl.bins, cl.dimB)
    a, b = calculate_ab(reshape(x, 1, length(x)), W)
    Vb = sum(V.*onehotorzerobatch_cpu(b, size(V,1)), dims=1)
    Vb1 = sum(V.*onehotorzerobatch_cpu(b.+1, size(V,1)), dims=1)
    res = reshape((Vb + a .* (Vb1 .- Vb)), cl.dimB, Int64(size(a, 2)/cl.dimB))
    return prod(res, dims=1)
end


"""
Calculates the weights of all input points (columns) of x

x must be the output samples of the model WITHOUT! having been mapped by cm (like for plotting a heatmap)
"""
function weights(m::Chain, cm::ChannelMapping, f::Function, x::T) where {T <: AbstractArray}
    return jacobian(m, cm, x) .* f(cm(m(x)))
end


"""
Calculates the loss as the average weight deviation from 1 
"""
function pearsonχ2divergence(m::Chain, cm::ChannelMapping, f::Function, x::T) where T<:AbstractArray{F}  where F<:Real
    weight_diffs = (weights(m, cm, f, x) .-1).^F(2)
    return sum(weight_diffs) / size(x,2)
end

"""
Calculates the loss of the batch of points x according to the Pearson-χ^2-Divergence between the target function and the 1/jacobian of the model
"""
function pearsonv2(m::Chain, cm::ChannelMapping, f::Function, x::T) where T<:AbstractArray{F}  where F<:Real
    zi = cm(m(x))
    g = 1 ./ jacobian(m, cm, x)
    fracs = (f(zi) .- g) .^F(2) ./ g
    return sum(fracs) / size(x,2)
end

function pearsonv3(m::Chain, cm::ChannelMapping, f::Function, x::T) where T<:AbstractArray{F} where F<:Real
    zi = cm(m(x))
    g = 1 ./ jacobian(m, cm, x)
    fracs = abs.(f(zi) .- g) .^F(1.5) ./ g
    return sum(fracs) / size(x,2)
end

function pearsonv4(m::Chain, cm::ChannelMapping, f::Function, x::T) where T<:AbstractArray{F} where F<:Real
    zi = cm(m(x))
    g = 1 ./ jacobian(m, cm, x)
    fracs = abs.(f(zi) .- g) .^F(1.5) ./ f(zi)
    return sum(fracs) / size(x,2)
end

function pearsonv5(m::Chain, cm::ChannelMapping, f::Function, x::T) where T<:AbstractArray{F} where F<:Real
    zi = cm(m(x))
    g = 1 ./ jacobian(m, cm, x)
    fracs = (f(zi)) .^F(2) ./ g
    return sum(fracs) / size(x,2)
end


function jacobian(m::Chain, cm::ChannelMapping, x::T) where {T <: AbstractArray{F}} where F<:Real
    ### Change this for different numbers of coupling layers
    cl1 = m[1]
    sl1 = m[2]
    cl2 = m[3]
    sl2 = m[4]
    cl3 = m[5]
    x2 = cl1(x)
    x2s = sl1(x2)
    x3 = cl2(x2s)
    x3s = sl2(x3)
    det1 = abs.(cldet(cl1,  x[cl1.dimA+1:cl1.d,:], cl1.m( x[1:cl1.dimA,:])...))
    det2 = abs.(cldet(cl2, x2s[cl2.dimA+1:cl2.d,:], cl2.m(x2s[1:cl2.dimA,:])...))
    det3 = abs.(cldet(cl3, x3s[cl3.dimA+1:cl3.d,:], cl3.m(x3s[1:cl3.dimA,:])...)) 
    return abs(cmdet(cm)) .* det1 .* det2 .* det3
end

function jacobian4cl(m::Chain, cm::ChannelMapping, x::T) where {T <: AbstractArray{F}} where F<:Real
    cl1 = m[1]
    sl1 = m[2]
    cl2 = m[3]
    sl2 = m[4]
    cl3 = m[5]
    sl3 = m[6]
    cl4 = m[7]
    x2 = cl1(x)
    x2s = sl1(x2)
    x3 = cl2(x2s) 
    x3s = sl2(x3)
    x4 = cl3(x3s)
    x4s = sl3(x4)
    det1 = abs.(cldet(cl1,  x[cl1.dimA+1:cl1.d,:], cl1.m( x[1:cl1.dimA,:])...))
    det2 = abs.(cldet(cl2, x2s[cl2.dimA+1:cl2.d,:], cl2.m(x2s[1:cl2.dimA,:])...))
    det3 = abs.(cldet(cl3, x3s[cl3.dimA+1:cl3.d,:], cl3.m(x3s[1:cl3.dimA,:])...)) 
    det4 = abs.(cldet(cl4, x4s[cl4.dimA+1:cl4.d,:], cl4.m(x4s[1:cl4.dimA,:])...)) 
    return abs(cmdet(cm)) .* det1 .* det2 .* det3 .* det4
end
