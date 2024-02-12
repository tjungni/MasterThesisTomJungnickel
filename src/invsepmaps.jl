#using Flux

abstract type InvSepMap end
struct PiecewiseQuadratic <: InvSepMap end
struct PiecewiseQuadraticCPU <: InvSepMap end


## functions that are needed for the pwq calculation

"""
Normalizing the columns of V to make them PDFs.
"""
function normalizeVW(Vh::T, Wh::T, bins::Signed, dimB::Signed) where {T <: AbstractArray{F}} where {F <: Real}
    samples = Int64(size(Wh,2))
    Vs = reshape(Vh,(Int64(bins+1), Int64(dimB*samples)))
    Ws = reshape(Wh,(bins, Int64(dimB*samples)))
    W  = softmax(Ws)
    expV = exp.(Vs)
    underfrac = sum( (expV[1:end-1,:] + expV[2:end,:]) .* W, dims=1 ) .* half_i
    V = expV ./ underfrac
    return V, W
end


"""
Finds the bin that cointains the x in xB (b) and the relative position in this bin (alpha).
"""
function calculate_ab(xB::T, W::T) where {T <: CuArray{F}} where F<:Real
    # the .+ eps() is very important because softmax() does not guarantee that the result actually sums to 1.0, often it only sums to 1.0-eps(), which means the columns in W don't sum to 1.0
    # when the input value xB is 1.0 (which happened regularly during training when Adam pushed the values to 1.0) the resulting b is greater than the number of bins in W which leads to Watb being 0 and a division by 0 at the alpha calculation
    Wfix = W + vcat(CuArray(zeros(F, size(W,1)-1, size(W,2))), CuArray(repeat([eps(F)], 1, size(W,2)))) 
    Wcum = cumsum(Wfix;dims=1)                          # bin widths cummulative (last value in each column is 1.0)
    Wtrue = Wcum .>= xB                                 # true for every position in W that contains the xi in xB
    b = -(sum(Wtrue, dims=1) .- (size(Wtrue,1)+1))'     # the b-array (bin numbers that contain the xi)  # this esentially does the same as just doing findfirst.(eachcol(Wtrue)) but is parallel!!
    W2 = Wcum .* onehotorzerobatch(b.-F(1), size(W,1))  
    sumb = xB - sum(W2, dims=1)                         # W summed up to the bin before b in each column
    Watb = sum(Wfix .* onehotorzerobatch(b, size(W,1)), dims=1) .+ eps(F)  # the width of the bin b in each column                                                             
    # again + eps because sumB can sometimes become bigger than Watb through float inaccuracy in the subtraction but Watb always has to be bigger so alpha is always <= 1.0                 
    alpha = sumb./Watb                                  # relative position of xi in each bin b  =  sum of all previous bins / that bin
    return alpha, b
end

function calculate_ab(xB::T, W::T) where {T <: AbstractArray{F}} where F<:Real
    Wfix = W + vcat(zeros(F, size(W,1)-1, size(W,2)), repeat([eps(F)], 1, size(W,2))) 
    Wcum = cumsum(Wfix;dims=1)                          # bin widths cummulative (last value in each column is 1.0)
    Wtrue = Wcum .>= xB                                 # true for every position in W that contains the xi in xB
    b = -(sum(Wtrue, dims=1) .- (size(Wtrue,1)+1))'   # the b-array (bin numbers that contain the xi)  # this esentially does the same as just doing findfirst.(eachcol(Wtrue)) but is parallel!!
    W2 = Wcum .* onehotorzerobatch_cpu(b.-F(1), size(W,1))  
    sumb = xB - sum(W2, dims=1)                         # W summed up to the bin before b in each column
    Watb = sum(Wfix .* onehotorzerobatch_cpu(b, size(W,1)), dims=1) .+ eps(F)  # the width of the bin b in each column                                                                      
    alpha = sumb./Watb                                  # relative position of xi in each bin b  =  sum of all previous bins / that bin
    return alpha, b
end


"""
Does the actual piecewise quadratic calculation which consists of 3 terms.
The first term is the interpolation of the random point xBi not being exactly on the edge of a bin (where the values is known and stored in V) but somehwere in between 2 edges.
The second term is the value of the bin containing xBi.
"""
function (C::PiecewiseQuadratic)(xB::T, V::T, W::T, ndim::Signed) where {T <: AbstractArray{F}} where F<:Real
    xBn = reshape(xB, 1, length(xB))
    alpha, b = calculate_ab(xBn, W)
    Vb = sum(V.*onehotorzerobatch(b, size(V,1)), dims=1)
    Vb1 = sum(V.*onehotorzerobatch(b.+1, size(V,1)), dims=1)
    Wb = sum(W.*onehotorzerobatch(b, size(W,1)), dims=1)
    term1 = alpha.^2 .* (Vb1 .- Vb) .* Wb * half_i
    term2 = alpha .* Vb .* Wb
    Vfrom1tobm1 = (V .* hotuptob(b.-1, size(V,1)))[1:end-1,:]
    Vfrom2tob   = (V .* hotuptob(b, size(V,1)))[2:end,:]
    Wfrom1tobm1 = W .* hotuptob(b.-1, size(W,1))
    term3 = sum( (Vfrom1tobm1 .+ Vfrom2tob) .* Wfrom1tobm1, dims=1) .* half_i
    allterms = term1 + term2 + term3
    return reshape(allterms, ndim, Int64(length(allterms)/ndim))
end

function (C::PiecewiseQuadraticCPU)(xB::T, V::T, W::T, ndim::Signed) where {T <: AbstractArray{F}} where F<:Real
    xBn = reshape(xB, 1, length(xB))
    alpha, b = calculate_ab(xBn, W)
    Vb = sum(V.*onehotorzerobatch_cpu(b, size(V,1)), dims=1)
    Vb1 = sum(V.*onehotorzerobatch_cpu(b.+1, size(V,1)), dims=1)
    Wb = sum(W.*onehotorzerobatch_cpu(b, size(W,1)), dims=1)
    term1 = alpha.^2 .* (Vb1 .- Vb) .* Wb * half_i
    term2 = alpha .* Vb .* Wb
    Vfrom1tobm1 = (V .* hotuptob_cpu(b.-1, size(V,1)))[1:end-1,:]
    Vfrom2tob   = (V .* hotuptob_cpu(b, size(V,1)))[2:end,:]
    Wfrom1tobm1 = W .* hotuptob_cpu(b.-1, size(W,1))
    term3 = sum( (Vfrom1tobm1 .+ Vfrom2tob) .* Wfrom1tobm1, dims=1) .* half_i
    allterms = term1 + term2 + term3
    return reshape(allterms, ndim, Int64(length(allterms)/ndim))
end
