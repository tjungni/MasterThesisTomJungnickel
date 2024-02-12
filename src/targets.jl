

"""
A simple gaussian centered around (0,0) so the input dimension could for example be ϵ [-1,1]^2
"""
function single_gauss(xy::T) where {T <: AbstractArray{F}} where {F <: Real}
    return exp.(-sum((xy.*F(1)).^2,dims=1))
end


"""
Two gaussians, one at the top right and one at the bottom left of the [0,1]^2. (at (0.25,0.25) and (0.75,0.75) to be exact)
"""
function double_gauss(xy::T) where {T <: AbstractArray{F}} where {F <: Real}
    return exp.(-sum((xy.-F(0.75)).^2 ./(F(0.2)^2),dims=1)) + exp.(-sum((xy.-F(0.25)).^2 ./(F(0.2)^2),dims=1))
end


"""
Compton scattering differential cross section for input (cθ, ω'bar) but the first input is actually ϵ [0,1] and gets translated to [-1,1] internally.
"""
function comptonf(x, omega, dphi)
    T = eltype(x)
    ω = T(omega)  
    Δϕ = T(dphi)   
    background = (x -> cossquare(x, Δϕ))
    cθ    = x[1,:]
    ω2bar = x[2,:]
    return dσdω2dcθ_ex(cθ, ω2bar, ω, background)
end
