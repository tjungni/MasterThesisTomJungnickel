## Functions needed to calculate compton scattering cross sections

Base.@irrational(m_e, 1.0, big(1.0))
Base.@irrational(m_e_sq, 1.0, big(m_e*m_e))
Base.@irrational(α, 0.007297352569311, big(0.007297352569311))  # ≈ 1/137
Base.@irrational(e_charge, 0.30282212087231697, sqrt(α*π*big(4)))
Base.@irrational(zero_i, 0.0, big(0.0))
Base.@irrational(one_i, 1.0, big(1.0))
Base.@irrational(two_i, 2.0, big(2.0))
Base.@irrational(half_i, 0.5, big(0.5))
Base.@irrational(pi_sq, 9.869604401089358, big(π*π))
#Base.@irrational(consts, 0.0001672940641563447, big(π*α^2/(m_e^2)))
Base.@irrational(consts, 1.0, big(1.0))  # set to 1 as explained in the thesis
Base.@irrational(consts2, 16.296042661379442, big(m_e*π^2/(two_i*e_charge)))


"""
ω of the photon after compton scattering.
"""
function calc_ω2(ω::T, cθ::T) where {T <: Real}
    return ω / ( one_i + ω/m_e*(one_i-cθ))
end


"""
Mandelstab variable t = (p1-p3)^2 = squared momentum transfer.
"""
function t(cθ::T, ω::T) where {T <: Real}
    return two_i * ω * calc_ω2(ω, cθ) * (cθ-one_i)
end


"""
Minimum value t can take in compton scattering. (Max is 0)
"""
function t_ex(ω::T) where {T <: Real}
    s = m_e_sq + two_i*ω*m_e
    return -(s-m_e_sq)^2/s
end


"""
Diff cross section for (non-pp) compton.
"""
function dσdcθ(cθ::T, ω::T) where {T <: Real}
    ω2 = calc_ω2(ω, cθ)
    return consts * (ω2/ω)^2 * (ω2/ω + ω/ω2 - (one_i-cθ^2)) # we assume ϕ=0
end


## Pulsed perturbative formulas

"""
Calculte ω2 for a given l.
"""
function lofω2(ω2::T, ω::T, cθ::T) where {T <: Real}
    return ω2/ω * one_i/(one_i-ω2/m_e*(one_i-cθ))
end

function lofω2(ω2::S, ω::T, cθ::S) where {S <: AbstractArray{T}} where {T <: Real}
    return ω2/ω .* one_i./(one_i.-ω2./m_e.*(one_i.-cθ))
end


"""
Diff cross section for pp_compton.
"""
function dσdldcθ(cθ::T, ω::T, l::T) where {T <: Real}
    return dσdcθ(cθ, ω*l)
end

function dσdldcθ(cθ::T, ω::T, l::T, F::Function) where {T <: Real}
    return dσdcθ(cθ, ω*l) * abs2(F(l))
end


"""
Diff cross section for pp_compton.
dσ/dωt = dσ/dcθ * dcθ/dt = dσ/dcθ * (dt/dcθ)^-1 = dσ/dcθ * jacobi 
"""
function dσdldt(cθ::T, ω::T, l::T) where {T <: Real}
    jacobi = (m_e - ω*l*cθ + ω*l)^2 / (two_i * m_e_sq * (ω*l)^2) 
    return dσdcθ(cθ, ω*l) * jacobi
end

function dσdldt(cθ::T, ω::T, l::T, F::Function) where {T <: Real}
    jacobi = (m_e - ω*l*cθ + ω*l)^2 / (two_i * m_e_sq * (ω*l)^2) 
    return dσdcθ(cθ, ω*l) * jacobi * abs2(F(l))
end


"""
Diff cross section for pp_compton.
dσ/dω2 = dσ/dcθ * dcθ/dω2 = dσ/dcθ * (dω2/dcθ)^-1 = dσ/dcθ * jacobi 
"""
function dσdω2dcθ(cθ::T, ω2::T, ω::T) where {T <: Real}
    l = lofω2(ω2, ω, cθ)
    jacobi = (m_e - ω*l*cθ + ω*l)^2 / (m_e_sq * ω * two_i)
    return consts * (ω2/ω)^2 * (ω2/ω + ω/ω2 - (one_i-cθ^2)) * jacobi # not using dσdldcθ(cθ, ω*l) here because that would calculate ω2 but we already have it
end

function dσdω2dcθ(cθ::T, ω2::T, ω::T, F::Function) where {T <: Real}
    l = lofω2(ω2, ω, cθ)
    jacobi = (m_e - ω*l*cθ + ω*l)^2 / (m_e_sq * ω * two_i)
    return consts * (ω2/(ω*l))^2 * (ω2/(ω*l) + (ω*l)/ω2 - (one_i-cθ^2)) * jacobi * abs2(F(l))
end

## ωbar Functions

function calc_ω2ex(cθ::S) where {S <: AbstractArray{T}} where {T <: Real}
    return m_e./(one_i.-cθ)
end

function dσdω2dcθ_ex(cθ::S, ω2bar::S, ω::T, F::Function) where {S <: AbstractArray{T}} where {T <: Real}
    ω2 = ω2bar .* calc_ω2ex(cθ)
    l = lofω2(ω2, ω, cθ)
    jacobi = (m_e .- ω*l.*cθ + ω*l).^2 ./ (m_e_sq * ω * two_i) .* ( m_e./(one_i.-cθ .+ m_e./ω2))
    return consts * (ω2./(ω.*l)).^2 .* (ω2./(ω*l) .+ (ω*l)./ω2 .- (one_i.-cθ.^2)) .* jacobi .* abs2.(F(l))
end
