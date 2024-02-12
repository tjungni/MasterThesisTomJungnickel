## Functions to map the [0,1]^d output of NIS models to the space that is required

abstract type ChannelMapping end

"""
This map just outputs the input.
It exists because the loss functions require a map so this one can be passed when no mapping is required.
"""
struct IdentityMapping <: ChannelMapping end
function (::IdentityMapping)(x)
    return x
end

"""
Determinant of the channel mapping
"""
function cmdet(cm::IdentityMapping)
    return one_i
end


### Compton

"""
This map maps x[1] to [-1,1] and doesn't affect x[2].
x[1] = cθ   ϵ [-1,1]
x[2] = ωbar ϵ [0,1]  
"""
struct HypercubeTocθωbar <: ChannelMapping end
function (::HypercubeTocθωbar)(x::CuArray)  # GPU case
    T = eltype(x)
    return x .* CuArray(repeat([T(2),T(1)],1,size(x,2))) .- CuArray(repeat([T(1),T(0)],1,size(x,2))) 
end


function (::HypercubeTocθωbar)(x)  # CPU case
    T = eltype(x)
    cθ = x[1,:]'
    ωbar = x[2,:]'
    return vcat(cθ .* T(2) .- T(1), ωbar)
end

"""
Determinant of the channel mapping
"""
function cmdet(cm::HypercubeTocθωbar)
    return two_i
end


### Trident

function get_E_lim(omega::F) where {F <: Real}
    ss = get_ss(omega)
    Et = (ss^2 + m_e_sq)/(F(2.0) * m_e)
    return (Et*(ss^2-3) + omega*sqrt((ss^2-1)*(ss^2-9)))/(2*ss^2) 
end

"""
Takes a matrix x of phase space points and one value for omega as input and returns the value
of the differential cross section of the trident process for each phase space point (column of x) and omega.
The colrows of x should represent:
    x[1,:] = E_a
    x[2,:] = cos(theta_a)
    x[3,:] = phi_a
    x[4,:] = E_b
    x[5,:] = cos(theta_b)

    detval = 
"""
struct trident_phasespace{F} <: ChannelMapping where {F <: Real} 
    omega::F
    Ea_lim::F # with offset -1
    Eb_lim::F # with offset -1
    det_val::F
end

function trident_phasespace(omega)
    return trident_phasespace(omega, 2.5, get_E_lim(omega)-1.0, 4.0*0.11*pi*get_E_lim(omega)*2.5)
end

function (cm::trident_phasespace)(x::AbstractArray{F}) where {F <: Real}
    Ea = x[1,:] .* cm.Ea_lim .+ F(1.0)
    ctha = x[2,:] .* F(2.0) .- F(1.0)
    phia = x[3,:] .* (F(2.0)*pi)
    Eb = x[4,:] .* cm.Eb_lim .+ F(1.0)
    cthb = x[5,:] .* F(0.11) .+ F(0.89)
    return Ea, ctha, phia, Eb, cthb
end

"""
Determinant of the channel mapping
"""
function cmdet(cm::trident_phasespace)
    return cm.det_val
end
