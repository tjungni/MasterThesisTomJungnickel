## Background field shapes (in fourier space)

function cossquare(l::S, Δϕ::T) where {S <: AbstractArray{T}} where {T <: Real}
    ξ = zero_i  # polarization parameter, lin. pol. for 0 (and π/2)
    # sinc(x) is actually sinc(π*x), so wee need to do sinc(x/π)
    # consts2 = m_e*π^2/(2*e_charge)
    return  consts2 * T(cos(ξ)) * Δϕ * (sinc.(Δϕ*(l.+one_i)./π)./(pi_sq .-Δϕ^2*(l.+one_i).^2) .+ sinc.(Δϕ*(l.-one_i)./π)./(pi_sq .-Δϕ^2*(l.-one_i).^2)) 
end


function cossquare(l::T, Δϕ::T) where {T <: Real}
    # legacy support for old notebooks 
    ξ = zero_i
    return m_e*π^2/(two_i*e_charge) * T(cos(ξ)) * Δϕ * (sinc(Δϕ*(l+one_i)/π)/(π^2 -Δϕ^2*(l+one_i)^2) + sinc(Δϕ*(l-one_i)/π)/(π^2 -Δϕ^2*(l-one_i)^2)) 
end
