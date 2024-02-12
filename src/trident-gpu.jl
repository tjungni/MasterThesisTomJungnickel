"""
Since auto-differentiation does not work yet for the trident process, the GPU code is located in this separate file which is not loaded by the package.
To use the gpu version of trident, just rename load this file manually or rename it to trident.jl while replace the CPU implementation.
"""

using QEDbase
using CUDA

# QEDbase fixes to allow Zygote auto-differentiation in combination with CUDA.jl

function _build_particle_booster(
    mom::T, mass::Real
) where {T<:AbstractLorentzVector{TE}} where {TE<:Real}
    #_check_spinor_input(mom, mass)
    return (slashed(mom) + mass * one(DiracMatrix)) / (sqrt(abs(mom.t) + mass))
end

function _build_antiparticle_booster(
    mom::T, mass::Real
) where {T<:AbstractLorentzVector{TE}} where {TE<:Real}
    #_check_spinor_input(mom, mass)
    return (mass * one(DiracMatrix) - slashed(mom)) / (sqrt(abs(mom.t) + mass))
end

function QEDbase.IncomingFermionSpinor(
    mom::T, mass::Real
) where {T<:AbstractLorentzVector{TE}} where {TE<:Real}
    return QEDbase.IncomingFermionSpinor(_build_particle_booster(mom, mass))
end

function QEDbase.OutgoingFermionSpinor(
    mom::T, mass::Real
) where {T<:AbstractLorentzVector{TE}} where {TE<:Real}
    return QEDbase.OutgoingFermionSpinor(_build_particle_booster(mom, mass))
end

function QEDbase.OutgoingAntiFermionSpinor(
    mom::T, mass::Real
) where {T<:AbstractLorentzVector{TE}} where {TE<:Real}
    return QEDbase.OutgoingAntiFermionSpinor(_build_antiparticle_booster(mom, mass))
end

function QEDbase.IncomingAntiFermionSpinor(
    mom::T, mass::Real
) where {T<:AbstractLorentzVector{TE}} where {TE<:Real}
    return QEDbase.IncomingAntiFermionSpinor(_build_antiparticle_booster(mom, mass))
end

function (SP::IncomingFermionSpinor)(spin::AbstractSpin)
    return SP.booster * base_particle_spinor(spin)
end

function (SP::OutgoingFermionSpinor)(spin::AbstractSpin)
    return AdjointBiSpinor(SP.booster * base_particle_spinor(spin)) * GAMMA[1]
end

function (SP::OutgoingAntiFermionSpinor)(spin::AbstractSpin)
    return SP.booster * base_antiparticle_spinor(spin)
end

function (SP::IncomingAntiFermionSpinor)(spin::AbstractSpin)
    return AdjointBiSpinor(SP.booster * base_antiparticle_spinor(spin)) * GAMMA[1]
end


function base_particle_spinor(spin::SpinUp)
    return BiSpinor(1.0, 0.0, 0.0, 0.0)
end

function base_particle_spinor(spin::SpinDown)
    return BiSpinor(0.0, 1.0, 0.0, 0.0)
end

function base_antiparticle_spinor(spin::SpinUp)
    return BiSpinor(0.0, 0.0, 1.0, 0.0)
end

function base_antiparticle_spinor(spin::SpinDown)
    return BiSpinor(0.0, 0.0, 0.0, 1.0)
end

abstract type AbstractPol end
struct XPol <: AbstractPol end
struct YPol <: AbstractPol end

function FourPolarisation2(p::XPol)
    return SLorentzVector{ComplexF64}(0, 1, 0, 0) 
end

function FourPolarisation2(p::YPol)
    return SLorentzVector{ComplexF64}(0, 0, 1, 0)
end

# constants

"""
The m_e-Matrix used in the Feynman-diagram calculation.
"""
const mem =  DiracMatrix([1.0 0 0 0 ;
        0 1.0 0 0 ;
        0 0 1.0 0 ;
        0 0 0 1.0 ])


# matrix elements

function matrixelC(k, p, p1, p2, p3, pol, spin0, spin1, spin2, spin3)
    # factor -im*e_charge^3 is moved outside of the sum of all M
    epsilon = FourPolarisation2(pol)
    hinten = (SpinorUbar.(p2, (m_e,)) .|> s -> s(spin2)) .* (GAMMA,) .* (SpinorV.(p1, (m_e,)) .|> s -> s(spin1))
    frac1 = (hinten   .* (GAMMA,) .* ((slashed(p)  + slashed(k) + mem),).* (slashed(epsilon),)) ./ ((p +k)*(p +k) - m_e_sq)
    frac2 = ((slashed(epsilon),) .* (slashed.(p3) .- (slashed(k),) .+ (mem,)) .* hinten .* (GAMMA,)  ) ./ ((p3.-(k,)).*(p3.-(k,)) .- m_e_sq)  
    return 1 ./ ((p1.+p2).*(p1.+p2)) .* (SpinorUbar.(p3, (m_e,)) .|> s -> s(spin3)) .* (frac1 .+ frac2) .* (SpinorU(p, m_e)(spin0),)  
end


function matrixelBW(k, p, p1, p2, p3, pol, spin0, spin1, spin2, spin3)
    epsilon = FourPolarisation2(pol)
    vorn = (SpinorUbar.(p3, (m_e,)) .|> s -> s(spin3)) .* (GAMMA,) .* (SpinorU(p, m_e)(spin0),)
    frac1 = (vorn     .* (GAMMA,) .* (slashed.(-p1) .+ (slashed(k),) .+ (mem,)) .* (slashed(epsilon),)) ./ ((-p1.+(k,)).*(-p1.+(k,)) .- m_e_sq)
    frac2 = ((slashed(epsilon),) .* (slashed.( p2) .- (slashed(k),) .+ (mem,)) .* vorn   .* (GAMMA,)  ) ./ (( p2.-(k,)).*( p2.-(k,)) .- m_e_sq)
    return 1 ./ (((p,).-p3).*((p,).-p3)) .* (SpinorUbar.(p2, (m_e,)) .|> s -> s(spin2)) .* (frac1 .+ frac2) .* (SpinorV.(p1, (m_e,)) .|> s -> s(spin1))
end

function matrixelCx(k, p, p1, p2, p3, pol, spin0, spin1, spin2, spin3)
    return matrixelC(k, p, p1, p3, p2, pol, spin0, spin1, spin3, spin2)
end

function matrixelBWx(k, p, p1, p2, p3, pol, spin0, spin1, spin2, spin3)
    return matrixelBW(k, p, p1, p3, p2, pol, spin0, spin1, spin3, spin2)
end

function MpT(k, p, p1, p2, p3, pol, spin0, spin1, spin2, spin3)
    return matrixelC(k,p,p1,p2,p3,pol,spin0,spin1,spin2,spin3) .+ matrixelBW(k,p,p1,p2,p3,pol,spin0,spin1,spin2,spin3) .- matrixelCx(k,p,p1,p2,p3,pol,spin0,spin1,spin2,spin3) .- matrixelBWx(k,p,p1,p2,p3,pol,spin0,spin1,spin2,spin3)
end


# diff cross section

function get_omega(ss)
    return (ss^2 - m_e_sq) / (2.0*m_e)
end

function get_ss(omega)
    return sqrt(2.0*omega*m_e + m_e_sq)
end

function unsafe_get_remaining_kin(ss, Ea, ctha, phia, Eb, cthb)
    omega = get_omega(ss)
    E = m_e
    rho = 0.0

    rhoa = sqrt.(Ea.^2 .- m_e_sq)
    stha = sqrt.(1 .- ctha.^2)

    rhob = sqrt.(Eb.^2 .- m_e_sq)
    sthb = sqrt.(1 .- cthb.^2)

    return (omega, E, rho, rhoa, stha, rhob, sthb)
end

function unsafe_get_kin_facs(ss, omega, E, rho, Ea, rhoa, ctha, stha, Eb, rhob, cthb, sthb)
    a = ss^2 + 1.0 .- 2 .* (Ea .* (omega + E) .- rhoa .* (omega - rho) .* ctha) .-
        2 * (Eb .* (omega + E) .- rhob .* (omega - rho) .* cthb) .+
        2 .* (Ea .* Eb .- rhoa .* rhob .* ctha .* cthb)
    b = 2 .* rhoa .* rhob .* stha .* sthb
    return a, b
end

function _get_phi2(a::AbstractArray, b::AbstractArray, phia::AbstractArray)
    phi_zero = acos.(a ./ b)
    phib_1 = mod2pi.(phi_zero .+ phia)
    phib_2 = mod2pi.(2 * pi .- phi_zero .+ phia)
    return phib_1, phib_2
end

function build_mom_out_fermion(E::AbstractArray, rho::AbstractArray, cth::AbstractArray, sth::AbstractArray, phi::AbstractArray)
    SFourMomentum.(E, rho .* sth .* cos.(phi), rho .* sth .* sin.(phi), rho .* cth)
end

function unsafe_build_out_momenta(
    K::QEDbase.AbstractFourMomentum, 
    P::QEDbase.AbstractFourMomentum, 
    Ea::AbstractArray, 
    rhoa::AbstractArray, 
    ctha::AbstractArray, 
    stha::AbstractArray,
    phia::AbstractArray,
    Eb::AbstractArray, 
    rhob::AbstractArray, 
    cthb::AbstractArray, 
    sthb::AbstractArray,
    phib1::AbstractArray,
    phib2::AbstractArray)

    Ptot = P + K
    Pa = build_mom_out_fermion(Ea, rhoa, ctha, stha, phia)
    Pb1 = build_mom_out_fermion(Eb, rhob, cthb, sthb, phib1)
    Pb2 = build_mom_out_fermion(Eb, rhob, cthb, sthb, phib2)
    Pc1 = (Ptot,) .- Pa .- Pb1
    Pc2 = (Ptot,) .- Pa .- Pb2

    return (Pa,Pb1,Pc1), (Pa,Pb2,Pc2)
end

function check_physics(omega::Real, ss::Real, E::Real, Ea::AbstractArray, Eb::AbstractArray, a_fac::AbstractArray, b_fac::AbstractArray)
    # only Float64 for now
    temp = (abs.(a_fac) .<= abs.(b_fac))  # putting this directly in the line below will make CUDA complain 
    (ss >= 3.0*m_e) .&& ((omega + E) .>= (Ea .+ Eb)) .&& temp
end

function unsafe_dσpT(omega::Real, ss::Real, E::Real, rho::Real, Ea::AbstractArray, rhoa::AbstractArray, stha::AbstractArray, ctha::AbstractArray, phia::AbstractArray, Eb::AbstractArray, rhob::AbstractArray, sthb::AbstractArray, cthb::AbstractArray, a1::AbstractArray, b1::AbstractArray)
    phib1, phib2 = _get_phi2(a1, b1, phia)
    k = SFourMomentum(omega, 0.0, 0.0, omega)
    p = SFourMomentum(E, 0.0, 0.0, -rho)       # k and p are just one momentum each, not arrays of momenta like the rest
    (pa, pb1, pc1), (pa1, pb2, pc2) = unsafe_build_out_momenta(k, p, Ea, rhoa, ctha, stha, phia, Eb, rhob, cthb, sthb, phib1, phib2)

    I = k * p
    
    N = 0.125  # 1/8
    prefac = 1.0/(4.0*I) * N .* (getRho.(pc1) .* getRho.(pb1)) ./ sqrt.(b1.^2 - a1.^2) .*  e_charge^6 ./ (4.0*(2.0*pi)^5)

    
    S1x = abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinUp(), SpinDown(), SpinDown(), SpinDown())) .+                 
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, XPol(), SpinDown(), SpinDown(), SpinDown(), SpinDown()))    

    S1y = abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinUp(), SpinDown(), SpinDown(), SpinDown())) .+         
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb1, pc1, YPol(), SpinDown(), SpinDown(), SpinDown(), SpinDown())) 

    S2x = abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinUp(), SpinDown(), SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, XPol(), SpinDown(), SpinDown(), SpinDown(), SpinDown()))    
                                            
    S2y = abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinUp(), SpinDown(), SpinDown(), SpinDown())) .+                                       
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinDown(), SpinUp(),   SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinUp(),   SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinDown(), SpinDown(), SpinUp())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinUp(),   SpinUp(),   SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinDown(), SpinUp(),   SpinDown())) .+ 
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinUp(),   SpinDown(), SpinDown())) .+
          abs2.(MpT(k, p, pa, pb2, pc2, YPol(), SpinDown(), SpinDown(), SpinDown(), SpinDown()))
    
    S = S1x .+ S1y .+ S2x .+ S2y

    return prefac .* S
end

"""
Calculates the differential cross section for the trident process; returns 0.0 if the input values are not physical.
Only works for Float64 for now because everything in QEDbase is Float64 and CUDA will complain about converts.
"""
function dσpT(omega::Float64, Ea::AbstractArray{Float64}, ctha::AbstractArray{Float64}, phia::AbstractArray{Float64}, Eb::AbstractArray{Float64}, cthb::AbstractArray{Float64})
    ss = get_ss(omega)
    (omega_test, E, rho, rhoa, stha, rhob, sthb) = unsafe_get_remaining_kin(ss, Ea, ctha, phia, Eb, cthb)  # E and rho are just Floats, the rest are Arrays 
    a1, b1 = unsafe_get_kin_facs(ss, omega, E, rho, Ea, rhoa, ctha, stha, Eb, rhob, cthb, sthb)

    return ifelse.(check_physics(omega, ss, E, Ea, Eb, a1, b1), unsafe_dσpT(omega, ss, E, rho, Ea, rhoa, stha, ctha, phia, Eb, rhob, sthb, cthb, a1, b1), CUDA.zeros(Float64, length(Ea)))
end

function dσpT_wrapper(x::AbstractArray{Float64})
    omega = 0.8
    Ea = x[1,:]
    ctha = x[2,:]
    phia = x[3,:]
    Eb = x[4,:]
    cthb = x[5,:]

    return dσpT.((omega,), Ea, ctha, phia, Eb, cthb)
end
