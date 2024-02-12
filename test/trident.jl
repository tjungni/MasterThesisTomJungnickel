using QEDbase
using CUDA
using Flux
using CSV
using DataFrames

#QEDbase.SPINOR_VALIDITY_CHECK[] = false  # because isonshell is still broken in QEDbase

data = CSV.read("pert_trident_mat_prefac_diffCS2.csv", DataFrame)

k_i  = map((k0,k1,k2,k3) -> SFourMomentum(k0,k1,k2,k3), data.K_0,   data.K_1,   data.K_2,   data.K_3)
p_i  = map((p0,p1,p2,p3) -> SFourMomentum(p0,p1,p2,p3), data.P_0,   data.P_1,   data.P_2,   data.P_3)
p1_i = map((p0,p1,p2,p3) -> SFourMomentum(p0,p1,p2,p3), data.Ppp_0, data.Ppp_1, data.Ppp_2, data.Ppp_3)
p2_i = map((p0,p1,p2,p3) -> SFourMomentum(p0,p1,p2,p3), data.Pep_0, data.Pep_1, data.Pep_2, data.Pep_3)
p3_i = map((p0,p1,p2,p3) -> SFourMomentum(p0,p1,p2,p3), data.Per_0, data.Per_1, data.Per_2, data.Per_3)

Ek_i = getE.(k_i)
Ea_i = getE.(p1_i)
cta_i = getCosTheta.(p1_i)
phia_i = getPhi.(p1_i)
Eb_i = getE.(p2_i)
ctb_i = getCosTheta.(p2_i)

gEk_i = CuArray(getE.(k_i))
gEa_i = CuArray(getE.(p1_i))
gcta_i = CuArray(getCosTheta.(p1_i))
gphia_i = CuArray(getPhi.(p1_i))
gEb_i = CuArray(getE.(p2_i))
gctb_i = CuArray(getCosTheta.(p2_i))


@testset "matrix element" begin
    @test isapprox(data.M_C_01111, abs2.(matrixelC.(k_i, p_i, p1_i, p2_i, p3_i, (XPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))  
    @test isapprox(data.M_C_10111, abs2.(matrixelC.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinDown(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))
    @test isapprox(data.M_C_11011, abs2.(matrixelC.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinDown(),)))) ##spins swapped
    @test isapprox(data.M_C_11101, abs2.(matrixelC.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinDown(),), (SpinUp(),))))
    @test isapprox(data.M_C_11110, abs2.(matrixelC.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinDown(),), (SpinUp(),), (SpinUp(),)))) ##spis swapped
    ##swapped Cx (file) with BW function
    @test isapprox(data.M_Cx_11111, abs2.(matrixelBW.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))   
    @test isapprox(data.M_Cx_01111, abs2.(matrixelBW.(k_i, p_i, p1_i, p2_i, p3_i, (XPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))   
    @test isapprox(data.M_Cx_10111, abs2.(matrixelBW.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinDown(),), (SpinUp(),), (SpinUp(),), (SpinUp(),)))) 
    @test isapprox(data.M_Cx_11011, abs2.(matrixelBW.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinDown(),))))  ##spins swapped
    @test isapprox(data.M_Cx_11101, abs2.(matrixelBW.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinDown(),), (SpinUp(),)))) 
    @test isapprox(data.M_Cx_11110, abs2.(matrixelBW.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinDown(),), (SpinUp(),), (SpinUp(),))))  ##spis swapped
    ##swapped BW file with Cx function
    @test isapprox(data.M_BW_11111, abs2.(matrixelCx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))   
    @test isapprox(data.M_BW_01111, abs2.(matrixelCx.(k_i, p_i, p1_i, p2_i, p3_i, (XPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))   
    @test isapprox(data.M_BW_10111, abs2.(matrixelCx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinDown(),), (SpinUp(),), (SpinUp(),), (SpinUp(),)))) 
    @test isapprox(data.M_BW_11011, abs2.(matrixelCx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinDown(),))))  ##spins swapped
    @test isapprox(data.M_BW_11101, abs2.(matrixelCx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinDown(),), (SpinUp(),)))) 
    @test isapprox(data.M_BW_11110, abs2.(matrixelCx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinDown(),), (SpinUp(),), (SpinUp(),))))  ##spis swapped
        
    @test isapprox(data.M_BWx_11111, abs2.(matrixelBWx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))   
    @test isapprox(data.M_BWx_01111, abs2.(matrixelBWx.(k_i, p_i, p1_i, p2_i, p3_i, (XPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinUp(),))))   
    @test isapprox(data.M_BWx_10111, abs2.(matrixelBWx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinDown(),), (SpinUp(),), (SpinUp(),), (SpinUp(),)))) 
    @test isapprox(data.M_BWx_11011, abs2.(matrixelBWx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinUp(),), (SpinDown(),))))  ##spins swapped
    @test isapprox(data.M_BWx_11101, abs2.(matrixelBWx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinUp(),), (SpinDown(),), (SpinUp(),)))) 
    @test isapprox(data.M_BWx_11110, abs2.(matrixelBWx.(k_i, p_i, p1_i, p2_i, p3_i, (YPol(),), (SpinUp(),), (SpinDown(),), (SpinUp(),), (SpinUp(),))))  ##spis swapped
end

@testset "differential cross section" begin
    @testset "cpu" begin
        @test isapprox(data.diffCS, dσpT.(Ek_i, Ea_i, cta_i, phia_i, Eb_i, ctb_i)) 
    end

    @testset "gpu" begin
        @test isapprox(data.diffCS|>gpu, dσpT.(gEk_i, gEa_i, gcta_i, gphia_i, gEb_i, gctb_i))
    end

    @testset "not physical" begin
        @test isapprox(dσpT(Ek_i[1]*0.01, Ea_i[1], cta_i[1], phia_i[1], Eb_i[1], ctb_i[1]), eps())
        @test isapprox(dσpT((Ek_i[1]*0.01)|>gpu, Ea_i[1]|>gpu, cta_i[1]|>gpu, phia_i[1]|>gpu, Eb_i[1]|>gpu, ctb_i[1]|>gpu), eps()|>gpu)
        ###more tests
    end
end
