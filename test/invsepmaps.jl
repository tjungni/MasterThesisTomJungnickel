using OneHotArrays

rng = MersenneTwister(708583836976)

d = gpu
N = 10
xi  = rand(rng, Float64, 2, N)|>d

@testset "normalizeVW" begin
    f  = normalizeVW
    Vh = Float32[-0.00015171438 -0.0021605862; 0.007835436 0.11158556; -0.0031442386 -0.044777554; 0.01974211 0.28115022; -0.013956764 -0.19876027]|>d
    Wh = Float32[-0.0005070344 -0.0072207493; 0.012016733 0.17113201; 0.0193653 0.27578402; -0.0028581806 -0.04070376]|>d   # bins=4, samples=2
    V  = Float32[0.99543947 0.92404944; 1.003422 1.0353675; 0.992465 0.88549656; 1.0154408 1.2266918; 0.98179173 0.7591246]|>d
    W  = Float32[0.24811895 0.22274546; 0.25124586 0.26623583; 0.25309896 0.29560804; 0.24753627 0.21541074]|>d
    Vout, Wout = f(Vh,Wh,4,1)  # dimb=1 because V/W are 2 wide and samples=2, width=samples*dimB

    @testset "match known result" begin
        @test isapprox(Vout, V) 
        @test isapprox(Wout, W)
    end

    @testset "W sums to 1" begin
        @test isapprox(sum(Wout, dims=1), ones(Float32, 1, 2)|>d)
    end

    # no sum test for V because the sum varies with a value around the number of bins
end


@testset "calculate_ab" begin
    f  = calculate_ab
    xB = [0.2 0.56]|>d
    W  = [0.1 0.35 ; 0.4 0.2 ; 0.5 0.45]|>d # bins=3
    
    @testset "correct a and b" begin
        a, b = f(xB,W)
        @test isapprox(a, [0.25 1/45]|>d)
        @test isapprox(b, [2 ; 3]|>d)
    end

    @testset "xB=0" begin
        x0 = [0.0 eps()]|>d
        a, b = f(x0,W)
        @test isapprox(a, [0.0 0.0]|>d, atol=1e-15)
        @test isapprox(b, [1 ; 1]|>d)
        @test eltype(b) <: Integer   # check that b doesn't contain 'nothing'
    end

    @testset "xB=1" begin
        x0 = [1.0 1.0-eps()]|>d 
        a, b = f(x0,W)
        @test isapprox(a, [1.0 1.0]|>d, atol=10*eps(eltype(a)))  ###TODO: remove the x10 and fix
        @test isapprox(b, [3 ; 3]|>d)
        @test eltype(b) <: Integer
    end

    @testset "x at edge" begin
        xB = [0.1 0.35]|>d
        a, b = f(xB,W)
        @test isapprox(a, [0.0 0.0]|>d) || isapprox(a, [1.0 1.0]|>d)
        @test isapprox(b, [1 ; 1]|>d) || isapprox(b, [2 ; 2]|>d)
    end
end

#=
@testset "piecewise quadratic" begin
    pwq = PiecewiseQuadratic
    xB = 
    V = 
    W = 
    ndim = 

    @testset "known results" begin
        
    end
end
=#
