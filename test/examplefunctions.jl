using Cubature

rng = MersenneTwister(708583836976)

d = gpu
N = 10
xi  = rand(rng, Float64, 2, N)|>d


@testset "single_gauss" begin
    f = single_gauss
    ft = y-> exp.(-sum(y.^2,dims=1))

    @testset "maximum" begin
        @test isapprox(f([0.0,0.0])[1], 1.0)
    end

    @testset "batching, gpu, random points" begin
        @test isapprox(f(xi), ft(xi))
    end

    @testset "symmetry" begin
        @test isapprox(f(xi.*(repeat([-1.0,1.0],1,N)|>d)), f(xi))  # mirrored on y-axis
        @test isapprox(f(xi.*(repeat([1.0,-1.0],1,N)|>d)), f(xi))  # mirrored on x-axis
        @test isapprox(f(-xi), f(xi))                              # mirrored on both
    end

    #@testset "integral" begin
    #    @test isapprox(hcubature(x-> f(x)[1], [-100, 100], [-100, 100])[1], π) #broken
    #end
end


@testset "double_gauss" begin
    f = double_gauss
    ft = y-> exp.(-sum((y.-0.75) .^2 ./(0.2^2), dims=1)) +  exp.(-sum((y.-0.25) .^2 ./(0.2^2), dims=1))
    mirror = y -> y .* (repeat([-1.0,-1.0],1,N)|>d) .+ 1

    @testset "batching, gpu, random points" begin
        @test isapprox(f(xi), ft(xi))
    end

    @testset "symmetry" begin
        @test isapprox(f(mirror(xi)), f(xi))  # mirrored on the y=1-x line
    end

    #@testset "integral" begin
    #    @test isapprox(hcubature(x-> f(x)[1], [-100, 100], [-100, 100])[1], 2*π/(0.2^4)) #broken
    #end    
end
