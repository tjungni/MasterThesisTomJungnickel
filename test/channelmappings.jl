
rng = MersenneTwister(708583836976)

d = gpu

x_2d  = [rand(rng, Float64, 2, 1)|>d, rand(rng, Float64, 2, 100)|>d, zeros(Float64, 2, 100)|>d,    ones(Float64, 2, 100)|>d]
x_all = [rand(rng, Float64, 2, 1)|>d, rand(rng, Float64, 2, 100)|>d, rand(rng, Float64, 10, 1)|>d, rand(rng, Float64, 10, 100)|>d, zeros(Float64, 10, 100)|>d, ones(Float64, 10, 100)|>d]
#_all = [1 2d vector                , 100 2d vectors               , 1 10d vector                , 100 10d vectors               , 100 10d 0.0 vectors       , 100 10d 1.0 vectors      ]

@testset "identity mapping" begin
    map = IdentityMapping()
    @testset "$x" for x in x_all
        @test isapprox(map(x), x)
    end
    @testset "determinant" begin
        @test isapprox(cmdet(map), 1.0)
    end
end


@testset "cθωbar" begin
    map = HypercubeTocθωbar()
    @testset "$x" for x in x_2d
        @test isapprox(map(x)[1,:], x[1,:].*2.0.-1.0)
        @test isapprox(map(x)[2,:], x[2,:])
    end
    @testset "determinant" begin
        @test isapprox(cmdet(map), 2.0)
    end
end
