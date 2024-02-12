using OneHotArrays

rng = MersenneTwister(708583836976)

d = gpu
N = 10
xi = rand(rng, Float64, 2, N)|>d

@testset "replace_nothing" begin
    f = replace_nothing
    badarray = [1 2 nothing 4 ; 5 nothing nothing 6]|>d
    
    @testset "pass input if it contains no none" begin
        @test isapprox(f.(xi, 10.0), xi)
    end

    @testset "type of output should not contain none anymore" begin
        @test typeof(f.(badarray, 10)) <: Union{CuArray{Int64},Matrix{Int64}}
    end
end


@testset "hotuptob" begin
    f = hotuptob
    b1 = [4;4;4]|>d
    b2 = [1;2;0;1]|>d

    @testset "all 1 up to b, all 0 after" begin
        @test isapprox(f(b1,10)[1:4,:], ones(4,3)|>d)
        @test isapprox(f(b1,10)[5:10,:], zeros(6,3)|>d)
    end

    @testset "column sum equals b-array" begin
        @test isapprox(sum(f(b2,5),dims=1), b2')
    end
end


@testset "onehotorzerobatch" begin
    f = onehotorzerobatch
    b0 = rand([1,2,3,4],10)|>d
    b1 = [1;2;3]|>d
    b2 = [1;2;0;1]|>d

    @testset "same as onehot for no zero" begin
        @test CUDA.@allowscalar isapprox(f(b0,5), onehotbatch(b0,1:5)*1)
    end

    @testset "columns have exactly one 1" begin
        @test isapprox(sum(f(b1,5), dims=1), [1 1 1]|>d)
    end

    @testset "0 input creates column of zeros" begin
        @test isapprox(sum(f(b2,5), dims=1), [1 1 0 1]|>d)
    end
end
