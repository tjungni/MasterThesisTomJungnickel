using DrWatson, Test
@quickactivate :masterarbeit
using Random
using CUDA
using Flux

@testset "masterarbeit.jl" begin
    include("channelmappings.jl")
    include("examplefunctions.jl")  ### some tests are failing
    include("gpufunctions.jl")
    include("invsepmaps.jl")  ### need to look up some definitions 
    include("trident.jl")
    #include("couplinglayers.jl")
    #include("loss.jl")
end

### files skipped for now: backgrounds, transformations, compton
### CPU inputs not tested for now
