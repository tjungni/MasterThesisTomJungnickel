using DrWatson
using Dates
using Plots
using Plots: plot, plot!     # Workaround for bug in vscode linter
using JLD2
using TOML
using Flux
using Flux: train!
using LaTeXStrings
using ProgressMeter
using Random


"""
Extracts the weights/biases from the model and stores them in an array that contains 1 array for each layer, which again contain (currently) 4 (cpu-)Arrays, which correspond to weight,bias,weight,bias (because the layers are split into 2 paths with a dense each).
The model is stored under dir/[NAME].png
When loading, converting to CuArrays is necessary (which load_params() does automatically)
"""
function save_params(dir, m)
    modelparams = []
    for j in eachindex(m)
        if (j-1)%2 == 0
            paramj = []
            for i in 1:length(Flux.params(m[j].m))
                p = Flux.params(m[j].m)[i] |> cpu
                push!(paramj, p)
            end
            push!(modelparams, paramj)
        end
    end
    save_object(joinpath(dir, "modelparameters.jld2"), modelparams)
end

function architecture_dict(model)
    d = Dict{String, String}()
    for i in 1:length(model)
        d[string(i)] =  string(typeof(model[i]).name.wrapper)
    end
    return d
end


"""
Saves metadata in dir
"""
function save_metadata(dir, model, target, dim, dimA, mapping, loss, nn_cons, bins, batchsize, optimizer, learning_rate, decay, steps, epochs, finalloss, pearson_loss, comment, w_max, w_avg, w_avg_n, float_type, samples=0)
    metadata = Dict(
    "date" => now(),
    "commit" => gitdescribe(),
    "target_distribution" => target,
    "dimension" => dim,
    "dimensionA" => dimA,
    "channel_mapping" => mapping,
    "NN_parameters" => Dict(
        "bins" => bins,
        "NN_constructor" => nn_cons
    ),
    "training_parameters" => Dict(
        "loss" => loss,
        "batchsize" => batchsize,
        "epochs" => epochs,
        "optimizer" => String(Symbol(optimizer)),
        "learning_rate" => learning_rate,
        "decay_per_step" => decay,
        "steps" => steps
    ),
    "plot_parameters" => Dict(
        "samples" => samples
    ),
    "metrics" => Dict(
        "final_loss" => finalloss,
        "pearson_loss" => pearson_loss,
        "w_max" => w_max,
        "w_avg" => w_avg,
        "w_avg_normalized" => w_avg_n
        #"wasserstein" => wasserstein(???)
    ),
    "architecture" => architecture_dict(model),
    "float_type" => String(Symbol(float_type)),
    "comment" => comment
    )

    fname = joinpath(dir, "metadata.toml")

    open(fname, "w") do io
        TOML.print(io, metadata)
    end
end


"""
Returns the absolute path of the dir for ml output data, which a subfolder created for the current time
"""
function get_mldir()
    datestring = string(Date(now()))*"-"*string(hour(now()))*"-"*string(minute(now()))*"-"*string(second(now()))
    dir = joinpath(plotsdir(), "ml", datestring)
    mkpath(dir)
    return dir
end

function get_batchdir(target, dim)
    date_target = string(Date(now()))*"-"*string(hour(now()))*"-"*string(minute(now()))*"-"*string(second(now()))*"_"*target*"_$(dim)d"
    dir = joinpath(plotsdir(), "ml_batches", date_target)
    mkpath(dir)
    return dir
end


"""
Creates dir with current date&time and saves
- metadata
- model parameters
- samples
- plot
"""
function save_everything(plot, model, f, dim, dimA, ytozmap, loss, nn_cons, bins, batchsize, optimizer, learning_rate, losses, pearson_loss, samples)
    dir = get_mldir()

    epochs = length(losses)
    finalloss = losses[end]
    N_samples = size(samples, 2)

    save_metadata(dir, model, f, dim, dimA, ytozmap, loss, nn_cons, bins, batchsize, optimizer, learning_rate, epochs, finalloss, pearson_loss, N_samples)
    save_params(dir, model)
    save_object(joinpath(dir, "samples.jld2"), samples)
    savefig(plot, joinpath(dir, "allplots.png"))
    save_object(joinpath(dir, "samples.jld2"), samples)
    save_object(joinpath(dir, "losses.jld2"), losses)
end


"""
Overwrites the weights/biases in m with ones stored using store_params() in the file plots/ml/[DATE]/[COMMIT]/[NAME].png, creates the direcory if it does not exist already.
"""
function load_params(m, path)
    modelparams = load_object(path)
    for j in eachindex(modelparams)
        for i in eachindex(modelparams[j])
            if j==1
                jf = 1
            else
                jf = (j-1)*2+1
            end
            Flux.params(m[jf].m)[i] .= CuArray(modelparams[j][i])
        end
    end
end

function load_params_cpu(m, path)
    modelparams = load_object(path)
    for j in eachindex(modelparams)
        for i in eachindex(modelparams[j])
            if j==1
                jf = 1
            else
                jf = (j-1)*2+1
            end
            Flux.params(m[jf].m)[i] .= modelparams[j][i]
        end
    end
end

function load_params(m, date, commit, name)
    path = load_object(joinpath(plotsdir(), "ml", date, commit, name))
    load_params(m, path)
end

#storeparams = collect(Flux.params(NN1))
#Flux.loadparams!(model, storeparams)


function train_NN(model::Chain, dim, lossf, losses, ytozmap, f; epochs=300, batchsize=16384, optimizer=Adam, learning_rate=0.0089, decay=0.05, ftype=Float32)
    # losses is an input because this way train_NN can be run multiple times (with different training parameters) and all losses be collected
    opt_state = Flux.setup(optimizer(learning_rate), model)
    @showprogress for epoch in 1:epochs
        data = CUDA.rand(ftype, dim, batchsize)
        val, grads = Flux.withgradient(
        m-> lossf(m,ytozmap,f,data), model
        )

        # Save the loss from the forward pass. (Done outside of gradient.)
        push!(losses, val)

        Flux.update!(opt_state, model, grads[1])
    end
    return losses
end

function train_NN_cpu(model::Chain, dim, lossf, losses, ytozmap, f; epochs=300, batchsize=16384, optimizer=Adam, learning_rate=0.0089, decay=0.05, ftype=Float64)
    opt_state = Flux.setup(optimizer(learning_rate), model)
    @showprogress for epoch in 1:epochs
        data = Random.rand(ftype, dim, batchsize)
        val, grads = Flux.withgradient(
        m-> lossf(m,ytozmap,f,data), model
        )

        push!(losses, val)

        Flux.update!(opt_state, model, grads[1])
    end
    return losses
end

function train_decay(model::Chain, dim, lossf, losses, ytozmap, f; epochs=300, batchsize=16384, optimizer=Adam, learning_rate=0.0089, decay=0.99)
    @showprogress for epoch in 1:epochs 
        opt_state = Flux.setup(optimizer(learning_rate), model)
        data = CUDA.rand(Float64, dim, batchsize)
        val, grads = Flux.withgradient(
        m-> lossf(m,ytozmap,f,data), model
        )

        push!(losses, val)

        Flux.update!(opt_state, model, grads[1])
        learning_rate = learning_rate * decay
    end
    return losses
end

function train_adjust(model::Chain, dim, lossf, losses, ytozmap, f; epochs=300, batchsize=16384, optimizer=Adam, learning_rate=0.0089, decay=0.99)
    st = Optimisers.setup(optimizer(learning_rate), model)
    @showprogress for epoch in 1:epochs 
        
        data = rand(Float64, dim, batchsize)
        val, grads = Flux.withgradient(
        m-> lossf(m,ytozmap,f,data), model
        )

        push!(losses, val)

        Optimisers.update!(st, model, grads[1])
        learning_rate = learning_rate * decay
        Optimisers.adjust!(st[1], learning_rate)
    end
    return losses
end

function sample_NN(m, ytozmap, dim, N, batchsize; ftype=Float32)
    if (N%batchsize != 0) 
        samples = ytozmap(m(CUDA.rand(ftype, dim, N%batchsize)))
        runs = N ÷ batchsize 
    else
        samples = ytozmap(m(CUDA.rand(ftype, dim,   batchsize)))
        runs = N ÷ batchsize - 1
    end
    for i in 1:runs
        x_random = CUDA.rand(ftype, dim, batchsize)
        samples = hcat(samples, ytozmap(m(x_random)) )
    end
    return samples |> cpu
end


function sample_trident(m, ytozmap, dim, N, batchsize)
    if (N%batchsize != 0) 
        Ea, cta, phia, Eb, ctb = ytozmap(m(Random.rand(Float64, dim, N%batchsize)))
        runs = N ÷ batchsize 
    else
        Ea, cta, phia, Eb, ctb = ytozmap(m(Random.rand(Float64, dim,   batchsize)))
        runs = N ÷ batchsize - 1
    end
    for i in 1:runs
        x_random = Random.rand(Float64, dim, batchsize)
        Ean, ctan, phian, Ebn, ctbn = ytozmap(m(x_random))
        Ea = vcat(Ea, Ean)
        cta = vcat(cta, ctan)
        phia = vcat(phia, phian)
        Eb = vcat(Eb, Ebn)
        ctb = vcat(ctb, ctbn)
    end
    return Ea, cta, phia, Eb, ctb
end

function weights_chunked(m, dim, cm, f, N, batchsize; ftype=Float32)
    if (N%batchsize != 0) 
        x = CUDA.rand(ftype, dim, N%batchsize)
        weights = jacobian(m, cm, x) .* f(cm(m(x))) |> cpu
        runs = N ÷ batchsize 
    else
        x = CUDA.rand(ftype, dim,   batchsize)
        weights = jacobian(m, cm, x) .* f(cm(m(x))) |> cpu
        runs = N ÷ batchsize - 1
    end
    for i in 1:runs
        x = CUDA.rand(ftype, dim, batchsize)
        weights = hcat(weights, jacobian(m, cm, x) .* f(cm(m(x))) |> cpu)
    end
    return weights
end

function sample_nomap(m, dim, N, batchsize; ftype=Float32)
    if (N%batchsize != 0) 
        samples = m(CUDA.rand(ftype, dim, N%batchsize)) |> cpu
        runs = N ÷ batchsize 
    else
        samples = m(CUDA.rand(ftype, dim,   batchsize)) |> cpu
        runs = N ÷ batchsize - 1
    end
    for i in 1:runs
        x_random = CUDA.rand(ftype, dim, batchsize)
        samples = hcat(samples, m(x_random) |> cpu )
    end
    return samples
end

function sample_nomap_cpu(m, dim, N, batchsize; ftype=Float64)
    if (N%batchsize != 0) 
        samples = m(Random.rand(ftype, dim, N%batchsize))
        runs = N ÷ batchsize 
    else
        samples = m(Random.rand(ftype, dim,   batchsize))
        runs = N ÷ batchsize - 1
    end
    for i in 1:runs
        x_random = Random.rand(ftype, dim, batchsize)
        samples = hcat(samples, m(x_random))
    end
    return samples
end

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

function plot_loss(losses, log=false)
    p1 = plot(losses; xaxis=("iteration"), yaxis="loss", label="per batch", title="training", dpi=200, size=(500,500))
    if log
        yaxis!(:log10) 
    end
    n_avg = 10
    plot!(n_avg:length(losses), moving_average(losses, n_avg), label="$n_avg epoch moving average") 
    return p1
end

function plot_samples(samples)
    p2 = plot(
        samples[1,:],
        samples[2,:],
        seriestype = :histogram2d,
        c = :thermal,
        nbins = 100,
        show_empty_bins = :true,
        size=(500,500),
        title="trained NN",
        ylims=(0.0,1.0)
    )
    return p2
end
    

function plot_samples(samplesx, samplesy, xname, yname)
    p2 = plot(
        samplesx,
        samplesy,
        seriestype = :histogram2d,
        c = :thermal,
        nbins = 100,
        show_empty_bins = :true,
        size=(500,500),
        title="trained NN",
        xlabel=xname,
        ylabel=yname
    )
    return p2
end

function plot_weights(weights)
    p2 = histogram(
        weights,
        c = :thermal,
        bins = 100,
        size=(500,500),
        title="normalized weights"
    )
    return p2
end

# for plotting the moving average of training loss
moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
