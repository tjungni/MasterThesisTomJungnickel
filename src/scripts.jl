using DrWatson
using Flux
using CUDA: rand
using Plots
using LaTeXStrings
using Statistics


function run_all(;mapping="identity", target="singlegauss", dim=2, dimA=1, loss="pearsonv3", nn_cons="nn1", omega=0.1, dphi=10.0, batchsize=4096, sample_batchsize=batchsize, epochs=1000, bins=10, learning_rate=0.0089, decay=0.5, steps=5, N_samples=2^20, optimizer=Adam, progress_plots=false, save=true, dir=get_mldir(), save_samples=true, comment="", masklayers=true, float_type=Float32)
    @info "Starting run with target=$target, omega=$omega, dphi=$dphi, batchsize=$batchsize, epochs=$epochs, bins=$bins, learning_rate=$learning_rate, decay=$decay, N_samples=$N_samples, optimizer=$(String(Symbol(optimizer))), progress_plots=$progress_plots"
    ENV["GKSwstype"] = "100"  # fixes the problem where running this script in the console with no display produces lots of errors when plotting
    
    starttime = time()
    start_learning_rate = learning_rate

    param_dict = Dict(
        # mappings
        "toomegabar" => HypercubeTocθωbar(),
        "identity" => IdentityMapping(),
        # targets
        "singlegauss" => single_gauss,
        "doublegauss" => double_gauss,
        "compton" => (x -> comptonf(x, omega, dphi)),
        # loss functions
        "pearson" => pearsonχ2divergence,
        "pearsonv2" => pearsonv2,
        "pearsonv3" => pearsonv3,
        "pearsonv4" => pearsonv4,
        "pearsonv5" => pearsonv5,
        # NN structure
        "nn1" => NN_constructor,   # old version of the NN
        "nn2" => n_NN,
        "nn3" => goetz_NN,  # with batchnorm
        "nn4" => NN4,
        "nn5" => NN5,
        "nn6" => NN6,
        "nn7" => NN7,
        "nn8" => NN8,
        "nn9" => NN9,
        "nn10" => NN10
        # more...optimizers
    )

    ytozmap = param_dict[mapping]
    if typeof(target) == String
        f = param_dict[target]
    else
        f = target
    end
    lossf = param_dict[loss]

    @info "Building model with 3 layers"
    if masklayers
        model_t = Chain(
            CouplingLayer(dim, dimA, bins, param_dict[nn_cons]),
            masterarbeit.MaskLayer([false, false, true, true, true]),
            CouplingLayer(dim, dimA, bins, param_dict[nn_cons]),
            masterarbeit.MaskLayer([false, false, true, true, true]),
            CouplingLayer(dim, dimA, bins, param_dict[nn_cons])#,
        )
        if float_type == Float32
            model = Flux.f32(model_t|> gpu)
        elseif float_type == Float64
            model = Flux.f64(model_t|> gpu)
        else
            println("Invalid data type specified! It should be Float32 or Float64.")
        end
    else
        cl1 = CouplingLayer(dim, dimA, bins, param_dict[nn_cons])
        cl2 = CouplingLayer(dim, dimA, bins, param_dict[nn_cons])
        cl3 = CouplingLayer(dim, dimA, bins, param_dict[nn_cons])
        sl = SwapLayer(dim, dimA)
        if float_type == Float32
            model = Flux.f32(Chain(
                cl1,
                sl,
                cl2,
                sl,
                cl3
            ) |> gpu)
        elseif float_type == Float64
            model = Flux.f64(Chain(
                cl1,
                sl,
                cl2,
                sl,
                cl3
            ) |> gpu)
        else
            println("Invalid data type specified! It should be Float32 or Float64.")
        end
    end


    @info "Compiling 1/4 - target function"
    xtest = rand(float_type, dim,batchsize)
    f(xtest)
    @info "Compiling 2/4 - model"
    model(xtest)
    @info "Compiling 3/4 - loss"
    lossf(model,ytozmap,f,xtest)
    @info "Compiling 4/4 - gradient of loss"
    Flux.withgradient(m-> lossf(m,ytozmap,f,xtest), model)


    @info "Training for $epochs epochs at $learning_rate learning rate, $decay decay, with batchsize $batchsize and optimizer $(String(Symbol(optimizer)))"
    t_train_start = time()
    if progress_plots
        losses = float_type[]
        loopslength = Int(epochs / steps)
        for i in 1:steps
            @info "Training step $i/$steps"
            losses = train_NN(model, dim, lossf, losses, ytozmap, f, epochs=loopslength, batchsize=batchsize, optimizer=optimizer, learning_rate=learning_rate, decay=decay)
            samples = sample_NN(model, ytozmap, dim, N_samples, batchsize)
            savefig(plot_samples(samples), joinpath(dir, "epoch$(i*loopslength)_samples.png"))
            learning_rate = learning_rate * decay
        end
    else
        losses = float_type[]
        losses = train_NN(model, dim, lossf, losses, ytozmap, f, epochs=epochs, batchsize=batchsize, optimizer=optimizer, decay=decay)
    end
    t_train_end = time()


    @info "Sampling"
    samples = sample_NN(model, ytozmap, dim, N_samples, sample_batchsize)

    @info "Calculating weights"
    wi = weights_chunked(model, dim, ytozmap, f, 10^7)[1,:]
    w_avg = mean(wi)
    w_max = maximum(wi)
    wi_n = wi ./ w_max
    w_avg_n = mean(wi_n)


    @info "Plotting losses"
    p1n = plot_loss(losses, false)
    p1l = plot_loss(losses, true)
    @info "Plotting samples"
    p2 = plot_samples(samples)
    @info "Plotting weights"
    p3 = plot_weights(wi_n)
    @info "Plotting ground truth"
    p4 = plot_truth(f, mapping)
    @info "Generating metrics"
    pearson_loss = pearsonv3(model, ytozmap, f, rand(float_type, dim, batchsize))


    if save
        @info "Saving parameters"
        save_params(dir, model)
        @info "Saving metadata"
        save_metadata(dir, model, target, dim, dimA, mapping, loss, nn_cons, bins, batchsize, optimizer, start_learning_rate, decay, steps, epochs, losses[end], pearson_loss, comment, w_max, w_avg, w_avg_n, float_type, N_samples)
        @info "Saving losses"
        save_object(joinpath(dir, "losses.jld2"), losses)
        if save_samples
            @info "Saving samples"
            save_object(joinpath(dir, "samples.jld2"), samples)
        end
        @info "Saving plots"
        savefig(p1n, joinpath(dir, "losses.png"))
        savefig(p1l, joinpath(dir, "losses_log.png"))
        savefig(p2, joinpath(dir, "sampling.png"))
        savefig(p3, joinpath(dir, "weights.png"))
        savefig(p4, joinpath(dir, "ground_truth.png"))
    else
        @info "Skipping saving"
    end

    endtime = time()
    @info "Done 😺"
    totaltime = round(endtime - starttime, digits=2)
    @info "The whole run took $totaltime s"  
    t_train = round(t_train_end - t_train_start, digits=2)
    @info "Training time: $t_train s"
    @info "The results have been saved under " * dir
end


function run_batch(target, dim, dimA)
    #a_epochs = [10, 20, 100, 500]#, 1000, 3000]
    a_batchsize = [16, 64, 256, 1024, 4096]
    a_bins = [5, 10, 15, 20, 25]
    a_learning_rate = [0.001, 0.005, 0.01, 0.2]
    a_decay_rate = [1.0, 0.8, 0.6, 0.4, 0.2]
    #a_optimizers = [Adam, Descent, Momentum, AdaMax, AdaGrad, AdamW]
    #a_layers = 
    params = Iterators.product(a_batchsize, a_bins, a_learning_rate)
    runs = length(params)
    i = 1
    batchdir = get_batchdir(target, dim)
    for (BA, BI, LE) in params
        @info "### Starting run $i/$runs ###"
        @info "target=$target, epochs=100, batchsize=$BA, bins=$BI, learning_rate=$LE"
        filename = "run=$(i)_batchsize=$(BA)_bins=$(BI)_learning_rate=$(LE)"
        dir = joinpath(batchdir, filename)
        mkpath(dir)
        run_all(target=target, dim=dim, dimA=dimA, epochs=250, dir=dir, batchsize=BA, bins=BI, learning_rate=LE, sample_batchsize=4096, N_samples=10^5, save_samples=false)
        i = i+1
    end
end


function run_batch_fixed_calls(calls)
    if (calls%4096) > 0
        println("Number of calls should be a multiple of 4096 so all runs can have exactly the same number of calls.")
        return
    end
    a_batchsize = [16, 64, 256, 1024, 4096]
    a_epochs = Int.(calls ./ a_batchsize)
    runs = length(a_batchsize)
    for i in eachindex(a_batchsize)
        @info "### Starting run $i/$runs ###"
        @info "target=3, epochs=$(a_epochs[i]), batchsize=$(a_batchsize[i])"
        run_all(target=3, epochs=a_epochs[i], batchsize=a_batchsize[i])
    end
end
