using masterarbeit
using Flux
using Random
using GLMakie
using LaTeXStrings
using Statistics
using Printf: @printf
using JLD2
using StatsBase

# arguments: epochs, steps, learning rate, decay, bins

fontsize_theme = Theme(fontsize=35)
set_theme!(fontsize_theme)
wblue = Makie.wong_colors()[1]
worange = Makie.wong_colors()[2]
wgreen = Makie.wong_colors()[3]
wpink = Makie.wong_colors()[4]
wlblue = Makie.wong_colors()[5]
worange = Makie.wong_colors()[6]
wyellow = Makie.wong_colors()[7]

function meminfo_julia()
    # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
    # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
    @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes()/2^20
    @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes()/2^20
    @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss()/2^20
end

starttime = time()
ENV["GKSwstype"] = "100"  # fixes the problem where running this script in the console with no display produces lots of errors when plotting
dir=get_mldir()

progress_plots=true
save_samples=true
save=true
comment="4cl, notebook subnet"

omega = 5.12
dphi = 10.0

ytozmap = trident_phasespace(omega)

function f(mapped)
    return dσpT_multithreaded((omega,), mapped...) .* 10000000f0
end

function jacobian_trident(m::Chain, cm::ChannelMapping, x::T) where {T <: AbstractArray{F}} where F<:Real
    cl1 = m[1]
    sl1 = m[2]
    cl2 = m[3]
    sl2 = m[4]
    cl3 = m[5]
    sl3 = m[6]
    cl4 = m[7]
    x2 = cl1(x)
    x2s = sl1(x2)
    x3 = cl2(x2s)
    x3s = sl2(x3)
    x4 = cl3(x3s)
    x4s = sl3(x4)
    det1 = abs.(masterarbeit.cldet_cpu(cl1,  x[cl1.dimA+1:cl1.d,:], cl1.m( x[1:cl1.dimA,:])...))
    det2 = abs.(masterarbeit.cldet_cpu(cl2, x2s[cl2.dimA+1:cl2.d,:], cl2.m(x2s[1:cl2.dimA,:])...))
    det3 = abs.(masterarbeit.cldet_cpu(cl3, x3s[cl3.dimA+1:cl3.d,:], cl3.m(x3s[1:cl3.dimA,:])...)) 
    det4 = abs.(masterarbeit.cldet_cpu(cl4, x4s[cl4.dimA+1:cl4.d,:], cl4.m(x4s[1:cl4.dimA,:])...)) 
    return abs(cmdet(cm)) .* det1 .* det2 .* det3 .* det4
end

function lossf(m::Chain, cm::ChannelMapping, f::Function, x::T) where T<:AbstractArray{F} where F<:Real
    zi = cm(m(x))
    g = 1 ./ jacobian_trident(m, cm, x)
    fz = f(zi)
    fracs = abs.(fz .- g) .^F(2.0) ./ fz
    return sum(fracs) / size(x,2)
end

dim = 5
dimA = 3
optimizer = Adam
activation = relu
batchsize = 1024
N_samples = 2^18  # = 260k
sample_batchsize = batchsize

epochs = 3#parse(Int, ARGS[1])#60
steps = 3#parse(Int, ARGS[2])#2
learning_rate = 0.01#parse(Float64, ARGS[3])#0.01
start_learning_rate = learning_rate
decay = 0.5 #parse(Float64, ARGS[4])#0.7
bins = 20 #parse(Int, ARGS[5])#10#20

@info "Starting trident cpu run with omega=$omega, dphi=$dphi, batchsize=$batchsize, epochs=$epochs, 
        steps=$steps, bins=$bins, learning_rate=$learning_rate, decay=$decay, N_samples=$N_samples, 
        optimizer=$(String(Symbol(optimizer))), activation=$activation, progress_plots=$progress_plots,
        subnet=notebook"

@info "Building model with 4 coupling layers"

function subnet(dimA::Signed, dimB::Signed, bins::Signed, width=32)
    return Chain(
        Split(
            Chain(
                BatchNorm(dimA),
                Dense(dimA => width, activation),
                BatchNorm(width),
                Dense(width => width, activation),
                BatchNorm(width),
                Dense(width => dimB*(bins+1))  
                ), 
            Chain(
                BatchNorm(dimA),
                Dense(dimA => width, activation),
                BatchNorm(width),
                Dense(width => width, activation),
                BatchNorm(width),
                Dense(width => dimB*bins)
                )
            ) 
        )
end

model = Flux.f64(Chain(
    CouplingLayerCPU(dim, dimA, bins, subnet),
    masterarbeit.MaskLayerCPU([false, false, true, true, true]),
    CouplingLayerCPU(dim, dimA, bins, subnet),
    masterarbeit.MaskLayerCPU([true, false, false, true, true]),
    CouplingLayerCPU(dim, dimA, bins, subnet),
    masterarbeit.MaskLayerCPU([false, true, false, true, true]),
    CouplingLayerCPU(dim, dimA, bins, subnet)
) |> cpu )

@info "Compiling 1/4 - target function"
xtest = Random.rand(dim, batchsize)
f(ytozmap(xtest))
@info "Compiling 2/4 - model"
model(xtest)
@info "Compiling 3/4 - loss"
lossf(model, ytozmap, f, xtest)
@info "Compiling 4/4 - gradient of loss"
Flux.withgradient(m -> lossf(m, ytozmap, f, xtest), model)

@info "Training for $epochs epochs at $learning_rate learning rate, $decay decay, with batchsize $batchsize and optimizer $(String(Symbol(optimizer)))"
t_train_start = time()
losses = Float64[]
if progress_plots
    loopslength = Int(epochs / steps)
    for i in 1:steps
        @info "Training step $i/$steps"
        global losses = train_NN_cpu(model, dim, lossf, losses, ytozmap, f, epochs=loopslength, batchsize=batchsize, optimizer=optimizer, learning_rate=learning_rate, decay=decay)
        #Ea_samples, cta_samples, phia_samples, Eb_samples, ctb_samples = sample_trident(model, ytozmap, dim, N_samples, batchsize)
        #savefig(plot_samples(Ea_samples, cta_samples, "Ea", "cos(theta_a)"), joinpath(dir, "epoch$(i*loopslength)_samples.png"))
        global learning_rate = learning_rate * decay
    end
else
    losses = train_NN_cpu(model, dim, lossf, losses, ytozmap, f, epochs=epochs, batchsize=batchsize, optimizer=optimizer, learning_rate=learning_rate, decay=decay)
end
t_train_end = time()

@info "Plotting losses"
fig = Figure(size=(1500,1000))
ax = Axis(fig[1,1], xlabel="epoch", ylabel="loss", yscale=log10, xlabelsize=50, ylabelsize=50)
lines!(1:length(losses), losses, linewidth=3, color=wblue, label="loss")
n = 10
lines!(n:length(losses), moving_average(losses, n), linewidth=4, color=worange, label="$n epoch \n moving average")
fig[1,2] = Legend(fig, ax)
save("trident_nis_loss_log.png", fig)
#save(joinpath(dir, "trident_nis_loss_log.png"), fig)

@info "Sampling"
t_sample_start = time()
samples = sample_nomap_cpu(model, dim, N_samples, sample_batchsize)
Ea_samples, cta_samples, phia_samples, Eb_samples, ctb_samples = ytozmap(samples)
t_sample_end = time()

function makie_samples(samplesx, samplesy, xname, yname)
    histo = fit(Histogram, (samplesx, samplesy), nbins=100)
    histo_n = StatsBase.normalize(histo, mode=:pdf)
    fig = Figure(size=(1200,1000), figure_padding=40)
    ax = Axis(fig[1,1], xlabel=latexstring(xname), ylabel=latexstring(xname), 
        aspect=1, xlabelsize=50, ylabelsize=50)
    hm = heatmap!(histo.edges[1], histo.edges[2], histo_n.weights)#, colorrange=(0,5), highclip=cgrad(:viridis)[end])
    fig[1, 2] = GridLayout(width = 20)
    Colorbar(fig[1,3], hm, width=40)
    #ylims=(0.0,1.0)
    return fig
end

@info "Plotting samples"
p2_Ea_cta = makie_samples(Ea_samples, cta_samples, "Ea", "cos(theta_a)") 
p2_phia_cta = makie_samples(phia_samples, cta_samples, "Ea", "phi_a") 
p2_Ea_Eb = makie_samples(Ea_samples, Eb_samples, "Ea", "Eb") 
save("sampling_Ea_cta.png", p2_Ea_cta)
savefig("sampling_phia_cta.png", p2_phia_cta)
savefig("sampling_Ea_Eb.png", p2_Ea_Eb)
#savefig(p2_Ea_cta, joinpath(dir, "sampling_Ea_cta.png"))
#savefig(p2_phia_cta, joinpath(dir, "sampling_phia_cta.png"))
#savefig(p2_Ea_Eb, joinpath(dir, "sampling_Ea_Eb.png"))



@info "Calculating weights"
function weights4cl(m::Chain, cm::ChannelMapping, f::Function, x::T) where {T <: AbstractArray}
    return jacobian_trident(m, cm, x) .* f(cm(m(x)))'
end

function weights4cl_chunked(m, dim, cm, f, N, batchsize)
    if (N%batchsize != 0) 
        x = Random.rand(dim, N%batchsize)
        weights = weights4cl(m, cm, f, x)
        runs = N ÷ batchsize 
    else
        x = Random.rand(dim,   batchsize)
        weights = weights4cl(m, cm, f, x)
        runs = N ÷ batchsize - 1
    end
    for i in 1:runs
        x = Random.rand(dim, batchsize)
        weights = hcat(weights, weights4cl(m, cm, f, x))
    end
    return weights
end

@info "Integrating"
f_evals = f(ytozmap(samples))[1,:]
nis_int = sum(f_evals) / size(samples,2) * ?
mcerror = sqrt(sum((f_evals .- mcint).^2) / (size(samples,2)-1))
println("mc integral = $nis_int")
println("standard deviation = $mcerror")

@info "Calculating weights"
wi = weights4cl_chunked(model, dim, ytozmap, f, N_samples, batchsize)[1, :]
wi_n = wi ./ w_nis_int
w_avg = mean(wi_n)
w_max = maximum(wi_n)
uw_eff = w_avg / w_max
@info "Unweighting efficiency = $uw_eff"

wi_filtered = wi_n[wi_normalized .< 10.01];
#=p3a = plot_weights(wi)
savefig(p3a, joinpath(dir, "weights_unnormalized.png"))
p3b = plot_weights(wi_n)
savefig(p3b, joinpath(dir, "weights_normalized.png"))
p3c = plot_weights(wi_filtered)
savefig(p3c, joinpath(dir, "weights_small.png"))=#

if save
    @info "Saving parameters"
    save_params(dir, model)
    @info "Saving metadata"
    save_metadata(dir, model, "trident", dim, dimA, "tridentmap", "see notebooks", "see_notebook", bins, batchsize, optimizer, start_learning_rate, decay, steps, epochs, losses[end], 0.0, comment, w_max, w_avg, w_avg_n, Float64, N_samples)
    @info "Saving losses"
    save_object(joinpath(dir, "losses.jld2"), losses)
    if save_samples
        @info "Saving samples"
        save_object(joinpath(dir, "samples.jld2"), samples)
    end    
else
    @info "Skipping saving"
end

endtime = time()
@info "Done 😺"
totaltime = round(endtime - starttime, digits=2)
@info "The whole run took $totaltime s"
t_train = round(t_train_end - t_train_start, digits=2)
@info "Training time: $t_train s"
t_sample = round(t_sample_end - t_sample_start, digits=2)
@info "Sampling time: $t_train s for $N_samples samples"
@info "The results have been saved under " * dir
