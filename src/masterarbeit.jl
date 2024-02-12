module masterarbeit

# constants
export m_e, me_sq, α, e_charge, consts, consts2, zero_i, one_i, two_i, half_i, pi_sq
# compton
export calc_ω2, t, t_ex, dσdcθ
export lofω2, dσdldcθ, dσdldt, dσdω2dcθ
export calc_ω2ex, dσdω2dcθ_ex
export cossquare
# mappings
export cmdet
export AffineTransform
export ChannelMapping, IdentityMapping, HypercubeTocθωbar
export get_E_lim, trident_phasespace
# targets
export bottomleftsquare, single_gauss, double_gauss, comptonf
# gpu tools
export replace_nothing, hotuptob, onehotorzerobatch
# nis
export InvSepMap, PiecewiseQuadratic, normalizeVW, calculate_ab
export cldet, weights, pearsonχ2divergence, pearsonv2, pearsonv3, pearsonv4, pearsonv5
export CouplingLayer, SwapLayer, Split, MaskLayer, NN_constructor, n_NN, o_NN
export CouplingLayerCPU, MaskLayerCPU
# notebook tools
export save_params, save_metadata, get_mldir, get_batchdir, save_everything, load_params, load_params_cpu, train_NN, train_NN_cpu, train_decay, sample_NN, sample_trident, plot_loss, plot_samples, plot_weights, plot_truth
export sample_nomap, sample_nomap_cpu
export run_all, run_batch, run_batch_fixed_calls
export weights_chunked
export moving_average
# trident
export SpinUp, SpinDown, FourPolarisation2, mem, XPol, YPol
export matrixelC, matrixelBW, matrixelCx, matrixelBWx
export MpT
export check_physics, get_omega, get_ss
export dσpT, dσpT_multithreaded


include("compton.jl")
include("backgrounds.jl")
include("targets.jl")
include("gpufunctions.jl")
include("invsepmaps.jl")
include("channelmappings.jl")
include("notebookfunctions.jl")
include("couplinglayers.jl")
include("loss.jl")
include("scripts.jl")
include("trident.jl")

end