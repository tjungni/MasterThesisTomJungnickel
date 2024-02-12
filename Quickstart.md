# Quick start guide

## 1. Loading required packages
Inside this project folder, launch a Julia REPL 
```
julia --project=.
```
install dependencies
```julia
] instantiate
```
and load the package
```julia
using masterarbeit
using Flux
using CUDA
```
If you want to let a script do everything automatically, refer to #5. Otherwise, continue reading.


## 2. Creating a stack of coupling layers
You can create a coupling layer using `CouplingLayer()` with the following arguments:
- dimension of the input (and output)
- dimension of the first partition
- number of bins per dimension
- input swap flag, this is needed for every second layer
- constructor of the internal NN of the layer
and chain multiple layers together to create a NIS model.
```julia
cl1 = CouplingLayer(2, 1, 5, false, NN_constructor)
cl2 = CouplingLayer(2, 1, 5, true, NN_constructor)
cl3 = CouplingLayer(2, 1, 5, false, NN_constructor)
model = Chain(
    cl1,
    cl2,
    cl3
) |> gpu
```
The model is created on the CPU first, then `|> gpu` moves it to the GPU.

Now you can test the model by putting in some random data:
```julia
model(rand(2,6) |> gpu)

2×6 CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}:
 0.461126  0.54698   0.428786  0.024478  0.912112  0.586583
 0.254602  0.398903  0.727484  0.895326  0.961121  0.776751
```
You can process a batch of N d-dimensional data points at once by passing a dxN CuArray as input. 
Using a larger batch size will speed up the training process, but will also require more memory.  

## 3. Training
To start training, a ground truth function, a loss function and a map to the target domain are required.
If you want to use custom functions for those, it is important that they process batches of points as described above for the model and only use functions that are differentiable by Zygote.

### Map
Sometimes the data in the target domain needs to satisfy certain conditions like the on-shell condition of four-momenta.
In those cases, and when the output points need to be mapped from [0,1]^d to a different domain, you can define your own mapping function.
If no mapping is needed, you can use the identity map `IdentityMapping()`. More examples are provided in `src/ground_truth.jl`.

### Loss
To train the model, a loss function is required to evaluate how close the model's output is to the ground truth.
Some useful loss functions are provided in `src/losses.jl`, the loss `pearsonv2` is a good place to start.

!!! note 
    Since Zygote needs to be able to differentiate the used functions, some common operations (for example array manipulation) are not allowed and will lead to an error.

To test if everything is differentiable, you can run
```julia
xtest = rand(d, 10) |> gpu
Flux.withgradient(m-> lossf(m, map, groundtruth, xtest), model)
```


### Training
Now you are ready to train the model. You can either use a pre-defined training loop, or write your own.
(TODO: make losses input optional?)
```julia
map = IdentityMapping()
ground_truth = double_gauss
lossf = pearsonv2
losses = []

losses = train_NN(model, lossf, losses, map, ground_truth, epochs=100, batchsize=64, optimizer=Adam, learning_rate=0.001)
```

(TODO: manual training loop)

## 4. Generating data
With your trained model you can now generate data points that should resemble the ground truth.
```julia
sample_NN(model, map, N, batchsize)
```

## 5. Training and sampling script
You can use a provided script to train, sample and save the results all in just 1 command.
```julia
run_all(map="identity", target="singlegauss", batchsize=4096, epochs=1000, bins=10, N_samples=2^20)
```
For more arguments, refer to the documentation of the function. (link here?)
The results will be saved under plots/ml/[DATE]
