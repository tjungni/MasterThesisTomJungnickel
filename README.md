# Master Thesis of Tom Jungnickel

This repository contains the code used to produce the results in the master's thesis of Tom Jungnickel.

This version of the code was intended for internal use. Therefore, many functions might be undocumented or not used anymore.
Variables/functions are often weirdly inconsistently named.
When saving or loading files, names or paths might need to be adjusted.

Notebooks are provided as an easy way to run the code. 
Since these notebooks generate plots, the package Makie.jl needs to be installed and loaded in addition to this package. 
It is not included by default due to its long precompilation time and problems that occurred when running it on some servers.

## DrWatson

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> masterarbeit

It is authored by Tom Jungnickel.

To (locally) reproduce this project, do the following:

0. Install Julia version 1.9.4.
1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "masterarbeit"
```
which auto-activate the project and enable local path handling from DrWatson.
