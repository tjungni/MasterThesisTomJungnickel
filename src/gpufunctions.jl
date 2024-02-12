## Some functions that can be useful when writing code for the GPU which doesn't allow loops, scalar indexing and more

using CUDA


"""
When findfirst() is broadcast over a CuArray and the target isn't found in some columns then 'nothing' is put into the output CuArray which makes its datatype unstable.
This function replaces the nothings with the max value.

This functions needs the be broadcast!
    replace_nothing.(CuArray)
"""
function replace_nothing(x, max)  
    if x===nothing
        return max
    else
        return x
    end
end


"""
This function creates columns that are ones up to a the index b, then zero until a length of l is reached.
This can be useful to achieve 'iteration' over specific ranges of a CuArray through matrix multiplication.

Example:
V = 
10  20  30
1   2   3
0.1 0.2 0.3
100 200 300
b =  [2; 4; 0]  ### why are the ; needed??

Now The calculation sum(x[1:b]) should be computed for x which are cuts of each column with a length defined by the values in b.
The result is then
sum(V .* hotuptob([2; 4; 0]|>gpu, size(V,1)), dims=1) = [11.0  222.2  0.0]
"""
function hotuptob(b, l)
    is = repeat(b,1,l)'
    js = CuArray(repeat(1:l,1,length(b)))
    return is .>= js
end

function hotuptob_cpu(b, l)
    is = repeat(b,1,l)'
    js = repeat(1:l,1,length(b))
    return is .>= js
end


"""
The function basically does the some thing as onehotbatch from OneHotArrays but it allows 0 in the input array, which makes the corresponding column all zeros with no one.
b defines where the ones should be in each column and l sets the length of all colums.

onehotorcoldbatch([1, 3, 0], 4) = 
1  0  0
0  0  0
0  1  0
0  0  0

Even if no index is zero this function should be used as a replacement for onehotbatch() because it will be faster! (since it actually runs on the gpu unlike onehotbatch)
"""
function onehotorzerobatch(b, l)
    is = repeat(b,1,l)'
    js = CuArray(repeat(1:l,1,length(b)))
    return is .== js
end

function onehotorzerobatch_cpu(b, l)
    is = repeat(b,1,l)'
    js = repeat(1:l,1,length(b))
    return is .== js
end
