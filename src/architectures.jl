abstract type AbstractMachine end

function filtrations end

function (m::AbstractMachine)(y₀)
    forward, backward = filtrations(m, y₀)
    return solve(y₀, m.W, m.σ, (forward, backward))
end

sum_dims(dims::Tuple) = prod(dims[1:end-2]) * sum(dims[end-1:end])

# glorot initialization, from https://github.com/FluxML/Flux.jl
glorot_normal(dims...) = randn(Float32, dims...) .* sqrt(2.0f0 / sum_dims(dims))
glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum_dims(dims))

function consecutive_ranges(dims::Vector{Int})
    return map(cumsum(dims), dims) do cumdim, dim
        return (cumdim - dim + 1):cumdim
    end
end

function dense_filtrations(y, dims::Vector{Int})
    channels, minibatch = sum(dims), size(y, 2)
    shapes = map(rg -> (rg, 1:minibatch), consecutive_ranges(dims))
    input(_) = (1:channels, 1:minibatch)
    function project(input_shape, output_shape)
        return (first(input_shape), first(output_shape))
    end
    return (
        forward=Filtration(shapes, input, project),
        backward=Filtration(reverse(shapes), input, project)
    )
end

"""
    DenseMachine(W, σ, dims::Vector{Int})

Create a `DenseMachine` object from a square weight matrix `W`, pointwise nonlinearity `σ`
and filtration sequence `dims`.
The values of `dims` specify how to split the input space into a sequence of subspaces.
In particular, it is required that `size(W) == (sum(dims), sum(dims))`.
"""
struct DenseMachine{T, F} <: AbstractMachine
    W::T
    σ::F
    dims::Vector{Int}
end

@functor DenseMachine (W, σ)

DenseMachine(σ, dims::Vector{Int}; init=glorot_uniform) = DenseMachine(dims, σ; init)

"""
    DenseMachine(dims::Vector{Int}, σ; init=glorot_uniform)

Return a `DenseMachine(W, σ, dims)` object, where `W = init(sum(dims), sum(dims))`.
Default to Glorot uniform initialization.
"""
function DenseMachine(dims::Vector{Int}, σ; init=glorot_uniform)
    W = init(sum(dims), sum(dims))
    return DenseMachine(W, σ, dims)
end

filtrations(m::DenseMachine, y₀) = dense_filtrations(y₀, m.dims)

nest_pads(::Tuple{}) = ()

function nest_pads(pad::Tuple)
    a, b, tail... = pad
    return ((a, b), nest_pads(tail)...)
end

function conv_range(output_range, (pad0, pad1)::Tuple{Int,Int})
    i0, i1 = first(output_range), last(output_range)
    return (i0 - pad0):(i1 + pad1)
end

function conv_filtrations(y::AbstractArray{T, N}, dims::Vector{Int}; pad::NTuple{M, Int}) where {T, N, M}
    M == 2 * (N - 2) || throw(ArgumentError("padding must have twice the length of image dimensions"))
    pads = nest_pads(pad)
    
    ranges = consecutive_ranges(dims)
    channels, minibatch = sum(dims), size(y, N)
    shapes = [ntuple(n -> ifelse(n == N - 1, rg, UnitRange(axes(y, n))), N) for rg in ranges]

    function input(shape)
        data = ntuple(n -> shape[n], N - 2)
        zdata = map(conv_range, data, pads)
        return (zdata..., 1:channels, 1:minibatch)
    end

    function input′(shape)
        data = ntuple(n -> shape[n], N - 2)
        zdata = map(conv_range, data, map(reverse, pads))
        return (zdata..., 1:channels, 1:minibatch)
    end

    function project(input_shape, output_shape)
        return ntuple(N) do n
            n == N && return output_shape[end - 1]
            n == N - 1 && return input_shape[end - 1]
            p0, p1 = pads[n]
            return 1:(p0 + p1 + 1)
        end
    end

    return (
        forward=Filtration(shapes, input, project),
        backward=Filtration(reverse(shapes), input′, project)
    )
end

"""
    ConvMachine(W, σ, dims::Vector{Int}, pad::Dims)

Create a `ConvMachine` object from a weight array `W`, pointwise nonlinearity `σ`,
filtration sequence `dims` and padding `pad`.
The values of `dims` specify how to split the input space into a sequence of subspaces.
In particular, it is required that `size(W) == (kernelsize..., sum(dims), sum(dims))`,
where `kernelsize` is such that convolution by `W` with padding `pad` preserves
input dimension.
Padding `pad` has length twice the number of kernel dimensions (for example, it takes
four values for image convolutions).
"""
struct ConvMachine{T, F, M} <: AbstractMachine
    W::T
    σ::F
    dims::Vector{Int}
    pad::NTuple{M, Int}
end

@functor ConvMachine (W, σ)

ConvMachine(W, σ, dims; pad) = ConvMachine(W, σ, dims, pad)

ConvMachine(σ, dims::Vector{Int}; pad, init=glorot_uniform) = ConvMachine(dims, σ; pad, init)

"""
    ConvMachine(dims::Vector{Int}, σ; pad, init=glorot_uniform)

Return a `ConvMachine(W, σ, dims, pad)` object, where `W = init(kernelsize..., sum(dims), sum(dims))`.
Default to Glorot uniform initialization.
Here `kernelsize` is such that convolution with a kernel of size `kernelsize`
and padding `pad` preserves input dimension.
Padding `pad` has length twice the number of kernel dimensions (for example, it takes
four values for image convolutions).
"""
function ConvMachine(dims::Vector{Int}, σ; pad, init=glorot_uniform)
    kernelsize = map(((p0, p1),) -> p0 + p1 + 1, nest_pads(pad))
    W = init(kernelsize..., sum(dims), sum(dims))
    return ConvMachine(W, σ, dims; pad)
end

filtrations(m::ConvMachine, y₀) = conv_filtrations(y₀, m.dims; m.pad)

function recur_filtrations(y, dims::Vector{Int}; pad, timeblock)
    ranges = consecutive_ranges(dims)
    datalength, channels, minibatch = size(y, 1), sum(dims), size(y, 3)
    N = ceil(Int, datalength / timeblock)
    shapes = [(timeblock  * (n  - 1) + 1 : timeblock * n, range, 1:minibatch) for n in 1:N for range in ranges]

    input((timerange, _, mb)) =  (conv_range(timerange, (pad, 0)), 1:channels, mb)
    input′((timerange, _, mb)) =  (conv_range(timerange, (0, pad)), 1:channels, mb)
    function project(input_shape, output_shape)
        input_range, output_range = first(input_shape), first(output_shape)
        w0 = 1 + clamp(first(output_range) - last(input_range), 0:pad)
        w1 = 1 + clamp(last(output_range) - first(input_range), 0:pad)
        return (w0:w1, input_shape[end-1], output_shape[end-1])
    end
    flip = (pad+1):-1:1
    function project′(input_shape, output_shape)
        input_range, output_range = first(input_shape), first(output_shape)
        w0 = 1 + clamp(first(input_range) - last(output_range), 0:pad)
        w1 = 1 + clamp(last(input_range) - first(output_range), 0:pad)
        return (flip[w1]:flip[w0], input_shape[end-1], output_shape[end-1])
    end
    return (
        forward=Filtration(shapes, input, project),
        backward=Filtration(reverse(shapes), input′, project′)
    )
end

"""
    RecurMachine(W, σ, dims::Vector{Int}, pad::Int, timeblock::Int)

Create a `RecurMachine` object from a weight array `W`, pointwise nonlinearity `σ`,
filtration sequence `dims`, padding `pad` and time block `timeblock`.
The values of `dims` specify how to split the input space into a sequence of subspaces.
In particular, it is required that `size(W) == (pad + 1, sum(dims), sum(dims))`.
"""
struct RecurMachine{T, F} <: AbstractMachine
    W::T
    σ::F
    dims::Vector{Int}
    pad::Int
    timeblock::Int
end

@functor RecurMachine (W, σ)

RecurMachine(W, σ, dims; pad, timeblock) = RecurMachine(W, σ, dims, pad, timeblock)

RecurMachine(σ, dims::Vector{Int}; pad, timeblock, init=glorot_uniform) = RecurMachine(dims, σ; pad, timeblock, init)

"""
    RecurMachine(dims::Vector{Int}, σ; pad, timeblock, init=glorot_uniform)

Return a `RecurMachine(W, σ, dims, pad, timeblock)` object, where `W = init(pad + 1, sum(dims), sum(dims))`.
Default to Glorot uniform initialization.
"""
function RecurMachine(dims::Vector{Int}, σ; pad, timeblock, init=glorot_uniform)
    W = init(pad + 1, sum(dims), sum(dims))
    return RecurMachine(W, σ, dims; pad, timeblock)
end

filtrations(m::RecurMachine, y₀) = recur_filtrations(y₀, m.dims; m.pad, m.timeblock)

function ChainRulesCore.rrule(::typeof(filtrations), m::AbstractMachine, y)
    res = filtrations(m, y)
    function filtrations_pullback(_, _)
        return NoTangent(), NoTangent(), NoTangent()
    end
    return res, filtrations_pullback
end
