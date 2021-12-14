# Fast method to compute the derivative of `f` knowing both input and output
derivative(output::Number, f::F, input::Number) where {F} = only(only(derivatives_given_output(output, f, input)))
derivative(output::Number, ::typeof(identity), input::T) where {T<:Number} = one(T)

linear!(y, z, W::AbstractMatrix) = mul!(y, W', z)
linear!(y, z, W::AbstractArray) = conv!(y, z, W, DenseConvDims(z, W))
∇weights!(W::AbstractMatrix, z, ȳ₀) = mul!(W, z, ȳ₀')
∇weights!(W::AbstractArray, z, ȳ₀) = ∇conv_filter!(W, z, ȳ₀, DenseConvDims(z, W))

function flip(W::AbstractArray{T, N}) where {T, N}
    dims = ntuple(identity, N - 2)
    Ŵ = permutedims(W, (dims..., N, N - 1))
    for dim in dims # CUDA does not yet support multiple dims at once, see https://github.com/JuliaGPU/CUDA.jl/issues/1126
        reverse!(Ŵ, dims=dim)
    end
    return Ŵ
end

function clean!(W, project, output_shapes)
    N = length(output_shapes)
    for i0 in 1:N, i1 in i0:N
        W[project(output_shapes[i1], output_shapes[i0])...] .= 0
    end
    return W
end

# `project` must be guaranteed to work for `project(W, input(shape), shape)`
# where `shape` is either in `shapes` or is the whole `y`.
# It must also work for `project(W, shape1, shape2)` with `shape1, shape2 ∈ shapes`.
struct Filtration{N, I, P}
    shapes::Vector{NTuple{N, UnitRange{Int}}}
    input::I
    project::P
end

clean!(W, f::Filtration) = clean!(W, f.project, f.shapes)
clean(W, f::Filtration) = clean!(copy(W), f.project, f.shapes)

const Tensor = AbstractArray{<:Number}
istensor(y) = y isa Tensor

# If `y` is an array of numbers, then `y`, else `z`
macro iftensor(x, y, z=NoTangent())
    return Expr(:if, Expr(:call, :istensor, esc(x)), esc(y), esc(z))
end

apply!(σ, z, y, shape) = view(z, shape...) .+= σ.(view(y, shape...))
apply!(σ::Tensor, z, y, shape) = view(z, shape...) .+= view(σ, shape...) .* view(y, shape...)

# consider `y`, `z` as inputs and solve machine equation in place
function solve!(y, z, W, σ, f::Filtration)
    yshapes, zshapes = f.shapes, map(f.input, f.shapes)
    Wshapes = map(f.project, zshapes, yshapes)
    slices = map(Slices, (y, z, W), (yshapes, zshapes, Wshapes))
    for (i, yshape) in enumerate(yshapes)
        yslice, zslice, Wslice = getindex.(slices, i)
        # linear step (first iteration is trivial)
        add_within!(y => yshape, linear!(yslice, zslice, Wslice) => axes(yslice))
        # nonlinear step
        apply!(σ, z, y, map(intersect, yshape, axes(y)))
    end
    return
end

function init(y₀, z₀, W::AbstractArray{T, N}, forward) where {T, N}
    y = expandchannels(y₀, axes(W, N)) # `y₀` could have fewer channels
    z = isnothing(z₀) ? zero(y) : copy(z₀)
    W̃ = clean(W, forward)
    return (; y, z, W̃)
end

function solve(y₀::Tensor, z₀::Union{Tensor, Nothing}, W, σ, (forward, _))
    y, z, W̃ = init(y₀, z₀, W, forward)
    solve!(y, z, W̃, σ, forward)
    return (; y, z)
end

# simpler method that takes input before nonlinearity and returns output after nonlinearity
solve(y₀::Tensor, W, σ, (forward, backward)) = solve(y₀, nothing, W, σ, (forward, backward)).z

function ChainRulesCore.rrule(::typeof(solve), y₀::Tensor, z₀::Union{Tensor, Nothing}, W, σ, (forward, backward))
    y, z, W̃ = init(y₀, z₀, W, forward)
    solve!(y, z, W̃, σ, forward)
    function solve_pullback(ΔΩ)
        Dσ = @iftensor σ σ derivative.(z .- something(z₀, false), σ, y)
        W′ = flip(W̃)
        Ω̄ = unthunk(ΔΩ)
        ȳ = @iftensor Ω̄.y copy(Ω̄.y) zero(y) # autodiff may pass `AbstractZero`
        z̄ = @iftensor Ω̄.z copy(Ω̄.z) zero(z) # autodiff may pass `AbstractZero`
        solve!(z̄, ȳ, W′, Dσ, backward)
        ȳ₀, z̄₀ = selectchannels(ȳ, axes(y₀)[end-1]), @iftensor z₀ z̄ # Return cotangent comparable to input
        W̄ = ∇weights!(W′, slice(z, forward.input(axes(y))), ȳ) # Reuse `W′` as a buffer
        clean!(W̄, forward) # Compute Σᵢ (πᵢ - πᵢ₋₁) W̄ πᵢ₋₁
        σ̄ = @iftensor σ y .* z̄ # return `NoTangent()` if `σ` is an activation function
        return NoTangent(), ȳ₀, z̄₀, W̄, σ̄, NoTangent()
    end
    return (; y, z), solve_pullback
end
