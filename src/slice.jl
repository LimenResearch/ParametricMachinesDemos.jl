function searchsortedrange(a, x)
    x0, x1 = extrema(x)
    i0 = searchsortedfirst(a, x0)
    i1 = searchsortedlast(a, x1)
    return i0:i1
end

# Only apply `f` in place within the shapes provided when they are all inbounds
function broadcast_within!(f, args...)
    idxs_within = map(args) do (array, idxs)
        return map(searchsortedrange, idxs, axes(array))
    end
    selected_idxs = map(intersect, idxs_within...)
    views = map(args) do (array, idxs)
        return view(array, map(getindex, idxs, selected_idxs)...)
    end
    broadcast!(f, views...)
    (target, _), _... = args
    return target
end

add_within!(t, args...) = broadcast_within!(+, t, t, args...)

"""
    slice!(buffer::AbstractVector, parent::AbstractArray, shape::Tuple{Vararg{AbstractRange}})

Non-allocating contiguous view of `parent` data at indices `shape`, which uses
memory reserved in `buffer`. Indices outside the axes of `parent` are set to `0`.
"""
function slice!(buffer::AbstractVector, parent::AbstractArray, shape::Tuple{Vararg{AbstractRange}})
    # maybe consider `slice = unsafe_wrap(Array, pointer(buffer), map(length, shape))` for performance
    trimmed = view(buffer, Base.OneTo(prod(length, shape)))
    slice = reshape(trimmed, map(length, shape))
    fill!(slice, 0)
    return broadcast_within!(identity, slice => axes(slice), parent => shape)
end

function slice(parent, shape)
    slice = similar(parent, map(length, shape))
    fill!(slice, 0)
    return broadcast_within!(identity, slice => axes(slice), parent => shape)
end

struct Slices{T, U, V}
    parent::T
    shapes::U
    buffer::V
end

function Slices(parent, shapes)
    l = maximum(shape -> prod(length, shape), shapes)
    buffer = similar(parent, l)
    return Slices(parent, shapes, buffer)
end

Base.getindex(s::Slices, i) = slice!(s.buffer, s.parent, s.shapes[i])

function selectchannels(y::AbstractArray{T, N}, rg::AbstractRange) where {T, N}
    shape = ntuple(n -> n == N - 1 ? rg : Colon(), N)
    return view(y, shape...)
end

function expandchannels(y₀::AbstractArray{T, N}, rg::AbstractRange) where {T, N}
    if axes(y₀, N - 1) ⊈ rg
        msg = "input has incompatible channels: $(axes(y₀, N - 1)) ⊈ $(rg)"
        throw(ArgumentError(msg))
    end
    shape = ntuple(n -> n == N - 1 ? rg : axes(y₀, n), N)
    return slice(y₀, shape)
end
