# ============================================================================
# The two reductions, written once: shared by the scalar functions, the drivers'
# kernels and the batch kernels (entries are read through ColumnView / RowView, so
# the same code runs on host arrays and inside GPU kernels)
# ============================================================================

# Entries `data[row, offset + s]` of one row of a matrix, s = 0, 1, …; unchecked like
# ColumnView (callers check axes first).
struct RowView{A}
    data::A
    row::Int
    offset::Int
end
@inline Base.getindex(r::RowView, s::Int) = @inbounds r.data[r.row, r.offset + s]

# Hemisphere moments of one (depth, channel) entry over orientations s = 0, …, n - 1.
@inline function _moment_sum(law, mu, weighted, n::Int, ::Type{FT}) where {FT}
    plus, minus = zero(FT), zero(FT)
    for s in 0:(n - 1)
        e = leaf_hemispheric_emission(law, mu[s], weighted[s], zero(FT))
        plus += convert(FT, e.upward)
        minus += convert(FT, e.downward)
    end
    return (; plus, minus)
end

# Up and down source strengths of one depth over channels s = 0, …, n - 1.
@inline function _bin_sum(plus, minus, front, back, n::Int, ::Type{FT}) where {FT}
    u, v = zero(FT), zero(FT)
    for s in 0:(n - 1)
        fc = convert(FT, front[s])
        bc = convert(FT, back[s])
        p_dc = convert(FT, plus[s])
        m_dc = convert(FT, minus[s])
        u += fc * p_dc + bc * m_dc
        v += fc * m_dc + bc * p_dc
    end
    return (; up = u, down = v)
end

"""
    hemispheric_moment(law::AbstractEmissionAngularLaw, mu, weighted, depth_idx, channel_idx)

Reduce orientation-weighted internal channel strength `weighted[:, depth_idx, channel_idx]`
with leaf-normal cosines `mu` under `law` into `(; plus, minus)` hemisphere
moments at `(depth_idx, channel_idx)`, respecting generic array axes.
"""
@inline function hemispheric_moment(
    law::AbstractEmissionAngularLaw,
    mu::AbstractVector,
    weighted::AbstractArray{<:Real,3},
    depth_idx::Integer,
    channel_idx::Integer,
)
    FT = common_scalar_type(_precision_operands(law)..., mu, weighted)
    @boundscheck begin
        axes(weighted, 1) == axes(mu, 1) ||
            throw(DimensionMismatch("weighted orientation axis must match axes(mu, 1)"))
        (depth_idx in axes(weighted, 2) && channel_idx in axes(weighted, 3)) ||
            throw(BoundsError(weighted, (:, depth_idx, channel_idx)))
    end
    o0 = firstindex(mu)
    return _moment_sum(
        law,
        ColumnView(mu, o0, ()),
        ColumnView(weighted, o0, (depth_idx, channel_idx)),
        length(mu),
        FT,
    )
end

"""
    hemispheric_bin(plus, minus, front, back, depth_idx)

Expand conditional channel escapes `front` and `back` against pre-integrated
hemisphere moments `plus[depth_idx, :]` and `minus[depth_idx, :]` into
`(; up, down)` hemispheric source strengths at `depth_idx`.
"""
@inline function hemispheric_bin(
    plus::AbstractMatrix,
    minus::AbstractMatrix,
    front::AbstractVector,
    back::AbstractVector,
    depth_idx::Integer,
)
    FT = common_scalar_type(plus, minus, front, back)
    @boundscheck begin
        axes(plus) == axes(minus) &&
            axes(front, 1) == axes(back, 1) == axes(plus, 2) ||
            throw(DimensionMismatch("hemispheric_bin plus, minus, front, and back axes must align"))
        depth_idx in axes(plus, 1) ||
            throw(BoundsError(plus, (depth_idx, :)))
    end
    c0 = firstindex(plus, 2)
    return _bin_sum(
        RowView(plus, depth_idx, c0),
        RowView(minus, depth_idx, c0),
        ColumnView(front, c0, ()),
        ColumnView(back, c0, ()),
        size(plus, 2),
        FT,
    )
end

@kernel function _hemispheric_moments_kernel!(
    plus,
    minus,
    law,
    mu,
    weighted,
    o0::Int,
    no::Int,
    d0::Int,
    c0::Int,
)
    d_rel, c_rel = @index(Global, NTuple)
    FT = eltype(plus)
    d_idx = d0 + d_rel - 1
    c_idx = c0 + c_rel - 1
    h = _moment_sum(law, ColumnView(mu, o0, ()), ColumnView(weighted, o0, (d_idx, c_idx)), no, FT)
    @inbounds begin
        plus[d_idx, c_idx] = h.plus
        minus[d_idx, c_idx] = h.minus
    end
end

const _OUTPUT_TYPE = "output element type must equal the promoted input scalar type"

# The CPU backend evaluates the scalar functions (hemispheric_moment, hemispheric_bin)
# in a loop; other backends launch one kernel. Same expressions, same order.
function _source_moments!(::KernelAbstractions.CPU, plus, minus, law, mu, weighted)
    FT = eltype(plus)
    @inbounds for c in axes(weighted, 3), d in axes(weighted, 2)
        h = hemispheric_moment(law, mu, weighted, d, c)
        plus[d, c] = convert(FT, h.plus)
        minus[d, c] = convert(FT, h.minus)
    end
    return nothing
end

function _source_moments!(backend, plus, minus, law, mu, weighted)
    k! = _hemispheric_moments_kernel!(backend)
    k!(
        plus,
        minus,
        law,
        mu,
        weighted,
        firstindex(mu, 1),
        length(mu),
        firstindex(plus, 1),
        firstindex(plus, 2);
        ndrange = size(plus),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    hemispheric_source_moments!(plus, minus, law::AbstractEmissionAngularLaw, mu, weighted)

Compute `(depth × channel)` hemisphere moments `plus` and `minus` from leaf
normal cosines `mu` (`[-1, 1]`) and nonnegative `(orientation × depth × channel)`
`weighted` internal sources across CPU, `OffsetArray`, and GPU arrays.
"""
function hemispheric_source_moments!(
    plus::AbstractMatrix{FT},
    minus::AbstractMatrix{FT},
    law::AbstractEmissionAngularLaw,
    mu::AbstractVector,
    weighted::AbstractArray{<:Real,3},
) where {FT<:Real}
    FT_req = common_scalar_type(plus, _precision_operands(law)..., mu, weighted)
    FT_req === FT || throw(ArgumentError(_OUTPUT_TYPE))
    axes(weighted, 1) == axes(mu, 1) &&
        axes(plus) == axes(minus) == (axes(weighted, 2), axes(weighted, 3)) ||
        throw(DimensionMismatch("Hemisphere moment axes must align with weighted[orientation, depth, channel]"))
    _disjoint_hemispheric_buffers((plus, minus), (mu, weighted))
    for x in _host_linear(mu)
        _checked(x, _cosine, "Leaf normal cosines mu must be finite and lie in [-1, 1]")
    end
    for x in _host_linear(weighted)
        _checked(x, _nonnegative, "Weighted internal source must be finite and nonnegative")
    end
    isempty(plus) && return plus, minus
    _source_moments!(KernelAbstractions.get_backend(plus), plus, minus, law, mu, weighted)
    return plus, minus
end

@kernel function _expand_hemispheric_kernel!(
    up,
    down,
    plus,
    minus,
    front,
    back,
    d0::Int,
    c0::Int,
    nc::Int,
)
    d_rel = @index(Global, Linear)
    FT = eltype(up)
    d_idx = d0 + d_rel - 1
    h = _bin_sum(
        RowView(plus, d_idx, c0),
        RowView(minus, d_idx, c0),
        ColumnView(front, c0, ()),
        ColumnView(back, c0, ()),
        nc,
        FT,
    )
    @inbounds begin
        up[d_idx] = h.up
        down[d_idx] = h.down
    end
end

function _expand_source!(::KernelAbstractions.CPU, up, down, plus, minus, front, back)
    FT = eltype(up)
    @inbounds for d in axes(plus, 1)
        h = hemispheric_bin(plus, minus, front, back, d)
        up[d] = convert(FT, h.up)
        down[d] = convert(FT, h.down)
    end
    return nothing
end

function _expand_source!(backend, up, down, plus, minus, front, back)
    k! = _expand_hemispheric_kernel!(backend)
    k!(
        up,
        down,
        plus,
        minus,
        front,
        back,
        firstindex(plus, 1),
        firstindex(plus, 2),
        size(plus, 2);
        ndrange = length(up),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    expand_hemispheric_source!(up, down, plus, minus, front, back)

Expand one emission bin's conditional face escapes `front` and `back` against
hemisphere moments `plus` and `minus` into depth-resolved hemispheric sources
`up` and `down` across CPU, `OffsetArray`, and GPU arrays.
"""
function expand_hemispheric_source!(
    up::AbstractVector{FT},
    down::AbstractVector{FT},
    plus::AbstractMatrix,
    minus::AbstractMatrix,
    front::AbstractVector,
    back::AbstractVector,
) where {FT<:Real}
    FT_req = common_scalar_type(up, plus, minus, front, back)
    FT_req === FT || throw(ArgumentError(_OUTPUT_TYPE))
    axes(plus) == axes(minus) &&
        axes(up) == axes(down) == (axes(plus, 1),) &&
        axes(front) == axes(back) == (axes(plus, 2),) ||
        throw(DimensionMismatch("Hemisphere expansion axes must align"))
    _disjoint_hemispheric_buffers((up, down), (plus, minus, front, back))
    for arr in (plus, minus, front, back)
        for x in _host_linear(arr)
            _checked(x, _nonnegative, "Hemisphere expansion inputs must be finite and nonnegative")
        end
    end
    isempty(up) && return up, down
    _expand_source!(KernelAbstractions.get_backend(up), up, down, plus, minus, front, back)
    return up, down
end


@kernel function _batch_moments_kernel!(
    plus,
    minus,
    law,
    mu,
    weighted,
    o0::Int,
    no::Int,
    d0::Int,
    c0::Int,
    b0::Int,
)
    d_rel, c_rel, b_rel = @index(Global, NTuple)
    FT = eltype(plus)
    d = d0 + d_rel - 1
    c = c0 + c_rel - 1
    b = b0 + b_rel - 1
    h = _moment_sum(law, ColumnView(mu, o0, ()), ColumnView(weighted, o0, (d, c, b)), no, FT)
    @inbounds begin
        plus[d, c, b] = h.plus
        minus[d, c, b] = h.minus
    end
end

"""
    batch_hemispheric_moments!(plus, minus, law, mu, weighted)

Compute batched `(n_depth × n_channel × n_batch)` hemisphere moments `plus` and
`minus` from `(n_orient × n_depth × n_channel × n_batch)` `weighted` sources on
CPU, `OffsetArray`, or GPU (`CuArray`).
"""
function batch_hemispheric_moments!(
    plus::AbstractArray{FT,3},
    minus::AbstractArray{FT,3},
    law::AbstractEmissionAngularLaw,
    mu::AbstractVector,
    weighted::AbstractArray{<:Real,4},
) where {FT<:Real}
    FT_req = common_scalar_type(plus, _precision_operands(law)..., mu, weighted)
    FT_req === FT || throw(ArgumentError(_OUTPUT_TYPE))
    axes(weighted, 1) == axes(mu, 1) &&
        axes(plus) == axes(minus) == (axes(weighted, 2), axes(weighted, 3), axes(weighted, 4)) ||
        throw(DimensionMismatch("Batch hemisphere moment dimensions or axes do not align"))
    _disjoint_hemispheric_buffers((plus, minus), (mu, weighted))
    isempty(plus) && return plus, minus
    backend = KernelAbstractions.get_backend(plus)
    k! = _batch_moments_kernel!(backend)
    k!(
        plus,
        minus,
        law,
        mu,
        weighted,
        firstindex(mu, 1),
        length(mu),
        firstindex(plus, 1),
        firstindex(plus, 2),
        firstindex(plus, 3);
        ndrange = size(plus),
    )
    KernelAbstractions.synchronize(backend)
    return plus, minus
end

@kernel function _batch_expand_sources_kernel!(
    up,
    down,
    plus,
    minus,
    front,
    back,
    d0::Int,
    m0::Int,
    c0::Int,
    nc::Int,
)
    d_rel, m_rel = @index(Global, NTuple)
    FT = eltype(up)
    d = d0 + d_rel - 1
    m = m0 + m_rel - 1
    h = _bin_sum(
        RowView(plus, d, c0),
        RowView(minus, d, c0),
        RowView(front, m, c0),
        RowView(back, m, c0),
        nc,
        FT,
    )
    @inbounds begin
        up[d, m] = h.up
        down[d, m] = h.down
    end
end

"""
    batch_expand_hemispheric_sources!(up, down, plus, minus, front, back)

Expand `(n_emit × n_channel)` conditional face tables `front` and `back` against
`(n_depth × n_channel)` hemisphere moments `plus` and `minus` into
`(n_depth × n_emit)` sources `up` and `down` on CPU, `OffsetArray`, or GPU
(`CuArray`).
"""
function batch_expand_hemispheric_sources!(
    up::AbstractMatrix{FT},
    down::AbstractMatrix{FT},
    plus::AbstractMatrix,
    minus::AbstractMatrix,
    front::AbstractMatrix,
    back::AbstractMatrix,
) where {FT<:Real}
    FT_req = common_scalar_type(up, plus, minus, front, back)
    FT_req === FT || throw(ArgumentError(_OUTPUT_TYPE))
    axes(plus) == axes(minus) &&
        axes(front) == axes(back) &&
        axes(plus, 2) == axes(front, 2) &&
        axes(up) == axes(down) == (axes(plus, 1), axes(front, 1)) ||
        throw(DimensionMismatch("Batch hemisphere source expansion dimensions or axes do not align"))
    _disjoint_hemispheric_buffers((up, down), (plus, minus, front, back))
    isempty(up) && return up, down
    backend = KernelAbstractions.get_backend(up)
    k! = _batch_expand_sources_kernel!(backend)
    k!(
        up,
        down,
        plus,
        minus,
        front,
        back,
        firstindex(up, 1),
        firstindex(up, 2),
        firstindex(plus, 2),
        size(plus, 2);
        ndrange = size(up),
    )
    KernelAbstractions.synchronize(backend)
    return up, down
end
