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
    FT = promote_compatible_scalar(
        hemispheric_moment,
        _law_scalar_probe(law, zero(eltype(mu))),
        zero(eltype(mu)),
        zero(eltype(weighted)),
    )
    @boundscheck begin
        axes(weighted, 1) == axes(mu, 1) ||
            throw(DimensionMismatch("weighted orientation axis must match axes(mu, 1)"))
        (depth_idx in axes(weighted, 2) && channel_idx in axes(weighted, 3)) ||
            throw(BoundsError(weighted, (:, depth_idx, channel_idx)))
    end
    plus, minus = zero(FT), zero(FT)
    @inbounds for o in eachindex(mu)
        e = leaf_hemispheric_emission(law, mu[o], weighted[o, depth_idx, channel_idx], zero(FT))
        plus += convert(FT, e.upward)
        minus += convert(FT, e.downward)
    end
    return (; plus, minus)
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
    FT = promote_compatible_scalar(
        hemispheric_bin,
        zero(eltype(plus)),
        zero(eltype(minus)),
        zero(eltype(front)),
        zero(eltype(back)),
    )
    @boundscheck begin
        axes(plus) == axes(minus) &&
            axes(front, 1) == axes(back, 1) == axes(plus, 2) ||
            throw(DimensionMismatch("hemispheric_bin plus, minus, front, and back axes must align"))
        depth_idx in axes(plus, 1) ||
            throw(BoundsError(plus, (depth_idx, :)))
    end
    u, v = zero(FT), zero(FT)
    @inbounds for c in axes(plus, 2)
        fc = convert(FT, front[c])
        bc = convert(FT, back[c])
        p_dc = convert(FT, plus[depth_idx, c])
        m_dc = convert(FT, minus[depth_idx, c])
        u += fc * p_dc + bc * m_dc
        v += fc * m_dc + bc * p_dc
    end
    return (; up = u, down = v)
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
    p_acc = zero(FT)
    m_acc = zero(FT)
    @inbounds for o_step in 0:(no - 1)
        o_idx = o0 + o_step
        e = leaf_hemispheric_emission(law, mu[o_idx], weighted[o_idx, d_idx, c_idx], zero(FT))
        p_acc += convert(FT, e.upward)
        m_acc += convert(FT, e.downward)
    end
    @inbounds begin
        plus[d_idx, c_idx] = p_acc
        minus[d_idx, c_idx] = m_acc
    end
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
    require_continuous_scalar(FT, hemispheric_source_moments!, (plus, minus, law, mu, weighted))
    FT_req = promote_compatible_scalar(
        hemispheric_source_moments!,
        zero(FT),
        _law_scalar_probe(law, zero(FT)),
        zero(eltype(mu)),
        zero(eltype(weighted)),
    )
    FT_req === FT ||
        throw(MethodError(hemispheric_source_moments!, (plus, minus, law, mu, weighted)))
    axes(weighted, 1) == axes(mu, 1) &&
        axes(plus) == axes(minus) == (axes(weighted, 2), axes(weighted, 3)) ||
        throw(DimensionMismatch("Hemisphere moment axes must align with weighted[orientation, depth, channel]"))
    _disjoint_hemispheric_buffers((plus, minus), (mu, weighted))
    P = primal_float_type(FT)
    for x in _host_linear(mu)
        finite_scalar(x) && -one(P) <= primal_scalar(x) <= one(P) ||
            throw(ArgumentError("Leaf normal cosines mu must be finite and lie in [-1, 1]"))
    end
    for x in _host_linear(weighted)
        finite_scalar(x) && primal_scalar(x) >= zero(P) ||
            throw(ArgumentError("Weighted internal source must be finite and nonnegative"))
    end
    isempty(plus) && return plus, minus
    backend = KernelAbstractions.get_backend(plus)
    if backend isa KernelAbstractions.CPU
        @inbounds for c in axes(weighted, 3), d in axes(weighted, 2)
            h = hemispheric_moment(law, mu, weighted, d, c)
            plus[d, c] = convert(FT, h.plus)
            minus[d, c] = convert(FT, h.minus)
        end
    else
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
    end
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
    u_acc = zero(FT)
    v_acc = zero(FT)
    @inbounds for c_step in 0:(nc - 1)
        c_idx = c0 + c_step
        fc = convert(FT, front[c_idx])
        bc = convert(FT, back[c_idx])
        p_dc = convert(FT, plus[d_idx, c_idx])
        m_dc = convert(FT, minus[d_idx, c_idx])
        u_acc += fc * p_dc + bc * m_dc
        v_acc += fc * m_dc + bc * p_dc
    end
    @inbounds begin
        up[d_idx] = u_acc
        down[d_idx] = v_acc
    end
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
    require_continuous_scalar(
        FT,
        expand_hemispheric_source!,
        (up, down, plus, minus, front, back),
    )
    FT_req = promote_compatible_scalar(
        expand_hemispheric_source!,
        zero(FT),
        zero(eltype(plus)),
        zero(eltype(minus)),
        zero(eltype(front)),
        zero(eltype(back)),
    )
    FT_req === FT ||
        throw(MethodError(expand_hemispheric_source!, (up, down, plus, minus, front, back)))
    axes(plus) == axes(minus) &&
        axes(up) == axes(down) == (axes(plus, 1),) &&
        axes(front) == axes(back) == (axes(plus, 2),) ||
        throw(DimensionMismatch("Hemisphere expansion axes must align"))
    _disjoint_hemispheric_buffers((up, down), (plus, minus, front, back))
    P = primal_float_type(FT)
    for arr in (plus, minus, front, back)
        for x in _host_linear(arr)
            finite_scalar(x) && primal_scalar(x) >= zero(P) ||
                throw(ArgumentError("Hemisphere expansion inputs must be finite and nonnegative"))
        end
    end
    isempty(up) && return up, down
    backend = KernelAbstractions.get_backend(up)
    if backend isa KernelAbstractions.CPU
        @inbounds for d in axes(plus, 1)
            h = hemispheric_bin(plus, minus, front, back, d)
            up[d] = convert(FT, h.up)
            down[d] = convert(FT, h.down)
        end
    else
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
    end
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
    p_acc = zero(FT)
    m_acc = zero(FT)
    @inbounds for o_step in 0:(no - 1)
        o = o0 + o_step
        e = leaf_hemispheric_emission(law, mu[o], weighted[o, d, c, b], zero(FT))
        p_acc += convert(FT, e.upward)
        m_acc += convert(FT, e.downward)
    end
    @inbounds begin
        plus[d, c, b] = p_acc
        minus[d, c, b] = m_acc
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
    require_continuous_scalar(FT, batch_hemispheric_moments!, (plus, minus, law, mu, weighted))
    FT_req = promote_compatible_scalar(
        batch_hemispheric_moments!,
        zero(FT),
        _law_scalar_probe(law, zero(FT)),
        zero(eltype(mu)),
        zero(eltype(weighted)),
    )
    FT_req === FT ||
        throw(MethodError(batch_hemispheric_moments!, (plus, minus, law, mu, weighted)))
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
    u_acc = zero(FT)
    v_acc = zero(FT)
    @inbounds for c_step in 0:(nc - 1)
        c = c0 + c_step
        fc = convert(FT, front[m, c])
        bc = convert(FT, back[m, c])
        p_dc = convert(FT, plus[d, c])
        m_dc = convert(FT, minus[d, c])
        u_acc += fc * p_dc + bc * m_dc
        v_acc += fc * m_dc + bc * p_dc
    end
    @inbounds begin
        up[d, m] = u_acc
        down[d, m] = v_acc
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
    require_continuous_scalar(
        FT,
        batch_expand_hemispheric_sources!,
        (up, down, plus, minus, front, back),
    )
    FT_req = promote_compatible_scalar(
        batch_expand_hemispheric_sources!,
        zero(FT),
        zero(eltype(plus)),
        zero(eltype(minus)),
        zero(eltype(front)),
        zero(eltype(back)),
    )
    FT_req === FT ||
        throw(MethodError(batch_expand_hemispheric_sources!, (up, down, plus, minus, front, back)))
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
