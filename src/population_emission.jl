"""
    depth_hemispheric_source(law, layout, leaf_mu, weights, front, back, depth_step::Integer, depth_weight)

Reduce population-escaped `front` and `back` emissions at 1-based depth ordinal
`depth_step` into `(; upward, downward)` hemispheric sources per unit `LAI`,
respecting generic vector axes on `leaf_mu`, `weights`, `front`, and `back`.
"""
@inline function depth_hemispheric_source(
    law::AbstractEmissionAngularLaw,
    layout::AbstractPopulationLayout,
    leaf_mu::AbstractVector,
    weights::AbstractVector,
    front::AbstractVector,
    back::AbstractVector,
    depth_step::Integer,
    depth_weight::Real,
)
    FT = common_scalar_type(
        _precision_operands(law)...,
        leaf_mu,
        weights,
        front,
        back,
        depth_weight,
    )
    n = length(leaf_mu)
    base = (depth_step - 1) * 2 * n
    @boundscheck begin
        depth_step >= 1 &&
            base + 2 * n <= min(length(weights), length(front), length(back)) ||
            throw(BoundsError((weights, front, back), depth_step))
    end
    o0 = firstindex(leaf_mu)
    w0 = firstindex(weights)
    f0 = firstindex(front)
    b0 = firstindex(back)
    up, down = zero(FT), zero(FT)
    @inbounds for local_index in 0:(2 * n - 1)
        ori_step, _ = _orientation_step(layout, local_index, n)
        e = leaf_hemispheric_emission(
            law,
            leaf_mu[o0 + ori_step],
            front[f0 + base + local_index],
            back[b0 + base + local_index],
        )
        w_k = convert(FT, weights[w0 + base + local_index])
        up += w_k * convert(FT, e.upward)
        down += w_k * convert(FT, e.downward)
    end
    dw = convert(FT, depth_weight)
    return (; upward = up / dw, downward = down / dw)
end

"""
    assemble_hemispheric_emission!(up, down, law, g::PopulationQuadrature, weights, front, back)

Assemble depth-resolved hemispheric sources per unit `LAI` into caller-owned
vectors `up` and `down` aligned with `axes(g.depths)`.
"""
function assemble_hemispheric_emission!(
    up::AbstractVector{FT},
    down::AbstractVector{FT},
    law::AbstractEmissionAngularLaw,
    g::PopulationQuadrature,
    weights::AbstractVector,
    front::AbstractVector,
    back::AbstractVector,
) where {FT<:Real}
    same_precision(up)
    axes(up) == axes(down) == axes(g.depths) ||
        throw(DimensionMismatch("Hemispheric source vectors must share axes with g.depths"))
    axes(weights) == axes(front) == axes(back) && length(weights) == length(g) ||
        throw(DimensionMismatch("Face emission and weights must share axes and match length(g)"))
    for w in _host_linear(g.depth_weights)
        _checked(w, _positive, "Depth quadrature weights must be finite and positive")
    end
    @inbounds for (step, d_idx) in enumerate(eachindex(g.depths))
        e = depth_hemispheric_source(
            law,
            g.layout,
            g.cos_inclinations,
            weights,
            front,
            back,
            step,
            g.depth_weights[d_idx],
        )
        up[d_idx] = convert(FT, e.upward)
        down[d_idx] = convert(FT, e.downward)
    end
    return up, down
end

"""
    depth_angular_source(law, layout, leaf_mu, leaf_phi, weights, front, back, depth_step::Integer, depth_weight, ray_mu, ray_phi)

Evaluate directional volume emission per steradian and per unit `LAI` at 1-based
depth ordinal `depth_step` into outgoing direction `(ray_mu, ray_phi)`.
"""
@inline function depth_angular_source(
    law::AbstractEmissionAngularLaw,
    layout::AbstractPopulationLayout,
    leaf_mu::AbstractVector,
    leaf_phi::AbstractVector,
    weights::AbstractVector,
    front::AbstractVector,
    back::AbstractVector,
    depth_step::Integer,
    depth_weight::Real,
    ray_mu::Real,
    ray_phi::Real,
)
    FT = common_scalar_type(
        _precision_operands(law)...,
        leaf_mu,
        leaf_phi,
        weights,
        front,
        back,
        depth_weight,
        ray_mu,
        ray_phi,
    )
    n = length(leaf_mu)
    base = (depth_step - 1) * 2 * n
    @boundscheck begin
        length(leaf_phi) == n ||
            throw(DimensionMismatch("leaf_mu and leaf_phi must have equal length"))
        depth_step >= 1 &&
            base + 2 * n <= min(length(weights), length(front), length(back)) ||
            throw(BoundsError((weights, front, back), depth_step))
    end
    o0 = firstindex(leaf_mu)
    p0 = firstindex(leaf_phi)
    w0 = firstindex(weights)
    f0 = firstindex(front)
    b0 = firstindex(back)
    val = zero(FT)
    @inbounds for local_index in 0:(2 * n - 1)
        ori_step, _ = _orientation_step(layout, local_index, n)
        e = leaf_angular_emission(
            law,
            leaf_mu[o0 + ori_step],
            leaf_phi[p0 + ori_step],
            ray_mu,
            ray_phi,
            front[f0 + base + local_index],
            back[b0 + base + local_index];
            mirrored_azimuth = true,
        )
        val += convert(FT, weights[w0 + base + local_index]) * convert(FT, e)
    end
    return val / convert(FT, depth_weight)
end

"""
    assemble_emission!(source, law, g::PopulationQuadrature, weights, front, back, ray_mu, ray_phi)

Assemble `(direction, depth)` directional emission per steradian and per unit
`LAI` into `source`, preserving generic axes `(axes(ray_mu, 1), axes(g.depths, 1))`.
"""
function assemble_emission!(
    source::AbstractMatrix{FT},
    law::AbstractEmissionAngularLaw,
    g::PopulationQuadrature,
    weights::AbstractVector,
    front::AbstractVector,
    back::AbstractVector,
    ray_mu::AbstractVector,
    ray_phi::AbstractVector,
) where {FT<:Real}
    same_precision(source)
    axes(weights) == axes(front) == axes(back) && length(weights) == length(g) ||
        throw(DimensionMismatch("Face emission and population weights must share axes and match length(g)"))
    axes(ray_mu) == axes(ray_phi) &&
        axes(source) == (axes(ray_mu, 1), axes(g.depths, 1)) ||
        throw(DimensionMismatch("Source storage must have (axes(ray_mu, 1), axes(g.depths, 1)) axes"))
    for w in _host_linear(g.depth_weights)
        _checked(w, _positive, "Depth quadrature weights must be finite and positive")
    end
    @inbounds for (step, d_idx) in enumerate(eachindex(g.depths)), r_idx in eachindex(ray_mu)
        source[r_idx, d_idx] = convert(
            FT,
            depth_angular_source(
                law,
                g.layout,
                g.cos_inclinations,
                g.azimuths,
                weights,
                front,
                back,
                step,
                g.depth_weights[d_idx],
                ray_mu[r_idx],
                ray_phi[r_idx],
            ),
        )
    end
    return source
end
