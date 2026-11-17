"""
    AbstractPopulationLayout

Supertype for leaf-population ordering conventions within each quadrature depth
([`PairedIllumination`](@ref) and [`BlockedIllumination`](@ref)).
"""
abstract type AbstractPopulationLayout <: RadiationRecord end

"""
    PairedIllumination()

Order shaded and sunlit entries as consecutive `(shaded, sunlit)` pairs for
each orientation within each depth node.

# Examples
```jldoctest
julia> PairedIllumination()
PairedIllumination()
```
"""
struct PairedIllumination <: AbstractPopulationLayout end

"""
    BlockedIllumination()

Order all shaded orientations followed by all sunlit orientations within each
depth node.

# Examples
```jldoctest
julia> BlockedIllumination()
BlockedIllumination()
```
"""
struct BlockedIllumination <: AbstractPopulationLayout end

"""
    PopulationQuadrature(layer, lai, lai_above, depths, depth_weights, cos_inclinations, azimuths, layout)

Store factorized depth and leaf-orientation quadrature coordinates for canopy
`layer` (`layer > 0`) with local one-sided `lai` and cumulative `lai_above`.

Supports generic vector axes (`OffsetArray` and views) on aligned `(depths, depth_weights)`
and aligned `(cos_inclinations, azimuths)`.

# Examples
```jldoctest
julia> pq = PopulationQuadrature(1, 1.0, 0.0, [0.5], [1.0], [0.5], [0.8], PairedIllumination());

julia> length(pq)
2
```
"""
struct PopulationQuadrature{
    FT<:Real,
    VD<:AbstractVector{FT},
    VO<:AbstractVector{FT},
    L<:AbstractPopulationLayout,
} <: RadiationRecord
    layer::Int
    lai::FT
    lai_above::FT
    depths::VD
    depth_weights::VD
    cos_inclinations::VO
    azimuths::VO
    layout::L

    function PopulationQuadrature(
        layer::Integer,
        lai::FT,
        lai_above::FT,
        depths::VD,
        depth_weights::VD,
        cos_inclinations::VO,
        azimuths::VO,
        layout::L,
    ) where {
        FT<:Real,
        VD<:AbstractVector{FT},
        VO<:AbstractVector{FT},
        L<:AbstractPopulationLayout,
    }
        same_precision(lai)
        P = primal_type(FT)
        axes(cos_inclinations) == axes(azimuths) ||
            throw(ArgumentError("Leaf inclination and azimuth vectors must share axes"))
        axes(depths) == axes(depth_weights) ||
            throw(ArgumentError("Depth coordinates and weights must share axes"))
        layer > 0 &&
            all_finite(lai) &&
            primal(lai) >= zero(P) &&
            all_finite(lai_above) &&
            primal(lai_above) >= zero(P) ||
            throw(ArgumentError("Invalid canopy layer index or leaf area"))
        isempty(depths) == iszero(primal(lai)) ||
            throw(ArgumentError("Only zero-LAI layers have empty depth quadrature"))
        isempty(azimuths) &&
            !iszero(primal(lai)) &&
            throw(ArgumentError("Nonempty layers require orientation quadrature"))
        return new{FT,VD,VO,L}(
            Int(layer),
            lai,
            lai_above,
            depths,
            depth_weights,
            cos_inclinations,
            azimuths,
            layout,
        )
    end
end

Base.eltype(::Type{<:PopulationQuadrature{FT}}) where {FT} = FT
Base.length(g::PopulationQuadrature) = 2 * length(g.depths) * length(g.azimuths)

@inline _orientation_step(::PairedIllumination, local_index::Integer, ::Integer) =
    (local_index ÷ 2, isodd(local_index))
@inline _orientation_step(::BlockedIllumination, local_index::Integer, n::Integer) =
    (local_index % n, local_index >= n)

"""
    population_coordinates(g::PopulationQuadrature, k::Integer)

Return geometry metadata for the `k`-th population entry (`1 <= k <= length(g)`),
including `depth_index`, `orientation_index`, `depth_lai`, `column_depth_lai`,
`depth_weight`, `cos_inclination`, `azimuth`, `sunlit`, and `mirrored_azimuth`.

# Examples
```jldoctest
julia> pq = PopulationQuadrature(1, 1.0, 0.5, [0.5], [1.0], [0.6], [0.3], PairedIllumination());

julia> c = population_coordinates(pq, 2);

julia> (c.sunlit, c.column_depth_lai)
(true, 1.0)
```
"""
@inline function population_coordinates(g::PopulationQuadrature, k::Integer)
    @boundscheck (1 <= k <= length(g)) || throw(BoundsError(g, k))
    n = length(g.azimuths)
    depth_step, local_index = divrem(k - 1, 2 * n)
    ori_step, sunlit = _orientation_step(g.layout, local_index, n)
    d_idx = firstindex(g.depths) + depth_step
    o_idx = firstindex(g.cos_inclinations) + ori_step
    @inbounds begin
        local_depth = g.depths[d_idx]
        return (;
            layer = g.layer,
            depth_index = d_idx,
            orientation_index = o_idx,
            depth_lai = local_depth,
            column_depth_lai = g.lai_above + local_depth,
            depth_weight = g.depth_weights[d_idx],
            cos_inclination = g.cos_inclinations[o_idx],
            azimuth = g.azimuths[o_idx],
            sunlit,
            mirrored_azimuth = true,
        )
    end
end

"""
    leaf_inclination_quadrature(lad::AbstractLeafDistribution, n_leaf::Integer, ::Type{FT} = Float64)

Compute `n_leaf`-node Gauss-Legendre leaf inclination angles `θ ∈ (0, π/2)` and
normalized leaf-area quadrature weights `w_measure` (`sum(w_measure) ≈ 1`) from
public `CanopyOptics.gauleg` and `Distributions.pdf`.

# Examples
```jldoctest
julia> using CanopyOptics

julia> q = leaf_inclination_quadrature(spherical_leaves(), 8, Float64);

julia> isapprox(sum(q.w_measure), 1.0; atol = 1e-14)
true
```
"""
function leaf_inclination_quadrature(
    lad::AbstractLeafDistribution,
    n_leaf::Integer,
    ::Type{FT} = Float64,
) where {FT<:Real}
    same_precision(zero(FT))
    n_leaf > 0 || throw(ArgumentError("n_leaf must be positive"))
    P = primal_type(FT)
    theta_p, w_p = gauleg(Int(n_leaf), zero(P), P(pi) / P(2))
    theta = FT.(theta_p)
    w = FT.(w_p)
    w_measure = Vector{FT}(undef, length(theta))
    inv_half_pi = FT(2) / FT(pi)
    scale = convert(FT, lad.scaling)
    dist = lad.LD
    mass = zero(FT)
    @inbounds for i in eachindex(theta, w, w_measure)
        wm = w[i] * convert(FT, pdf(dist, inv_half_pi * theta[i])) * scale
        w_measure[i] = wm
        mass += wm
    end
    _checked(mass, _positive, "Leaf-inclination quadrature must have finite positive mass")
    inv_mass = inv(mass)
    @inbounds for i in eachindex(w_measure)
        w_measure[i] *= inv_mass
    end
    return (; θ = theta, w_measure)
end

"""
    orientation_hemisphere_samples(lad::AbstractLeafDistribution, mu_s; n_leaf = 16, n_azimuth = 16)

Construct folded `[0, π]` solar-azimuth and leaf-inclination quadrature samples
`(; projections, areas, cos_inclinations, azimuths)` split cleanly at the leaf
horizon `a + b cos(ϕ) = 0` so `|a + b cos(ϕ)|` is integrated smoothly on each
subinterval.

# Examples
```jldoctest
julia> using CanopyOptics

julia> s = orientation_hemisphere_samples(spherical_leaves(), 0.5; n_leaf = 8, n_azimuth = 8);

julia> isapprox(sum(s.areas), 1.0; atol = 1e-13) && isapprox(sum(s.areas .* s.projections), 0.5; atol = 1e-3)
true
```
"""
function orientation_hemisphere_samples(
    lad::AbstractLeafDistribution,
    mu_s::Real;
    n_leaf::Integer = 16,
    n_azimuth::Integer = 16,
)
    same_precision(mu_s)
    FT = typeof(mu_s)
    n_leaf > 0 && n_azimuth > 0 ||
        throw(ArgumentError("n_leaf and n_azimuth must be positive"))
    P = primal_type(FT)
    all_finite(mu_s) && -one(P) <= primal(mu_s) <= one(P) ||
        throw(ArgumentError("Solar cosine mu_s must be finite and lie in [-1, 1]"))
    leaf = leaf_inclination_quadrature(lad, n_leaf, FT)
    sin_s = unit_sin_from_cos(mu_s)
    xi_p, w_p = gauleg(Int(n_azimuth), -one(P), one(P))
    xi = FT.(xi_p)
    w_ref = FT.(w_p)
    projections = FT[]
    areas = FT[]
    cos_inclinations = FT[]
    azimuths = FT[]
    for (th, w_th) in zip(leaf.θ, leaf.w_measure)
        cos_th = cos(th)
        sin_th = sin(th)
        a = mu_s * cos_th
        b = sin_s * sin_th
        split =
            primal(b) > primal(abs(a)) ?
            acos(clamp(-a / b, -one(FT), one(FT))) :
            (primal(a) >= zero(P) ? FT(pi) : zero(FT))
        for (left, right) in ((zero(FT), split), (split, FT(pi)))
            primal(right) <= primal(left) && continue
            half = (right - left) / FT(2)
            mid = (right + left) / FT(2)
            for j in eachindex(xi, w_ref)
                phi = mid + half * xi[j]
                wp = half * w_ref[j]
                proj = a + b * cos(phi)
                push!(projections, primal(proj) >= zero(P) ? proj : -proj)
                push!(areas, w_th * wp / FT(pi))
                push!(cos_inclinations, cos_th)
                push!(azimuths, phi)
            end
        end
    end
    return (; projections, areas, cos_inclinations, azimuths)
end
