"""
    source_projection(leaf_cosine, leaf_azimuth, source_cosine, source_azimuth)

Compute the signed Cartesian inner product between the upward-referenced
anatomical front leaf normal `(leaf_cosine, leaf_azimuth)` and the unit vector
pointing toward a radiation source `(source_cosine, source_azimuth)`.

Evaluates polar sines via [`unit_sin_from_cos`](@ref) on each cosine prior to
scalar promotion so zenith illumination (`|source_cosine| == 1`) produces zero
`NaN` partials under automatic differentiation with respect to other inputs.

# Examples
```jldoctest
julia> source_projection(1.0, 0.0, 0.5, 1.2)
0.5
```
"""
@inline function source_projection(
    leaf_cosine::Real,
    leaf_azimuth::Real,
    source_cosine::Real,
    source_azimuth::Real,
)
    FT = promote_compatible_scalar(
        source_projection,
        leaf_cosine,
        leaf_azimuth,
        source_cosine,
        source_azimuth,
    )
    sin_l = convert(FT, unit_sin_from_cos(leaf_cosine))
    sin_s = convert(FT, unit_sin_from_cos(source_cosine))
    mu_l = convert(FT, leaf_cosine)
    mu_s = convert(FT, source_cosine)
    dphi = convert(FT, leaf_azimuth) - convert(FT, source_azimuth)
    return mu_l * mu_s + sin_l * sin_s * cos(dphi)
end

"""
    beam_face_absorption(projection, beam_normal, absorptance)

Partition collimated beam absorption `(; front, back)` per unit one-sided leaf
area at the illuminated anatomical face from signed [`source_projection`](@ref),
beam-normal photon flux `beam_normal`, and leaf `absorptance`.

# Examples
```jldoctest
julia> beam_face_absorption(0.5, 1000.0, 0.85)
(front = 425.0, back = 0.0)
```
"""
@inline function beam_face_absorption(
    projection::Real,
    beam_normal::Real,
    absorptance::Real,
)
    FT = promote_compatible_scalar(
        beam_face_absorption,
        projection,
        beam_normal,
        absorptance,
    )
    proj = convert(FT, projection)
    bn = convert(FT, beam_normal)
    abs_f = convert(FT, absorptance)
    return (;
        front = abs_f * bn * max(proj, zero(FT)),
        back = abs_f * bn * max(-proj, zero(FT)),
    )
end

"""
    isotropic_face_absorption(leaf_cosine, down, up, absorptance)

Partition isotropic downward (`down`) and upward (`up`) horizontal fluxes into
anatomical `(; front, back)` absorbed fluxes per unit one-sided leaf area,
satisfying `front + back == absorptance * (down + up)`.

# Examples
```jldoctest
julia> f = isotropic_face_absorption(0.4, 200.0, 50.0, 0.8);

julia> isapprox(f.front + f.back, 200.0; atol = 1e-13)
true
```
"""
@inline function isotropic_face_absorption(
    leaf_cosine::Real,
    down::Real,
    up::Real,
    absorptance::Real,
)
    FT = promote_compatible_scalar(
        isotropic_face_absorption,
        leaf_cosine,
        down,
        up,
        absorptance,
    )
    mu = convert(FT, leaf_cosine)
    dn = convert(FT, down)
    up_f = convert(FT, up)
    abs_f = convert(FT, absorptance)
    front_fraction = (one(FT) + mu) / FT(2)
    back_fraction = (one(FT) - mu) / FT(2)
    return (;
        front = abs_f * (front_fraction * dn + back_fraction * up_f),
        back = abs_f * (back_fraction * dn + front_fraction * up_f),
    )
end

"""
    regular_plate_passive(plate)

Evaluate passive hemispherical `(; reflectance, transmittance, absorptance, rt_absorptance)`
for a dielectric plate prepared by `CanopyOptics.diffuse_plate_optics` across
primal floats and automatic-differentiation scalars.

Retains direct traversal `absorptance` without subtraction cancellation near
transparent limits alongside `rt_absorptance = 1 - (reflectance + transmittance)`.

# Examples
```jldoctest
julia> using CanopyOptics

julia> plate = diffuse_plate_optics(0.15, 1.4);

julia> p = regular_plate_passive(plate);

julia> isapprox(p.reflectance + p.transmittance + p.absorptance, 1.0; atol = 1e-14)
true
```
"""
@inline function regular_plate_passive(plate)
    FT = promote_compatible_scalar(
        regular_plate_passive,
        plate.traversal_transmission,
        plate.traversal_absorptance,
        plate.interface_entry_transmission,
        plate.interface_exit_transmission,
    )
    tau = convert(FT, plate.traversal_transmission)
    loss = convert(FT, plate.traversal_absorptance)
    entry = convert(FT, plate.interface_entry_transmission)
    exit = convert(FT, plate.interface_exit_transmission)
    reflection = one(FT) - exit
    one_minus_roundtrip = exit + reflection * loss
    denominator = one_minus_roundtrip * (one(FT) + reflection * tau)
    transmittance = entry * exit * tau / denominator
    reflectance = (one(FT) - entry) + entry * exit * reflection * tau * tau / denominator
    absorptance = entry * loss / one_minus_roundtrip
    return (;
        reflectance,
        transmittance,
        absorptance,
        rt_absorptance = one(FT) - (reflectance + transmittance),
    )
end

"""
    AbstractPopulationLayout

Supertype for leaf-population ordering conventions within each quadrature depth
([`PairedIllumination`](@ref) and [`BlockedIllumination`](@ref)).
"""
abstract type AbstractPopulationLayout end

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

broadcastable(l::PairedIllumination) = Ref(l)

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

broadcastable(l::BlockedIllumination) = Ref(l)

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
}
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
        require_continuous_scalar(
            FT,
            PopulationQuadrature,
            (layer, lai, lai_above, depths, depth_weights, cos_inclinations, azimuths, layout),
        )
        P = primal_float_type(FT)
        axes(cos_inclinations) == axes(azimuths) ||
            throw(ArgumentError("Leaf inclination and azimuth vectors must share axes"))
        axes(depths) == axes(depth_weights) ||
            throw(ArgumentError("Depth coordinates and weights must share axes"))
        layer > 0 &&
            finite_scalar(lai) &&
            primal_scalar(lai) >= zero(P) &&
            finite_scalar(lai_above) &&
            primal_scalar(lai_above) >= zero(P) ||
            throw(ArgumentError("Invalid canopy layer index or leaf area"))
        isempty(depths) == iszero(primal_scalar(lai)) ||
            throw(ArgumentError("Only zero-LAI layers have empty depth quadrature"))
        isempty(azimuths) &&
            !iszero(primal_scalar(lai)) &&
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
Base.eltype(g::PopulationQuadrature) = typeof(g.lai)
Base.length(g::PopulationQuadrature) = 2 * length(g.depths) * length(g.azimuths)

Base.:(==)(a::PopulationQuadrature, b::PopulationQuadrature) =
    a.layer == b.layer &&
    a.lai == b.lai &&
    a.lai_above == b.lai_above &&
    a.layout == b.layout &&
    axes(a.depths) == axes(b.depths) &&
    axes(a.cos_inclinations) == axes(b.cos_inclinations) &&
    a.depths == b.depths &&
    a.depth_weights == b.depth_weights &&
    a.cos_inclinations == b.cos_inclinations &&
    a.azimuths == b.azimuths

Base.isequal(a::PopulationQuadrature, b::PopulationQuadrature) =
    a.layer == b.layer &&
    isequal(a.lai, b.lai) &&
    isequal(a.lai_above, b.lai_above) &&
    a.layout == b.layout &&
    _isequal_array(a.depths, b.depths) &&
    _isequal_array(a.depth_weights, b.depth_weights) &&
    _isequal_array(a.cos_inclinations, b.cos_inclinations) &&
    _isequal_array(a.azimuths, b.azimuths)

function Base.hash(g::PopulationQuadrature, h::UInt)
    h = hash(:PopulationQuadrature, h)
    h = hash(g.layer, h)
    h = _hash_scalar(g.lai, h)
    h = _hash_scalar(g.lai_above, h)
    h = _hash_array(g.depths, h)
    h = _hash_array(g.depth_weights, h)
    h = _hash_array(g.cos_inclinations, h)
    h = _hash_array(g.azimuths, h)
    return hash(g.layout, h)
end

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
    require_continuous_scalar(FT, leaf_inclination_quadrature, (lad, n_leaf, FT))
    n_leaf > 0 || throw(ArgumentError("n_leaf must be positive"))
    P = primal_float_type(FT)
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
    finite_scalar(mass) && primal_scalar(mass) > zero(P) ||
        throw(ArgumentError("Leaf-inclination quadrature must have finite positive mass"))
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
    mu_s::FT;
    n_leaf::Integer = 16,
    n_azimuth::Integer = 16,
) where {FT<:Real}
    require_continuous_scalar(FT, orientation_hemisphere_samples, (lad, mu_s))
    n_leaf > 0 && n_azimuth > 0 ||
        throw(ArgumentError("n_leaf and n_azimuth must be positive"))
    P = primal_float_type(FT)
    finite_scalar(mu_s) && -one(P) <= primal_scalar(mu_s) <= one(P) ||
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
            primal_scalar(b) > primal_scalar(abs(a)) ?
            acos(clamp(-a / b, -one(FT), one(FT))) :
            (primal_scalar(a) >= zero(P) ? FT(pi) : zero(FT))
        for (left, right) in ((zero(FT), split), (split, FT(pi)))
            primal_scalar(right) <= primal_scalar(left) && continue
            half = (right - left) / FT(2)
            mid = (right + left) / FT(2)
            for j in eachindex(xi, w_ref)
                phi = mid + half * xi[j]
                wp = half * w_ref[j]
                proj = a + b * cos(phi)
                push!(projections, primal_scalar(proj) >= zero(P) ? proj : -proj)
                push!(areas, w_th * wp / FT(pi))
                push!(cos_inclinations, cos_th)
                push!(azimuths, phi)
            end
        end
    end
    return (; projections, areas, cos_inclinations, azimuths)
end

"""
    AbstractEmissionAngularLaw

Supertype for leaf-face angular emission distributions
([`LambertianFaces`](@ref) and [`MixedLambertianFaces`](@ref)).
"""
abstract type AbstractEmissionAngularLaw end

"""
    LambertianFaces()

Pure cosine-projected two-face Lambertian leaf emission law.

# Examples
```jldoctest
julia> LambertianFaces()
LambertianFaces()
```
"""
struct LambertianFaces <: AbstractEmissionAngularLaw end

broadcastable(l::LambertianFaces) = Ref(l)

"""
    MixedLambertianFaces(lambertian_fraction)

Convex mixture `λ ∈ [0, 1]` between two-face Lambertian emission (`λ = 1`) and
face-symmetric isotropic emission (`λ = 0`):

```math
J(\\mathbf{\\Omega}_r) = \\lambda J_{\\text{Lambertian}}(\\mathbf{\\Omega}_r) + (1 - \\lambda)\\frac{\\text{front} + \\text{back}}{4\\pi} \\ge 0.
```

Preserves nonnegativity on the entire sphere, exact total escape `upward + downward == front + back`,
and exact consistency between directional solid-angle integration and
[`leaf_hemispheric_emission`](@ref).

# Examples
```jldoctest
julia> law = MixedLambertianFaces(0.7);

julia> h = leaf_hemispheric_emission(law, 0.5, 0.6, 0.2);

julia> isapprox(h.upward + h.downward, 0.8; atol = 1e-15)
true
```
"""
struct MixedLambertianFaces{FT<:Real} <: AbstractEmissionAngularLaw
    lambertian_fraction::FT

    function MixedLambertianFaces(lambertian_fraction::FT) where {FT<:Real}
        require_continuous_scalar(FT, MixedLambertianFaces, (lambertian_fraction,))
        P = primal_float_type(FT)
        finite_scalar(lambertian_fraction) &&
            zero(P) <= primal_scalar(lambertian_fraction) <= one(P) ||
            throw(ArgumentError("lambertian_fraction must be finite and lie in [0, 1]"))
        return new{FT}(lambertian_fraction)
    end
end

Base.eltype(::Type{MixedLambertianFaces{FT}}) where {FT} = FT
Base.eltype(l::MixedLambertianFaces) = typeof(l.lambertian_fraction)
broadcastable(l::MixedLambertianFaces) = Ref(l)
Base.:(==)(a::MixedLambertianFaces, b::MixedLambertianFaces) =
    a.lambertian_fraction == b.lambertian_fraction
Base.isequal(a::MixedLambertianFaces, b::MixedLambertianFaces) =
    isequal(a.lambertian_fraction, b.lambertian_fraction)
Base.hash(l::MixedLambertianFaces, h::UInt) =
    _hash_scalar(l.lambertian_fraction, hash(:MixedLambertianFaces, h))

@inline _law_scalar_probe(::LambertianFaces, fallback) = fallback
@inline _law_scalar_probe(law::MixedLambertianFaces, _) = law.lambertian_fraction

"""
    leaf_hemispheric_emission(law::AbstractEmissionAngularLaw, leaf_cosine, front, back)

Integrate directional leaf emission analytically over global upward and downward
hemispheres, returning `(; upward, downward)` satisfying `upward + downward == front + back`.

# Examples
```jldoctest
julia> leaf_hemispheric_emission(LambertianFaces(), 0.5, 0.75, 0.25)
(upward = 0.625, downward = 0.375)
```
"""
@inline function leaf_hemispheric_emission(
    ::LambertianFaces,
    leaf_cosine::Real,
    front::Real,
    back::Real,
)
    FT = promote_compatible_scalar(leaf_hemispheric_emission, leaf_cosine, front, back)
    mu = convert(FT, leaf_cosine)
    fr = convert(FT, front)
    bk = convert(FT, back)
    f = (one(FT) + mu) / FT(2)
    g = (one(FT) - mu) / FT(2)
    return (; upward = f * fr + g * bk, downward = g * fr + f * bk)
end

@inline function leaf_hemispheric_emission(
    law::MixedLambertianFaces,
    leaf_cosine::Real,
    front::Real,
    back::Real,
)
    FT = promote_compatible_scalar(
        leaf_hemispheric_emission,
        law.lambertian_fraction,
        leaf_cosine,
        front,
        back,
    )
    lam = convert(FT, law.lambertian_fraction)
    mu = convert(FT, leaf_cosine)
    fr = convert(FT, front)
    bk = convert(FT, back)
    f = (one(FT) + lam * mu) / FT(2)
    g = (one(FT) - lam * mu) / FT(2)
    return (; upward = f * fr + g * bk, downward = g * fr + f * bk)
end

"""
    leaf_angular_emission(law::AbstractEmissionAngularLaw, leaf_cosine, leaf_azimuth, ray_cosine, ray_azimuth, front, back; mirrored_azimuth = false)

Evaluate photons (or energy) emitted per steradian and per unit one-sided leaf
area into outgoing ray `(ray_cosine, ray_azimuth)` from anatomical `front` and
`back` escapes.

Uses [`unit_sin_from_cos`](@ref) prior to cross-argument promotion so nadir or
zenith rays (`|ray_cosine| == 1`) remain free of `NaN` automatic-differentiation
tangents.

# Examples
```jldoctest
julia> isapprox(leaf_angular_emission(LambertianFaces(), 1.0, 0.0, 1.0, 0.0, Float64(pi), 0.0), 1.0; atol = 1e-15)
true
```
"""
@inline function leaf_angular_emission(
    ::LambertianFaces,
    leaf_cosine::Real,
    leaf_azimuth::Real,
    ray_cosine::Real,
    ray_azimuth::Real,
    front::Real,
    back::Real;
    mirrored_azimuth::Bool = false,
)
    FT = promote_compatible_scalar(
        leaf_angular_emission,
        leaf_cosine,
        leaf_azimuth,
        ray_cosine,
        ray_azimuth,
        front,
        back,
    )
    mu_l = convert(FT, leaf_cosine)
    mu_r = convert(FT, ray_cosine)
    sin_l = convert(FT, unit_sin_from_cos(leaf_cosine))
    sin_r = convert(FT, unit_sin_from_cos(ray_cosine))
    phi_l = convert(FT, leaf_azimuth)
    phi_r = convert(FT, ray_azimuth)
    fr = convert(FT, front)
    bk = convert(FT, back)
    a = mu_l * mu_r
    b = sin_l * sin_r
    proj = a + b * cos(phi_l - phi_r)
    val = fr * max(proj, zero(FT)) + bk * max(-proj, zero(FT))
    if mirrored_azimuth
        refl_proj = a + b * cos(-phi_l - phi_r)
        val += fr * max(refl_proj, zero(FT)) + bk * max(-refl_proj, zero(FT))
        val /= FT(2)
    end
    return val / FT(pi)
end

@inline function leaf_angular_emission(
    law::MixedLambertianFaces,
    leaf_cosine::Real,
    leaf_azimuth::Real,
    ray_cosine::Real,
    ray_azimuth::Real,
    front::Real,
    back::Real;
    mirrored_azimuth::Bool = false,
)
    FT = promote_compatible_scalar(
        leaf_angular_emission,
        law.lambertian_fraction,
        leaf_cosine,
        leaf_azimuth,
        ray_cosine,
        ray_azimuth,
        front,
        back,
    )
    lam = convert(FT, law.lambertian_fraction)
    fr = convert(FT, front)
    bk = convert(FT, back)
    lamb = leaf_angular_emission(
        LambertianFaces(),
        convert(FT, leaf_cosine),
        convert(FT, leaf_azimuth),
        convert(FT, ray_cosine),
        convert(FT, ray_azimuth),
        fr,
        bk;
        mirrored_azimuth,
    )
    iso = (fr + bk) / (FT(4) * FT(pi))
    return lam * lamb + (one(FT) - lam) * iso
end

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
    FT = promote_compatible_scalar(
        depth_hemispheric_source,
        _law_scalar_probe(law, zero(eltype(leaf_mu))),
        zero(eltype(leaf_mu)),
        zero(eltype(weights)),
        zero(eltype(front)),
        zero(eltype(back)),
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
    require_continuous_scalar(FT, assemble_hemispheric_emission!, (up, down))
    axes(up) == axes(down) == axes(g.depths) ||
        throw(DimensionMismatch("Hemispheric source vectors must share axes with g.depths"))
    axes(weights) == axes(front) == axes(back) && length(weights) == length(g) ||
        throw(DimensionMismatch("Face emission and weights must share axes and match length(g)"))
    P = primal_float_type(FT)
    for w in _host_linear(g.depth_weights)
        finite_scalar(w) && primal_scalar(w) > zero(P) ||
            throw(ArgumentError("Depth quadrature weights must be finite and positive"))
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
    FT = promote_compatible_scalar(
        depth_angular_source,
        _law_scalar_probe(law, zero(eltype(leaf_mu))),
        zero(eltype(leaf_mu)),
        zero(eltype(leaf_phi)),
        zero(eltype(weights)),
        zero(eltype(front)),
        zero(eltype(back)),
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
    require_continuous_scalar(FT, assemble_emission!, (source,))
    axes(weights) == axes(front) == axes(back) && length(weights) == length(g) ||
        throw(DimensionMismatch("Face emission and population weights must share axes and match length(g)"))
    axes(ray_mu) == axes(ray_phi) &&
        axes(source) == (axes(ray_mu, 1), axes(g.depths, 1)) ||
        throw(DimensionMismatch("Source storage must have (axes(ray_mu, 1), axes(g.depths, 1)) axes"))
    P = primal_float_type(FT)
    for w in _host_linear(g.depth_weights)
        finite_scalar(w) && primal_scalar(w) > zero(P) ||
            throw(ArgumentError("Depth quadrature weights must be finite and positive"))
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

"""
    AngularEmissionBoundary(reflection, emission, mu, weights, fourier_order)

Represent top-of-canopy boundary response `U = emission + reflection * D` for
one Fourier azimuth order `fourier_order >= 0` on discrete-ordinate polar
cosines `mu` and quadrature `weights`.

# Examples
```jldoctest
julia> b = AngularEmissionBoundary([0.1;;], [0.5;;], [0.7], [1.0], 0);

julia> b.fourier_order
0
```
"""
struct AngularEmissionBoundary{
    FT<:Real,
    R<:AbstractMatrix{FT},
    S<:AbstractMatrix{FT},
    V<:AbstractVector{FT},
}
    reflection::R
    emission::S
    mu::V
    weights::V
    fourier_order::Int

    function AngularEmissionBoundary(
        reflection::R,
        emission::S,
        mu::V,
        weights::V,
        fourier_order::Integer,
    ) where {FT<:Real,R<:AbstractMatrix{FT},S<:AbstractMatrix{FT},V<:AbstractVector{FT}}
        require_continuous_scalar(
            FT,
            AngularEmissionBoundary,
            (reflection, emission, mu, weights, fourier_order),
        )
        n = length(mu)
        n > 0 &&
            axes(mu) == axes(weights) &&
            axes(reflection) == (axes(mu, 1), axes(mu, 1)) &&
            axes(emission, 1) == axes(mu, 1) ||
            throw(DimensionMismatch("Boundary operators, sources, and ordinates must align"))
        fourier_order >= 0 || throw(ArgumentError("Fourier order must be nonnegative"))
        return new{FT,R,S,V}(reflection, emission, mu, weights, Int(fourier_order))
    end
end

Base.eltype(::Type{<:AngularEmissionBoundary{FT}}) where {FT} = FT
Base.eltype(b::AngularEmissionBoundary) = eltype(b.mu)

Base.:(==)(a::AngularEmissionBoundary, b::AngularEmissionBoundary) =
    a.fourier_order == b.fourier_order &&
    axes(a.mu) == axes(b.mu) &&
    axes(a.emission) == axes(b.emission) &&
    a.reflection == b.reflection &&
    a.emission == b.emission &&
    a.mu == b.mu &&
    a.weights == b.weights

Base.isequal(a::AngularEmissionBoundary, b::AngularEmissionBoundary) =
    a.fourier_order == b.fourier_order &&
    _isequal_array(a.reflection, b.reflection) &&
    _isequal_array(a.emission, b.emission) &&
    _isequal_array(a.mu, b.mu) &&
    _isequal_array(a.weights, b.weights)

function Base.hash(b::AngularEmissionBoundary, h::UInt)
    h = hash(:AngularEmissionBoundary, h)
    h = hash(b.fourier_order, h)
    h = _hash_array(b.reflection, h)
    h = _hash_array(b.emission, h)
    h = _hash_array(b.mu, h)
    return _hash_array(b.weights, h)
end

"""
    AbstractAtmosphericTransfer

Supertype for source-free atmospheric operators above an
[`AngularEmissionBoundary`](@ref) ([`VacuumAtmosphere`](@ref) and
[`AngularAtmosphericTransfer`](@ref)).
"""
abstract type AbstractAtmosphericTransfer end

"""
    VacuumAtmosphere()

Transparent atmosphere with zero downward backscatter and unit upward
transmission.

# Examples
```jldoctest
julia> VacuumAtmosphere()
VacuumAtmosphere()
```
"""
struct VacuumAtmosphere <: AbstractAtmosphericTransfer end

broadcastable(a::VacuumAtmosphere) = Ref(a)

"""
    AngularAtmosphericTransfer(return_down, transmit_up, mu, weights, fourier_order)

Source-free atmosphere seen from the top of the canopy (`D = return_down * U`,
`TOA = transmit_up * U`) on discrete-ordinate polar cosines `mu` and `weights`.

# Examples
```jldoctest
julia> atm = AngularAtmosphericTransfer([0.05;;], [0.9;;], [0.7], [1.0], 0);

julia> atm.fourier_order
0
```
"""
struct AngularAtmosphericTransfer{
    FT<:Real,
    R<:AbstractMatrix{FT},
    T<:AbstractMatrix{FT},
    V<:AbstractVector{FT},
} <: AbstractAtmosphericTransfer
    return_down::R
    transmit_up::T
    mu::V
    weights::V
    fourier_order::Int

    function AngularAtmosphericTransfer(
        return_down::R,
        transmit_up::T,
        mu::V,
        weights::V,
        fourier_order::Integer,
    ) where {FT<:Real,R<:AbstractMatrix{FT},T<:AbstractMatrix{FT},V<:AbstractVector{FT}}
        require_continuous_scalar(
            FT,
            AngularAtmosphericTransfer,
            (return_down, transmit_up, mu, weights, fourier_order),
        )
        n = length(mu)
        n > 0 &&
            axes(mu) == axes(weights) &&
            axes(return_down) == axes(transmit_up) == (axes(mu, 1), axes(mu, 1)) ||
            throw(DimensionMismatch("Atmospheric operators and ordinates must align"))
        fourier_order >= 0 || throw(ArgumentError("Fourier order must be nonnegative"))
        return new{FT,R,T,V}(return_down, transmit_up, mu, weights, Int(fourier_order))
    end
end

Base.eltype(::Type{<:AngularAtmosphericTransfer{FT}}) where {FT} = FT
Base.eltype(a::AngularAtmosphericTransfer) = eltype(a.mu)

Base.:(==)(a::AngularAtmosphericTransfer, b::AngularAtmosphericTransfer) =
    a.fourier_order == b.fourier_order &&
    axes(a.mu) == axes(b.mu) &&
    a.return_down == b.return_down &&
    a.transmit_up == b.transmit_up &&
    a.mu == b.mu &&
    a.weights == b.weights

Base.isequal(a::AngularAtmosphericTransfer, b::AngularAtmosphericTransfer) =
    a.fourier_order == b.fourier_order &&
    _isequal_array(a.return_down, b.return_down) &&
    _isequal_array(a.transmit_up, b.transmit_up) &&
    _isequal_array(a.mu, b.mu) &&
    _isequal_array(a.weights, b.weights)

function Base.hash(a::AngularAtmosphericTransfer, h::UInt)
    h = hash(:AngularAtmosphericTransfer, h)
    h = hash(a.fourier_order, h)
    h = _hash_array(a.return_down, h)
    h = _hash_array(a.transmit_up, h)
    h = _hash_array(a.mu, h)
    return _hash_array(a.weights, h)
end

function _check_shared_ordinate_grid(
    canopy::AngularEmissionBoundary,
    atmosphere::AngularAtmosphericTransfer,
)
    canopy.fourier_order == atmosphere.fourier_order &&
        axes(canopy.mu) == axes(atmosphere.mu) &&
        primal_scalar.(canopy.mu) == primal_scalar.(atmosphere.mu) &&
        primal_scalar.(canopy.weights) == primal_scalar.(atmosphere.weights) ||
        throw(ArgumentError("Coupling requires identical ordinate grids, weights, and Fourier order"))
    return nothing
end

"""
    couple_emission(canopy::AngularEmissionBoundary, atmosphere::AbstractAtmosphericTransfer)

Solve coupled canopy-atmosphere multiple scattering (`U = (I - R_c R_a) \\ e`,
`D = R_a U`, `TOA = T_a U`), returning `(; canopy_up, canopy_down, top_up)`.

Supports compatible mixed primal and automatic-differentiation scalar types.

# Examples
```jldoctest
julia> b = AngularEmissionBoundary([0.2;;], [1.0;;], [0.7], [1.0], 0);

julia> atm = AngularAtmosphericTransfer([0.1;;], [0.8;;], [0.7], [1.0], 0);

julia> c = couple_emission(b, atm);

julia> isapprox(c.canopy_up[1, 1], 1.0 / (1.0 - 0.02); atol = 1e-15)
true
```
"""
function couple_emission(
    canopy::AngularEmissionBoundary{FT1},
    atmosphere::AngularAtmosphericTransfer{FT2},
) where {FT1<:Real,FT2<:Real}
    FT = compatible_scalar(FT1, FT2)
    FT === Union{} && throw(MethodError(couple_emission, (canopy, atmosphere)))
    _check_shared_ordinate_grid(canopy, atmosphere)
    rc = FT === FT1 ? canopy.reflection : convert.(FT, canopy.reflection)
    em = FT === FT1 ? canopy.emission : convert.(FT, canopy.emission)
    ra = FT === FT2 ? atmosphere.return_down : convert.(FT, atmosphere.return_down)
    ta = FT === FT2 ? atmosphere.transmit_up : convert.(FT, atmosphere.transmit_up)
    factor = lu(I - rc * ra)
    canopy_up = factor \ em
    canopy_down = ra * canopy_up
    top_up = ta * canopy_up
    return (; canopy_up, canopy_down, top_up)
end

function couple_emission(canopy::AngularEmissionBoundary, ::VacuumAtmosphere)
    canopy_up = copy(canopy.emission)
    canopy_down = zero(canopy.emission)
    top_up = copy(canopy.emission)
    return (; canopy_up, canopy_down, top_up)
end

"""
    one_way_emission(canopy::AngularEmissionBoundary, atmosphere::AbstractAtmosphericTransfer)

Evaluate single-pass upward atmospheric transmission `T_a * emission` ignoring
downward atmospheric return.

# Examples
```jldoctest
julia> b = AngularEmissionBoundary([0.2;;], [1.0;;], [0.7], [1.0], 0);

julia> one_way_emission(b, VacuumAtmosphere())
1×1 Matrix{Float64}:
 1.0
```
"""
function one_way_emission(
    canopy::AngularEmissionBoundary{FT1},
    atmosphere::AngularAtmosphericTransfer{FT2},
) where {FT1<:Real,FT2<:Real}
    FT = compatible_scalar(FT1, FT2)
    FT === Union{} && throw(MethodError(one_way_emission, (canopy, atmosphere)))
    _check_shared_ordinate_grid(canopy, atmosphere)
    em = FT === FT1 ? canopy.emission : convert.(FT, canopy.emission)
    ta = FT === FT2 ? atmosphere.transmit_up : convert.(FT, atmosphere.transmit_up)
    return ta * em
end

one_way_emission(canopy::AngularEmissionBoundary, ::VacuumAtmosphere) = copy(canopy.emission)
