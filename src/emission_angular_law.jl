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
