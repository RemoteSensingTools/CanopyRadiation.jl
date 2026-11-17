"""
    AbstractEmissionAngularLaw

Supertype for leaf-face angular emission distributions
([`LambertianFaces`](@ref) and [`MixedLambertianFaces`](@ref)).
"""
abstract type AbstractEmissionAngularLaw <: RadiationRecord end

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

    function MixedLambertianFaces(lambertian_fraction::Real)
        same_precision(lambertian_fraction)
        _checked(
            lambertian_fraction,
            _unit_interval,
            "lambertian_fraction must be finite and lie in [0, 1]",
        )
        return new{typeof(lambertian_fraction)}(lambertian_fraction)
    end
end

Base.eltype(::Type{<:MixedLambertianFaces{FT}}) where {FT} = FT

# Scalars of an emission law that take part in the precision of a kernel (none for a
# law without parameters).
@inline _precision_operands(::AbstractEmissionAngularLaw) = ()
@inline _precision_operands(law::MixedLambertianFaces) = (law,)

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
    FT = common_scalar_type(leaf_cosine, front, back)
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
    FT = common_scalar_type(law.lambertian_fraction, leaf_cosine, front, back)
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
    FT = common_scalar_type(
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
    FT = common_scalar_type(
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
