"""
    unit_sin_from_cos(mu)

Evaluate `sqrt(max(0, (1 - mu) * (1 + mu)))` cleanly across primal floats and
automatic-differentiation dual numbers.

When `(1 - mu) * (1 + mu)` is identically zero ([`CanopyRadiation.exact_zero`](@ref)) or has a
nonpositive primal value (such as `mu = ±1` at zenith illumination or nadir
viewing), return `zero(typeof(mu))` directly without invoking `sqrt` at zero,
avoiding singular `NaN` tangents when differentiating with respect to other
parameters.

# Examples
```jldoctest
julia> CanopyRadiation.unit_sin_from_cos(0.6)
0.8

julia> CanopyRadiation.unit_sin_from_cos(1.0)
0.0
```
"""
@inline function unit_sin_from_cos(mu::Real)
    same_precision(mu)
    FT = typeof(mu)
    P = primal_type(FT)
    s2 = (one(FT) - mu) * (one(FT) + mu)
    if exact_zero(s2) || primal(s2) <= zero(P)
        return zero(FT)
    end
    return sqrt(s2)
end

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
    FT = common_scalar_type(leaf_cosine, leaf_azimuth, source_cosine, source_azimuth)
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
    FT = common_scalar_type(projection, beam_normal, absorptance)
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
    FT = common_scalar_type(leaf_cosine, down, up, absorptance)
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
    FT = common_scalar_type(
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
