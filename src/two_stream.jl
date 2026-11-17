"""
    AbstractTwoStreamMethod

Supertype for homogeneous-slab two-stream operator closures
([`NativeTwoStream`](@ref) and [`AnalyticTwoStream`](@ref)).
"""
abstract type AbstractTwoStreamMethod <: RadiationRecord end

"""
    NativeTwoStream(thin_norm, source_terms)
    NativeTwoStream(::Type{FT} = Float64; thin_norm = FT(0.05), source_terms = 12)

Specify thin-layer Taylor integration plus adding-doubling controls for
homogeneous two-stream canopy layers.

`thin_norm` bounds the dimensionless sub-slab norm `h * max(a + b, k)` prior to
repeated doubling, and `source_terms` sets the Taylor order for the direct-beam
source column on the thin sub-slab. `thin_norm` must be a finite positive
floating-point number (`ArgumentError` otherwise).

Accuracy: every doubling roughly doubles the rounding error of the diffuse operator.
For conservative scattering `|1 - R - T|` is about `1.2 × 2^d × eps(FT)` after `d`
doublings, and the number of doublings is set by `max(a + b, k)`, so a low sun
(large `k`) degrades the diffuse operator too: in `Float64` at most about `1.5e-11`
at `d = 16`; in `Float32` about `2e-5` at `d = 7` (LAI 1, solar zenith 85°) and
`1.3e-3` at `d = 13` (LAI 1, solar zenith 89.9°), with negative layer absorption of
the same size. Prefer [`AnalyticTwoStream`](@ref) for `Float32` at low sun. With
dual numbers the thin sub-slab uses a series where floats use the closed form; the
value of a dual result then differs from the plain-float result by about `2^d` times
a few eps (about 24 eps at `d = 4`, 450 at `d = 8`, `1e4` at `d = 16`). An infinite
`lai` does not terminate.

# Examples
```jldoctest
julia> m = NativeTwoStream(0.05, 12)
NativeTwoStream{Float64}(0.05, 12)
```
"""
struct NativeTwoStream{FT<:Real} <: AbstractTwoStreamMethod
    thin_norm::FT
    source_terms::Int

    function NativeTwoStream(thin_norm::Real, source_terms::Integer)
        same_precision(thin_norm)
        _checked(thin_norm, _positive, "NativeTwoStream thin_norm must be finite and positive")
        source_terms > 0 ||
            throw(ArgumentError("NativeTwoStream source_terms must be positive"))
        return new{typeof(thin_norm)}(thin_norm, Int(source_terms))
    end
end

function NativeTwoStream(
    ::Type{FT} = Float64;
    thin_norm::Real = primal_type(FT)(0.05),
    source_terms::Integer = 12,
) where {FT<:Real}
    S = common_scalar_type(FT, thin_norm)
    return NativeTwoStream(convert(S, thin_norm), source_terms)
end

Base.eltype(::Type{<:NativeTwoStream{FT}}) where {FT} = FT

"""
    AnalyticTwoStream()

Closed-form homogeneous two-stream layer closure without thin-slab halving or
doubling loops.

Evaluates diffuse reflection/transmission and direct-beam coupling analytically
in an `r_∞`-free Green's-function form that remains nonsingular and smooth under
automatic differentiation across conservative scattering (`α → 0`), resonant
beam extinction (`k → m`), and optically thick slabs.

# Examples
```jldoctest
julia> AnalyticTwoStream()
AnalyticTwoStream()
```
"""
struct AnalyticTwoStream <: AbstractTwoStreamMethod end

# Scalars of a two-stream method that take part in the precision of a layer
# computation (none for a method without parameters).
@inline _precision_operands(::AbstractTwoStreamMethod) = ()
@inline _precision_operands(m::NativeTwoStream) = (m,)

"""
    CanopyRates(absorption, backscatter, beam_up, beam_down, extinction)

Store differential two-stream coefficients per unit one-sided leaf area index
(`LAI`), where diffuse attenuation is `a = absorption + backscatter`.

All five rates must be floating-point numbers of one precision: `Float32` with
`Float64`, integers, rationals and irrationals throw an `ArgumentError`; a
`Float64` mixed with a `Float64`-valued dual number promotes to the dual.

# Fields
- `absorption::FT`: diffuse absorption rate `α = a - b` per unit `LAI`.
- `backscatter::FT`: diffuse backward scattering rate `b` per unit `LAI`.
- `beam_up::FT`: upward diffuse scatter from direct beam `s_b` per unit `LAI`.
- `beam_down::FT`: downward diffuse scatter from direct beam `s_f` per unit `LAI`.
- `extinction::FT`: direct-beam extinction rate `k` per unit `LAI`.

# Examples
```jldoctest
julia> r = CanopyRates(0.12, 0.08, 0.05, 0.06, 0.7)
CanopyRates{Float64}(0.12, 0.08, 0.05, 0.06, 0.7)
```
"""
struct CanopyRates{FT<:Real} <: RadiationRecord
    absorption::FT
    backscatter::FT
    beam_up::FT
    beam_down::FT
    extinction::FT

    function CanopyRates{FT}(
        absorption::FT,
        backscatter::FT,
        beam_up::FT,
        beam_down::FT,
        extinction::FT,
    ) where {FT<:Real}
        same_precision(FT)
        return new{FT}(absorption, backscatter, beam_up, beam_down, extinction)
    end
end

@inline function CanopyRates(
    absorption::Real,
    backscatter::Real,
    beam_up::Real,
    beam_down::Real,
    extinction::Real,
)
    FT = common_scalar_type(absorption, backscatter, beam_up, beam_down, extinction)
    return CanopyRates{FT}(
        convert(FT, absorption),
        convert(FT, backscatter),
        convert(FT, beam_up),
        convert(FT, beam_down),
        convert(FT, extinction),
    )
end

Base.eltype(::Type{<:CanopyRates{FT}}) where {FT} = FT
Base.convert(::Type{CanopyRates{FT}}, r::CanopyRates) where {FT} =
    _convert_record(CanopyRates{FT}, r)

"""
    foursail_canopy_rates(geom::FourSAILGeometry, reflectance, transmittance)
    foursail_canopy_rates(geom::FourSAILGeometry, reflectance, transmittance, clumping)

Map precomputed 4SAIL angular geometry `geom` and leaf hemispherical
`reflectance` and `transmittance` (plus optional `clumping` index `Ω ≥ 0`,
defaulting to `1`) into differential two-stream [`CanopyRates`](@ref).

Leaf absorptance is evaluated internally as `1 - (reflectance + transmittance)`
so the 4SAIL extinction identity `a = absorption + backscatter = 1 - sigf` holds
identically. Positional arguments allow broadcasting over CPU and GPU wavelength
vectors via `foursail_canopy_rates.((geom,), reflectance, transmittance)`.

# Examples
```jldoctest
julia> using CanopyOptics

julia> geom = FourSAILGeometry(spherical_leaves(); sza_deg = 30.0, vza_deg = 0.0, raa_deg = 0.0);

julia> rates = foursail_canopy_rates(geom, 0.08, 0.06);

julia> isapprox(rates.absorption, 0.86; atol = 1e-14)
true
```
"""
@inline function foursail_canopy_rates(
    geom::FourSAILGeometry,
    reflectance::Real,
    transmittance::Real,
)
    FT = common_scalar_type(geom.ks, reflectance, transmittance)
    P = primal_type(FT)
    r = convert(FT, reflectance)
    t = convert(FT, transmittance)
    all_finite(r) && all_finite(t) && all_finite(geom.ks) ||
        throw(ArgumentError("Leaf optical inputs and geometry must be finite"))
    primal(r) >= zero(P) &&
        primal(t) >= zero(P) &&
        primal(r + t) <= one(P) ||
        throw(ArgumentError("Leaf reflectance and transmittance must satisfy R >= 0, T >= 0, and R + T <= 1"))
    ddb = convert(FT, geom.ddb)
    ddf = convert(FT, geom.ddf)
    sdb = convert(FT, geom.sdb)
    sdf = convert(FT, geom.sdf)
    ks = convert(FT, geom.ks)
    return CanopyRates{FT}(
        one(FT) - (r + t),
        ddb * r + ddf * t,
        sdb * r + sdf * t,
        sdf * r + sdb * t,
        ks,
    )
end

@inline function foursail_canopy_rates(
    geom::FourSAILGeometry,
    reflectance::Real,
    transmittance::Real,
    clumping::Real,
)
    FT = common_scalar_type(geom.ks, reflectance, transmittance, clumping)
    om = _checked(
        convert(FT, clumping),
        _nonnegative,
        "Clumping index must be finite and nonnegative",
    )
    base = foursail_canopy_rates(geom, convert(FT, reflectance), convert(FT, transmittance))
    return CanopyRates{FT}(
        om * base.absorption,
        om * base.backscatter,
        om * base.beam_up,
        om * base.beam_down,
        om * base.extinction,
    )
end

"""
    LayerOperator(reflection, transmission, beam_up, beam_down, beam_transmission)

Store the dimensionless scattering and transmission response of a homogeneous
canopy layer to unit boundary diffuse and direct-beam fluxes.

# Fields
- `reflection::FT`: bi-hemispherical diffuse reflectance `R`.
- `transmission::FT`: bi-hemispherical diffuse transmittance `T`.
- `beam_up::FT`: upward diffuse escape at top from unit top horizontal direct beam.
- `beam_down::FT`: downward diffuse escape at bottom from unit top horizontal direct beam.
- `beam_transmission::FT`: unscattered direct-beam transmittance `exp(-k * LAI)`.

# Examples
```jldoctest
julia> op = LayerOperator(0.05, 0.72, 0.04, 0.06, 0.5)
LayerOperator{Float64}(0.05, 0.72, 0.04, 0.06, 0.5)
```
"""
struct LayerOperator{FT<:Real} <: RadiationRecord
    reflection::FT
    transmission::FT
    beam_up::FT
    beam_down::FT
    beam_transmission::FT

    function LayerOperator{FT}(
        reflection::FT,
        transmission::FT,
        beam_up::FT,
        beam_down::FT,
        beam_transmission::FT,
    ) where {FT<:Real}
        same_precision(FT)
        return new{FT}(reflection, transmission, beam_up, beam_down, beam_transmission)
    end
end

@inline function LayerOperator(
    reflection::Real,
    transmission::Real,
    beam_up::Real,
    beam_down::Real,
    beam_transmission::Real,
)
    FT = common_scalar_type(reflection, transmission, beam_up, beam_down, beam_transmission)
    return LayerOperator{FT}(
        convert(FT, reflection),
        convert(FT, transmission),
        convert(FT, beam_up),
        convert(FT, beam_down),
        convert(FT, beam_transmission),
    )
end

Base.eltype(::Type{<:LayerOperator{FT}}) where {FT} = FT
Base.convert(::Type{LayerOperator{FT}}, op::LayerOperator) where {FT} =
    _convert_record(LayerOperator{FT}, op)

# ============================================================================
# Even power series in q = m² h² (smooth through the conservative limit q → 0)
# ============================================================================

# cosh(√q)
@inline function even_cosh_c0(q)
    FT = typeof(q)
    r = one(FT)
    r = one(FT) + q * r * inv(FT(240))
    r = one(FT) + q * r * inv(FT(182))
    r = one(FT) + q * r * inv(FT(132))
    r = one(FT) + q * r * inv(FT(90))
    r = one(FT) + q * r * inv(FT(56))
    r = one(FT) + q * r * inv(FT(30))
    r = one(FT) + q * r * inv(FT(12))
    r = one(FT) + q * r * inv(FT(2))
    return r
end

# sinh(√q) / √q
@inline function even_sinh_s0(q)
    FT = typeof(q)
    r = one(FT)
    r = one(FT) + q * r * inv(FT(272))
    r = one(FT) + q * r * inv(FT(210))
    r = one(FT) + q * r * inv(FT(156))
    r = one(FT) + q * r * inv(FT(110))
    r = one(FT) + q * r * inv(FT(72))
    r = one(FT) + q * r * inv(FT(42))
    r = one(FT) + q * r * inv(FT(20))
    r = one(FT) + q * r * inv(FT(6))
    return r
end

# (cosh(√q) - 1) / q
@inline function even_cosh_m1_s1(q)
    FT = typeof(q)
    r = one(FT)
    r = one(FT) + q * r * inv(FT(306))
    r = one(FT) + q * r * inv(FT(240))
    r = one(FT) + q * r * inv(FT(182))
    r = one(FT) + q * r * inv(FT(132))
    r = one(FT) + q * r * inv(FT(90))
    r = one(FT) + q * r * inv(FT(56))
    r = one(FT) + q * r * inv(FT(30))
    r = one(FT) + q * r * inv(FT(12))
    return inv(FT(2)) * r
end

"""
    diffuse_response(absorption, backscatter, lai)

Compute exact diffuse slab reflectance, transmittance, and constant/linear
volume-emission source moments `(; reflection, transmission, source_zero, source_first)`
for a homogeneous slab of leaf area index `lai`.

Exponentially scaled `expm1` expressions prevent thick-slab overflow, and an
`e^{+z}`-canceled even power series in `q = α(α + 2b)L²` preserves smooth
first- and second-order automatic differentiation all the way through the
conservative limit `α → 0`.

# Examples
```jldoctest
julia> resp = diffuse_response(0.0, 0.25, 2.0);

julia> isapprox(resp.reflection + resp.transmission, 1.0; atol = 1e-15)
true
```
"""
@inline function diffuse_response(absorption::Real, backscatter::Real, lai::Real)
    FT = common_scalar_type(absorption, backscatter, lai)
    return _diffuse_response(
        convert(FT, absorption),
        convert(FT, backscatter),
        convert(FT, lai),
    )
end

# Arguments of one scalar type (checked and converted by the caller).
@inline function _diffuse_response(absorption, backscatter, lai)
    FT = typeof(lai)
    P = primal_type(FT)
    a = absorption + backscatter
    b = backscatter
    m2 = absorption * (absorption + FT(2) * b)
    q = m2 * lai * lai
    # Even series at q = 0 and, for dual numbers, at small q (smooth derivatives through
    # the conservative limit); closed form otherwise.
    if iszero(q) || (!(FT <: AbstractFloat) && primal(q) <= sqrt(sqrt(eps(P))))
        vt = lai * even_sinh_s0(q)
        wt = (lai * lai) * even_cosh_m1_s1(q)
        dt = even_cosh_c0(q) + a * vt
        return (;
            reflection = b * vt / dt,
            transmission = one(FT) / dt,
            source_zero = vt / dt,
            source_first = wt / dt,
        )
    else
        m = sqrt(m2)
        z = m * lai
        attenuation = exp(-z)
        v = -expm1(-FT(2) * z) / (FT(2) * m)
        eh = -expm1(-z) / m
        w = (eh * eh) / FT(2)
        denominator = (one(FT) + attenuation * attenuation) / FT(2) + a * v
        return (;
            reflection = b * v / denominator,
            transmission = attenuation / denominator,
            source_zero = v / denominator,
            source_first = w / denominator,
        )
    end
end

@inline function double_layer(p::LayerOperator)
    FT = eltype(p)
    r, t, u, d, beam =
        p.reflection, p.transmission, p.beam_up, p.beam_down, p.beam_transmission
    inverse = inv(one(FT) - r * r)
    return LayerOperator{FT}(
        r + t * t * r * inverse,
        t * t * inverse,
        u + t * (r * d + u * beam) * inverse,
        d * beam + t * (d + r * u * beam) * inverse,
        beam * beam,
    )
end

"""
    layer_operator(method::AbstractTwoStreamMethod, rates::CanopyRates, lai)

Assemble the dimensionless [`LayerOperator`](@ref) for a homogeneous layer of
leaf area index `lai` using `method` ([`NativeTwoStream`](@ref) or
[`AnalyticTwoStream`](@ref)).

A `lai` that is an exact zero (value and every derivative component, see
[`CanopyRadiation.exact_zero`](@ref)) returns the identity operator
`LayerOperator(0, 1, 0, 0, 1)`.

# Examples
```jldoctest
julia> r = CanopyRates(0.15, 0.08, 0.04, 0.05, 0.65);

julia> op_n = layer_operator(NativeTwoStream(0.02, 14), r, 1.2);

julia> op_a = layer_operator(AnalyticTwoStream(), r, 1.2);

julia> isapprox(op_n.beam_up, op_a.beam_up; rtol = 1e-12)
true
```
"""
@inline function layer_operator(
    method::AbstractTwoStreamMethod,
    rates::CanopyRates,
    lai::Real,
)
    FT = common_scalar_type(_precision_operands(method)..., rates, lai)
    return _layer_operator(method, convert(CanopyRates{FT}, rates), convert(FT, lai))
end

# Rates and lai of one scalar type (checked and converted by `layer_operator`).
@inline function _layer_operator(method::NativeTwoStream, rates::CanopyRates, lai)
    FT = typeof(lai)
    thin_norm = convert(FT, method.thin_norm)
    source_terms = method.source_terms
    exact_zero(lai) &&
        return LayerOperator{FT}(zero(FT), one(FT), zero(FT), zero(FT), one(FT))
    P = primal_type(FT)
    b, alpha, k = rates.backscatter, rates.absorption, rates.extinction
    a = alpha + b
    h = lai
    doublings = 0
    scale = max(a + b, k)
    while primal(h * scale) > primal(thin_norm)
        h /= FT(2)
        doublings += 1
    end
    m2 = alpha * (alpha + FT(2) * b)
    q_h = m2 * h * h
    v = if iszero(q_h) || (!(FT <: AbstractFloat) && primal(q_h) <= sqrt(sqrt(eps(P))))
        h * even_sinh_s0(q_h)
    else
        m = sqrt(m2)
        sinh(m * h) / m
    end
    diffuse = _diffuse_response(alpha, b, h)
    r, t = diffuse.reflection, diffuse.transmission
    td, tu, tb = zero(FT), zero(FT), one(FT)
    md, mu = zero(FT), zero(FT)
    for n in 1:source_terms
        factor = h / FT(n)
        td, tu, tb =
            factor * (-a * td + b * tu + rates.beam_down * tb),
            factor * (-b * td + a * tu - rates.beam_up * tb),
            -factor * k * tb
        md += td
        mu += tu
    end
    up = -mu * t
    down = md + b * v * up
    result = LayerOperator{FT}(r, t, up, down, exp(-k * h))
    for _ in 1:doublings
        result = double_layer(result)
    end
    return result
end

@inline function _exp_quotient_e1(u)
    FT = typeof(u)
    P = primal_type(FT)
    if primal(abs(u)) <= P(0.1)
        half = u * inv(FT(2))
        return exp(-half) * even_sinh_s0(half * half)
    else
        return -expm1(-u) / u
    end
end

@inline function _analytic_beam_j1(k, m, lai)
    FT = typeof(lai)
    P = primal_type(FT)
    diff = (k - m) * lai
    if primal(abs(diff)) <= P(0.1)
        half_diff = diff * inv(FT(2))
        return lai * exp(-inv(FT(2)) * (k + m) * lai) * even_sinh_s0(half_diff * half_diff)
    elseif primal(k) >= primal(m)
        return exp(-m * lai) * (-expm1(-diff) / (k - m))
    else
        return exp(-k * lai) * (-expm1(diff) / (m - k))
    end
end

@inline function _n_series_scalar(u, eu, p::Int)
    FT = typeof(u)
    acc = inv(FT(p + 1))
    uj = one(FT)
    for j in 1:72
        uj *= u * inv(FT(j))
        acc += uj * inv(FT(p + 1 + j))
    end
    return eu * acc
end

@inline function _small_q_beam_moments(u, q)
    FT = typeof(u)
    P = primal_type(FT)
    eu = exp(-u)
    if primal(u) <= P(18)
        m_hi = zero(FT)
        for idx in 64:-1:19
            m_hi = (u * m_hi + eu) * inv(FT(idx))
        end
        m18 = (u * m_hi + eu) * inv(FT(18))
        m17 = (u * m18 + eu) * inv(FT(17))
        ad = m17
        bd = m18
        au = _n_series_scalar(u, eu, 16)
        bu = _n_series_scalar(u, eu, 17)
        m_curr = m17
        for n in 8:-1:1
            c_even = inv(FT(2 * n * (2 * n - 1)))
            c_odd = inv(FT(2 * n * (2 * n + 1)))
            m_odd = (u * m_curr + eu) * inv(FT(2 * n))
            m_even = (u * m_odd + eu) * inv(FT(2 * n - 1))
            bd = m_odd + q * c_odd * bd
            ad = m_even + q * c_even * ad
            bu = _n_series_scalar(u, eu, 2 * n - 1) + q * c_odd * bu
            au = _n_series_scalar(u, eu, 2 * n - 2) + q * c_even * au
            m_curr = m_even
        end
        return ad, au, bd, bu
    else
        inv_u = inv(u)
        m_even = -expm1(-u) * inv_u
        n_even = m_even
        m_odd = (m_even - eu) * inv_u
        n_odd = (one(FT) - n_even) * inv_u
        w_even = one(FT)
        w_odd = one(FT)
        ad = m_even
        au = n_even
        bd = m_odd
        bu = n_odd
        m_prev = m_odd
        n_prev = n_odd
        for n in 1:8
            w_even *= q * inv(FT(2 * n * (2 * n - 1)))
            w_odd *= q * inv(FT(2 * n * (2 * n + 1)))
            k_even = 2 * n
            m_even = (FT(k_even) * m_prev - eu) * inv_u
            n_even = (one(FT) - FT(k_even) * n_prev) * inv_u
            k_odd = 2 * n + 1
            m_odd = (FT(k_odd) * m_even - eu) * inv_u
            n_odd = (one(FT) - FT(k_odd) * n_even) * inv_u
            ad += w_even * m_even
            au += w_even * n_even
            bd += w_odd * m_odd
            bu += w_odd * n_odd
            m_prev = m_odd
            n_prev = n_odd
        end
        return ad, au, bd, bu
    end
end

@inline function _layer_operator(::AnalyticTwoStream, rates::CanopyRates, lai)
    FT = typeof(lai)
    exact_zero(lai) &&
        return LayerOperator{FT}(zero(FT), one(FT), zero(FT), zero(FT), one(FT))
    P = primal_type(FT)
    alpha, b, su, sd, k =
        rates.absorption, rates.backscatter, rates.beam_up, rates.beam_down, rates.extinction
    a = alpha + b
    m2 = alpha * (alpha + FT(2) * b)
    q = m2 * lai * lai
    beam_trans = exp(-k * lai)
    if primal(q) <= sqrt(sqrt(eps(P)))
        vt = lai * even_sinh_s0(q)
        dt = even_cosh_c0(q) + a * vt
        refl = b * vt / dt
        trans = one(FT) / dt
        ad, au, bd, bu = _small_q_beam_moments(k * lai, q)
        ad_t = lai * ad
        au_t = lai * au
        bd_t = (lai * lai) * bd
        bu_t = (lai * lai) * bu
        up = (su * (au_t + a * bu_t) + sd * b * bu_t) / dt
        down = (sd * (ad_t + a * bd_t) + su * b * bd_t) / dt
        return LayerOperator{FT}(refl, trans, up, down, beam_trans)
    else
        m = sqrt(m2)
        z = m * lai
        ez = exp(-z)
        v = -expm1(-FT(2) * z) / (FT(2) * m)
        denom = (one(FT) + ez * ez) / FT(2) + a * v
        refl = b * v / denom
        trans = ez / denom
        j1 = _analytic_beam_j1(k, m, lai)
        j2 = lai * _exp_quotient_e1((k + m) * lai)
        inv2 = inv(FT(2))
        inv2m = inv(FT(2) * m)
        au = (j2 + ez * j1) * inv2
        bu = (j2 - ez * j1) * inv2m
        ad = (j1 + ez * j2) * inv2
        bd = (j1 - ez * j2) * inv2m
        up = (su * (au + a * bu) + sd * b * bu) / denom
        down = (sd * (ad + a * bd) + su * b * bd) / denom
        return LayerOperator{FT}(refl, trans, up, down, beam_trans)
    end
end

"""
    ColumnFluxWorkspace(::Type{FT}, n_layers::Integer)
    ColumnFluxWorkspace(direct, down, up, absorption)

Hold caller-owned column profile buffers for zero-scratch in-place passive
column solves via [`solve_column!`](@ref).

`direct`, `down`, and `up` store `n_layers + 1` top-to-bottom interface fluxes
and share identical index axes; `absorption` stores `n_layers` layer totals.

# Examples
```jldoctest
julia> ws = ColumnFluxWorkspace(Float64, 3);

julia> (length(ws.down), length(ws.absorption))
(4, 3)
```
"""
struct ColumnFluxWorkspace{FT<:Real,V<:AbstractVector{FT},A<:AbstractVector{FT}}
    direct::V
    down::V
    up::V
    absorption::A

    function ColumnFluxWorkspace(
        direct::V,
        down::V,
        up::V,
        absorption::A,
    ) where {FT<:Real,V<:AbstractVector{FT},A<:AbstractVector{FT}}
        same_precision(direct)
        axes(direct) == axes(down) == axes(up) ||
            throw(DimensionMismatch("ColumnFluxWorkspace interface vectors must share axes"))
        length(direct) == length(absorption) + 1 ||
            throw(DimensionMismatch("ColumnFluxWorkspace requires length(direct) == length(absorption) + 1"))
        _disjoint_hemispheric_buffers((direct, down, up, absorption), ())
        return new{FT,V,A}(direct, down, up, absorption)
    end
end

function ColumnFluxWorkspace(::Type{FT}, n_layers::Integer) where {FT<:Real}
    same_precision(FT)
    n_layers >= 0 || throw(ArgumentError("n_layers must be nonnegative"))
    n = Int(n_layers)
    return ColumnFluxWorkspace(
        Vector{FT}(undef, n + 1),
        Vector{FT}(undef, n + 1),
        Vector{FT}(undef, n + 1),
        Vector{FT}(undef, n),
    )
end

Base.eltype(::Type{<:ColumnFluxWorkspace{FT}}) where {FT} = FT

"""
    solve_column!(ws::ColumnFluxWorkspace, layers::AbstractVector{<:LayerOperator}, direct, diffuse, soil)

Solve a top-to-bottom stack of [`LayerOperator`](@ref)s over Lambertian ground
reflectance `soil` into `ws` with zero scratch allocations.

During the backward adding sweep, `ws.down` and `ws.up` temporarily hold
interface effective reflectance and beam-induced upward source before being
overwritten in place during the forward flux sweep, matching [`solve_column`](@ref)
on a tuple in exact IEEE-754 arithmetic order for one or more layers (for zero
layers the upward flux and ground absorption can differ by up to 2 ulp, see
[`solve_column`](@ref)).

The inputs must have the precision of `ws` and promote to its element type
(`ArgumentError` otherwise).
"""
@inline function solve_column!(
    ws::ColumnFluxWorkspace,
    layers::AbstractVector{<:LayerOperator},
    direct::Real,
    diffuse::Real,
    soil::Real,
)
    FT = eltype(ws)
    common_scalar_type(ws.direct, layers, direct, diffuse, soil) === FT ||
        throw(ArgumentError("ColumnFluxWorkspace element type must match promoted input type"))
    n = length(layers)
    length(ws.absorption) == n ||
        throw(DimensionMismatch("Workspace layer count does not match layers"))
    # The sweep indexes without bounds checks: the vectors may have been resized since
    # the workspace was built.
    length(ws.direct) == length(ws.down) == length(ws.up) == n + 1 ||
        throw(DimensionMismatch("Workspace interface vectors must have length(layers) + 1 entries"))
    ground = _passive_column_sweep!(
        column_view(ws.direct),
        column_view(ws.down),
        column_view(ws.up),
        column_view(ws.absorption),
        column_view(layers),
        n,
        convert(FT, direct),
        convert(FT, diffuse),
        convert(FT, soil),
    )
    return (;
        direct = ws.direct,
        down = ws.down,
        up = ws.up,
        absorption = ws.absorption,
        ground,
    )
end

"""
    solve_column(layers, direct, diffuse, soil)

Solve a top-to-bottom stack of [`LayerOperator`](@ref)s (`NTuple` or
`AbstractVector`) under top horizontal `direct` and `diffuse` incident fluxes
above Lambertian ground reflectance `soil`.

Return `(; direct, down, up, absorption, ground)` satisfying column flux
conservation `direct[1] + diffuse == up[1] + sum(absorption) + ground`.

The tuple and vector methods (and [`batch_solve_columns!`](@ref)) evaluate the same
expressions in the same order for one or more layers, so their results are
bit-identical. For an empty column the tuple method computes the reflected flux as
`soil * (direct + diffuse)` and the vector method as `soil * diffuse + soil * direct`;
the two can differ by up to 2 ulp (as in v0.1.0).

# Examples
```jldoctest
julia> r = CanopyRates(0.15, 0.08, 0.04, 0.05, 0.65);

julia> ops = (layer_operator(AnalyticTwoStream(), r, 1.0), layer_operator(AnalyticTwoStream(), r, 1.5));

julia> sol = solve_column(ops, 0.7, 0.3, 0.15);

julia> isapprox(sol.up[1] + sum(sol.absorption) + sol.ground, 1.0; atol = 1e-14)
true
```
"""
@inline function solve_column(
    ::Tuple{},
    direct::Real,
    diffuse::Real,
    soil::Real,
)
    FT0 = common_scalar_type(direct, diffuse, soil)
    d0 = convert(FT0, direct)
    f0 = convert(FT0, diffuse)
    s0 = convert(FT0, soil)
    u0 = s0 * (d0 + f0)
    g0 = (one(FT0) - s0) * (d0 + f0)
    return (; direct = (d0,), down = (f0,), up = (u0,), absorption = (), ground = g0)
end

@inline function solve_column(
    layers::Tuple{LayerOperator,Vararg{LayerOperator}},
    direct::Real,
    diffuse::Real,
    soil::Real,
)
    FT = common_scalar_type(_layer_operands(layers)..., direct, diffuse, soil)
    return _solve_column_tuple(
        _convert_tuple(LayerOperator{FT}, layers),
        convert(FT, direct),
        convert(FT, diffuse),
        convert(FT, soil),
    )
end

@inline _beam_sweep_tuple(::Tuple{}, b::FT) where {FT<:Real} = (b,)
@inline function _beam_sweep_tuple(
    layers::Tuple{LayerOperator{FT},Vararg{LayerOperator{FT}}},
    b::FT,
) where {FT<:Real}
    p = first(layers)
    return (b, _beam_sweep_tuple(Base.tail(layers), b * p.beam_transmission)...)
end

@inline _passive_back_tuple(::Tuple{}, beam::Tuple{FT}, soil::FT) where {FT<:Real} =
    ((soil,), (soil * first(beam),))
@inline function _passive_back_tuple(
    layers::Tuple{LayerOperator{FT},Vararg{LayerOperator{FT}}},
    beam::Tuple{FT,FT,Vararg{FT}},
    soil::FT,
) where {FT<:Real}
    p = first(layers)
    b_i = first(beam)
    r_tail, s_tail = _passive_back_tuple(Base.tail(layers), Base.tail(beam), soil)
    r_next = first(r_tail)
    s_next = first(s_tail)
    den = inv(one(FT) - p.reflection * r_next)
    r_curr = p.reflection + p.transmission^2 * r_next * den
    s_curr =
        p.beam_up * b_i +
        p.transmission * den * (s_next + r_next * p.beam_down * b_i)
    return ((r_curr, r_tail...), (s_curr, s_tail...))
end

@inline _passive_fwd_tuple(
    ::Tuple{},
    ::Tuple{FT},
    ::Tuple{FT},
    ::Tuple{FT},
    d_curr::FT,
    u_curr::FT,
) where {FT<:Real} = ((d_curr,), (u_curr,), ())

@inline function _passive_fwd_tuple(
    layers::Tuple{LayerOperator{FT},Vararg{LayerOperator{FT}}},
    beam::Tuple{FT,FT,Vararg{FT}},
    reflect::Tuple{FT,FT,Vararg{FT}},
    source::Tuple{FT,FT,Vararg{FT}},
    d_curr::FT,
    u_curr::FT,
) where {FT<:Real}
    p = first(layers)
    b_i = first(beam)
    b_rest = Base.tail(beam)
    r_rest = Base.tail(reflect)
    s_rest = Base.tail(source)
    r_next = first(r_rest)
    s_next = first(s_rest)
    d_next =
        (p.transmission * d_curr + p.reflection * s_next + p.beam_down * b_i) /
        (one(FT) - p.reflection * r_next)
    u_next = r_next * d_next + s_next
    abs_i = d_curr + b_i + u_next - d_next - first(b_rest) - u_curr
    d_tail, u_tail, abs_tail =
        _passive_fwd_tuple(Base.tail(layers), b_rest, r_rest, s_rest, d_next, u_next)
    return ((d_curr, d_tail...), (u_curr, u_tail...), (abs_i, abs_tail...))
end

@inline function _solve_column_tuple(
    layers::NTuple{N,LayerOperator{FT}},
    direct::FT,
    diffuse::FT,
    soil::FT,
) where {N,FT<:Real}
    beam = _beam_sweep_tuple(layers, direct)
    reflect, source = _passive_back_tuple(layers, beam, soil)
    down, up, absorption = _passive_fwd_tuple(
        layers,
        beam,
        reflect,
        source,
        diffuse,
        first(reflect) * diffuse + first(source),
    )
    ground = (one(FT) - soil) * (down[end] + beam[end])
    return (; direct = beam, down, up, absorption, ground)
end

function solve_column(
    layers::AbstractVector{<:LayerOperator},
    direct::Real,
    diffuse::Real,
    soil::Real,
)
    FT = common_scalar_type(layers, direct, diffuse, soil)
    ws = ColumnFluxWorkspace(FT, length(layers))
    return solve_column!(ws, layers, direct, diffuse, soil)
end

"""
    interior_flux(method, rates, lai, x, down_top, up_bottom, beam_top)

Reconstruct `(; down, up, beam)` inside a homogeneous layer of total leaf area
index `lai` at local depth `x ∈ [0, lai]` from solved interface boundary fluxes
`down_top`, `up_bottom`, and `beam_top`.

# Examples
```jldoctest
julia> r = CanopyRates(0.15, 0.08, 0.04, 0.05, 0.65);

julia> m = AnalyticTwoStream();

julia> op = layer_operator(m, r, 1.4);

julia> sol = solve_column((op,), 0.8, 0.2, 0.1);

julia> mid = interior_flux(m, r, 1.4, 0.7, sol.down[1], sol.up[2], sol.direct[1]);

julia> mid.down > 0 && mid.up > 0 && mid.beam > 0
true
```
"""
@inline function interior_flux(
    method::AbstractTwoStreamMethod,
    rates::CanopyRates,
    lai::Real,
    x::Real,
    down_top::Real,
    up_bottom::Real,
    beam_top::Real,
)
    FT = common_scalar_type(
        _precision_operands(method)...,
        rates,
        lai,
        x,
        down_top,
        up_bottom,
        beam_top,
    )
    lai_f = convert(FT, lai)
    x_f = convert(FT, x)
    dt = convert(FT, down_top)
    ub = convert(FT, up_bottom)
    bt = convert(FT, beam_top)
    rates_f = convert(CanopyRates{FT}, rates)
    upper = layer_operator(method, rates_f, x_f)
    lower = layer_operator(method, rates_f, lai_f - x_f)
    beam = bt * upper.beam_transmission
    source = lower.transmission * ub + lower.beam_up * beam
    down =
        (upper.transmission * dt + upper.reflection * source + upper.beam_down * bt) /
        (one(FT) - upper.reflection * lower.reflection)
    up = lower.reflection * down + source
    return (; down, up, beam)
end
