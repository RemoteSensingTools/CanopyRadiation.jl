"""
    AbstractVolumeEmission

Supertype for internal hemispheric volume-emission closures within a canopy
layer ([`NoEmission`](@ref), [`UniformEmission`](@ref), and
[`QuadratureEmission`](@ref)).
"""
abstract type AbstractVolumeEmission <: RadiationRecord end

"""
    NoEmission()

Zero internal volume emission within a canopy slab.

# Examples
```jldoctest
julia> NoEmission()
NoEmission()
```
"""
struct NoEmission <: AbstractVolumeEmission end

"""
    UniformEmission(upward, downward)

Prescribe constant upward and downward hemispheric emission rates per unit
one-sided leaf area index (`LAI`) across a homogeneous canopy layer.

# Examples
```jldoctest
julia> UniformEmission(0.4, 0.35)
UniformEmission{Float64}(0.4, 0.35)
```
"""
struct UniformEmission{FT<:Real} <: AbstractVolumeEmission
    upward::FT
    downward::FT

    function UniformEmission{FT}(upward::FT, downward::FT) where {FT<:Real}
        same_precision(upward)
        return new{FT}(upward, downward)
    end
end

@inline function UniformEmission(upward::Real, downward::Real)
    FT = common_scalar_type(upward, downward)
    return UniformEmission{FT}(convert(FT, upward), convert(FT, downward))
end

Base.eltype(::Type{<:UniformEmission{FT}}) where {FT} = FT
Base.convert(::Type{UniformEmission{FT}}, e::UniformEmission) where {FT} =
    _convert_record(UniformEmission{FT}, e)

# Volume emission entering the precision of a layer computation (none for NoEmission).
@inline _precision_operands(::NoEmission) = ()
@inline _precision_operands(e::UniformEmission) = (e,)

"""
    QuadratureEmission(depths, weights, upward, downward)

Represent a depth-varying hemispheric emission source sampled at unit-interval
fractional depths `depths ∈ [0, 1]` with quadrature `weights` and per-`LAI`
hemispheric source densities `upward` and `downward`.

All four vectors must share identical axes and one floating-point element type
`FT`, with finite nonnegative weights and source strengths (`ArgumentError`
otherwise). `validate = false` skips the value checks (used for device transfers).

# Examples
```jldoctest
julia> q = QuadratureEmission([0.25, 0.75], [0.5, 0.5], [0.3, 0.2], [0.3, 0.2]);

julia> length(q.depths)
2
```
"""
struct QuadratureEmission{FT<:Real,V<:AbstractVector{FT}} <: AbstractVolumeEmission
    depths::V
    weights::V
    upward::V
    downward::V

    function QuadratureEmission(
        depths::V,
        weights::V,
        upward::V,
        downward::V;
        validate::Bool = true,
    ) where {FT<:Real,V<:AbstractVector{FT}}
        same_precision(depths)
        axes(depths) == axes(weights) == axes(upward) == axes(downward) ||
            throw(DimensionMismatch("Source depths, weights, and strengths must share axes"))
        if validate
            for x in _host_linear(depths)
                _checked(x, _unit_interval, "Source depths must be finite and lie in [0, 1]")
            end
            for vec in (weights, upward, downward)
                for x in _host_linear(vec)
                    _checked(
                        x,
                        _nonnegative,
                        "Source weights and strengths must be finite and nonnegative",
                    )
                end
            end
        end
        return new{FT,V}(depths, weights, upward, downward)
    end
end

Base.eltype(::Type{<:QuadratureEmission{FT}}) where {FT} = FT

"""
    EmittingLayer(passive::LayerOperator, upward, downward, generated)

Bundle a layer's passive diffuse [`LayerOperator`](@ref) with the upward and
downward fluxes escaping its boundaries from internal volume emission and its
total `generated` emission per unit horizontal ground area.

# Examples
```jldoctest
julia> el = emitting_layer(0.1, 0.08, 1.5, UniformEmission(0.2, 0.2));

julia> isapprox(el.generated, 0.6; atol = 1e-15)
true
```
"""
struct EmittingLayer{FT<:Real} <: RadiationRecord
    passive::LayerOperator{FT}
    upward::FT
    downward::FT
    generated::FT

    function EmittingLayer{FT}(
        passive::LayerOperator{FT},
        upward::FT,
        downward::FT,
        generated::FT,
    ) where {FT<:Real}
        same_precision(upward)
        return new{FT}(passive, upward, downward, generated)
    end
end

@inline function EmittingLayer(
    passive::LayerOperator,
    upward::Real,
    downward::Real,
    generated::Real,
)
    FT = common_scalar_type(passive, upward, downward, generated)
    return EmittingLayer{FT}(
        convert(LayerOperator{FT}, passive),
        convert(FT, upward),
        convert(FT, downward),
        convert(FT, generated),
    )
end

Base.eltype(::Type{<:EmittingLayer{FT}}) where {FT} = FT
Base.convert(::Type{EmittingLayer{FT}}, e::EmittingLayer) where {FT} =
    _convert_record(EmittingLayer{FT}, e)

"""
    point_emission_response(absorption, backscatter, lai, x, upward, downward)

Compute boundary escapes and immediate internal interface fluxes
`(; upward, downward, up_at_source, down_at_source)` for an integrated planar
source `(upward, downward)` at local `LAI` depth `x ∈ [0, lai]` inside a
passive homogeneous slab.

# Examples
```jldoctest
julia> r = point_emission_response(0.0, 0.2, 1.0, 0.4, 0.5, 0.5);

julia> isapprox(r.upward + r.downward, 1.0; atol = 1e-14)
true
```
"""
@inline function point_emission_response(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    x::Real,
    upward::Real,
    downward::Real,
)
    FT = common_scalar_type(absorption, backscatter, lai, x, upward, downward)
    return _point_emission_response(
        convert(FT, absorption),
        convert(FT, backscatter),
        convert(FT, lai),
        convert(FT, x),
        convert(FT, upward),
        convert(FT, downward),
    )
end

# Arguments of one scalar type (checked and converted by the caller).
@inline function _point_emission_response(absorption, backscatter, lai, x, upward, downward)
    FT = typeof(lai)
    upper = _diffuse_response(absorption, backscatter, x)
    lower = _diffuse_response(absorption, backscatter, lai - x)
    inverse = inv(one(FT) - upper.reflection * lower.reflection)
    up_at_source = (upward + lower.reflection * downward) * inverse
    down_at_source = (downward + upper.reflection * upward) * inverse
    return (;
        upward = upper.transmission * up_at_source,
        downward = lower.transmission * down_at_source,
        up_at_source,
        down_at_source,
    )
end

"""
    quadrature_emitting_layer(absorption, backscatter, lai, depths, weights, upward, downward)

Integrate point-source Green's-function escapes over unit-interval `depths` and
`weights` to form an [`EmittingLayer`](@ref).

# Examples
```jldoctest
julia> el = quadrature_emitting_layer(0.1, 0.05, 1.2, [0.5], [1.0], [0.3], [0.2]);

julia> isapprox(el.generated, 0.6; atol = 1e-15)
true
```
"""
@inline function quadrature_emitting_layer(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    depths::AbstractVector,
    weights::AbstractVector,
    upward::AbstractVector,
    downward::AbstractVector,
)
    axes(depths) == axes(weights) == axes(upward) == axes(downward) ||
        throw(DimensionMismatch("Quadrature source vectors must share axes"))
    FT = common_scalar_type(absorption, backscatter, lai, depths, weights, upward, downward)
    abs_f = convert(FT, absorption)
    back_f = convert(FT, backscatter)
    lai_f = convert(FT, lai)
    p = _diffuse_response(abs_f, back_f, lai_f)
    passive = LayerOperator{FT}(p.reflection, p.transmission, zero(FT), zero(FT), one(FT))
    up, down, generated = zero(FT), zero(FT), zero(FT)
    @inbounds for k in eachindex(depths, weights, upward, downward)
        measure = lai_f * convert(FT, weights[k])
        qu = measure * convert(FT, upward[k])
        qd = measure * convert(FT, downward[k])
        resp = _point_emission_response(
            abs_f,
            back_f,
            lai_f,
            lai_f * convert(FT, depths[k]),
            qu,
            qd,
        )
        up += resp.upward
        down += resp.downward
        generated += qu + qd
    end
    return EmittingLayer{FT}(passive, up, down, generated)
end

"""
    emitting_layer(absorption, backscatter, lai, emission::AbstractVolumeEmission)

Construct an [`EmittingLayer`](@ref) for `emission` (`NoEmission`,
`UniformEmission`, or `QuadratureEmission`) on a homogeneous slab of leaf area
index `lai`.

# Examples
```jldoctest
julia> el = emitting_layer(0.0, 0.2, 1.5, UniformEmission(0.4, 0.6));

julia> isapprox(el.upward + el.downward, el.generated; atol = 1e-14)
true
```
"""
@inline function emitting_layer(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    ::NoEmission,
)
    FT = common_scalar_type(absorption, backscatter, lai)
    p = _diffuse_response(convert(FT, absorption), convert(FT, backscatter), convert(FT, lai))
    passive = LayerOperator{FT}(p.reflection, p.transmission, zero(FT), zero(FT), one(FT))
    return EmittingLayer{FT}(passive, zero(FT), zero(FT), zero(FT))
end

@inline function emitting_layer(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    emission::UniformEmission,
)
    FT = common_scalar_type(absorption, backscatter, lai, emission)
    abs_f = convert(FT, absorption)
    b = convert(FT, backscatter)
    lai_f = convert(FT, lai)
    qu = convert(FT, emission.upward)
    qd = convert(FT, emission.downward)
    a = abs_f + b
    p = _diffuse_response(abs_f, b, lai_f)
    passive = LayerOperator{FT}(p.reflection, p.transmission, zero(FT), zero(FT), one(FT))
    up = p.source_zero * qu + p.source_first * (a * qu + b * qd)
    down = p.source_zero * qd + p.source_first * (a * qd + b * qu)
    return EmittingLayer{FT}(passive, up, down, (qu + qd) * lai_f)
end

@inline function emitting_layer(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    source::QuadratureEmission,
)
    return quadrature_emitting_layer(
        absorption,
        backscatter,
        lai,
        source.depths,
        source.weights,
        source.upward,
        source.downward,
    )
end

"""
    EmittingColumnWorkspace(::Type{FT}, n_layers::Integer)
    EmittingColumnWorkspace(down, up, net_layer, absorption)

Hold caller-owned column buffers for zero-scratch in-place cavity emission and
longwave solves via [`solve_emitting_column!`](@ref).

`down` and `up` store `n_layers + 1` interface fluxes sharing identical axes;
`net_layer` and `absorption` store `n_layers` layer values sharing identical axes.

# Examples
```jldoctest
julia> ws = EmittingColumnWorkspace(Float64, 3);

julia> (length(ws.down), length(ws.net_layer))
(4, 3)
```
"""
struct EmittingColumnWorkspace{FT<:Real,V<:AbstractVector{FT},A<:AbstractVector{FT}}
    down::V
    up::V
    net_layer::A
    absorption::A

    function EmittingColumnWorkspace(
        down::V,
        up::V,
        net_layer::A,
        absorption::A,
    ) where {FT<:Real,V<:AbstractVector{FT},A<:AbstractVector{FT}}
        same_precision(down)
        axes(down) == axes(up) ||
            throw(DimensionMismatch("EmittingColumnWorkspace down and up must share axes"))
        axes(net_layer) == axes(absorption) ||
            throw(DimensionMismatch("EmittingColumnWorkspace net_layer and absorption must share axes"))
        length(down) == length(absorption) + 1 ||
            throw(DimensionMismatch("EmittingColumnWorkspace requires length(down) == length(absorption) + 1"))
        _disjoint_hemispheric_buffers((down, up, net_layer, absorption), ())
        return new{FT,V,A}(down, up, net_layer, absorption)
    end
end

function EmittingColumnWorkspace(::Type{FT}, n_layers::Integer) where {FT<:Real}
    same_precision(zero(FT))
    n_layers >= 0 || throw(ArgumentError("n_layers must be nonnegative"))
    n = Int(n_layers)
    return EmittingColumnWorkspace(
        Vector{FT}(undef, n + 1),
        Vector{FT}(undef, n + 1),
        Vector{FT}(undef, n),
        Vector{FT}(undef, n),
    )
end

Base.eltype(::Type{<:EmittingColumnWorkspace{FT}}) where {FT} = FT

"""
    solve_emitting_column!(ws::EmittingColumnWorkspace, layers::AbstractVector{<:EmittingLayer}, down_top, soil_reflect, soil_emit)

Solve a diffuse cavity with internal layer emission, top downward boundary flux
`down_top` (such as sky thermal longwave), ground reflectance `soil_reflect`,
and upward ground emission `soil_emit` into `ws` with zero scratch allocations.

Return `(; down, up, net_layer, absorption, net_soil, ground_absorbed, generated)`
satisfying both net divergence balance (`down_top - up[1] == sum(net_layer) + net_soil`)
and gross radiative balance (`down_top + generated + soil_emit == up[1] + sum(absorption) + ground_absorbed`).

The inputs must have the precision of `ws` and promote to its element type
(`ArgumentError` otherwise).
"""
@inline function solve_emitting_column!(
    ws::EmittingColumnWorkspace,
    layers::AbstractVector{<:EmittingLayer},
    down_top::Real,
    soil_reflect::Real,
    soil_emit::Real,
)
    FT = eltype(ws)
    common_scalar_type(ws.down, layers, down_top, soil_reflect, soil_emit) === FT ||
        throw(ArgumentError("EmittingColumnWorkspace element type must match promoted input type"))
    n = length(layers)
    length(ws.absorption) == n ||
        throw(DimensionMismatch("Workspace layer count does not match layers"))
    dt0 = convert(FT, down_top)
    sr0 = convert(FT, soil_reflect)
    se0 = convert(FT, soil_emit)
    f0 = firstindex(ws.down)
    l0 = firstindex(layers)
    a0 = firstindex(ws.absorption)
    gen_sum = zero(FT)
    @inbounds begin
        ws.down[f0 + n] = sr0
        ws.up[f0 + n] = se0
        for s in (n - 1):-1:0
            e = convert(EmittingLayer{FT}, layers[l0 + s])
            p = e.passive
            r_next = ws.down[f0 + s + 1]
            s_next = ws.up[f0 + s + 1]
            inv_den = inv(one(FT) - p.reflection * r_next)
            ws.down[f0 + s] = p.reflection + p.transmission^2 * r_next * inv_den
            ws.up[f0 + s] =
                e.upward + p.transmission * (s_next + r_next * e.downward) * inv_den
        end
        r1 = ws.down[f0]
        s1 = ws.up[f0]
        ws.down[f0] = dt0
        ws.up[f0] = r1 * dt0 + s1
        for s in 0:(n - 1)
            e = convert(EmittingLayer{FT}, layers[l0 + s])
            p = e.passive
            d_curr = ws.down[f0 + s]
            u_curr = ws.up[f0 + s]
            r_next = ws.down[f0 + s + 1]
            s_next = ws.up[f0 + s + 1]
            val =
                (p.transmission * d_curr + p.reflection * s_next + e.downward) /
                (one(FT) - p.reflection * r_next)
            u_next = r_next * val + s_next
            ws.down[f0 + s + 1] = val
            ws.up[f0 + s + 1] = u_next
            net_l = (d_curr - u_curr) - (val - u_next)
            ws.net_layer[a0 + s] = net_l
            ws.absorption[a0 + s] = e.generated + net_l
            gen_sum += e.generated
        end
        net_soil = ws.down[f0 + n] - ws.up[f0 + n]
        ground_absorbed = (one(FT) - sr0) * ws.down[f0 + n]
    end
    return (;
        down = ws.down,
        up = ws.up,
        net_layer = ws.net_layer,
        absorption = ws.absorption,
        net_soil,
        ground_absorbed,
        generated = gen_sum,
    )
end

"""
    solve_emitting_column(layers, down_top, soil_reflect, soil_emit)

Solve a top-to-bottom stack of [`EmittingLayer`](@ref)s (`NTuple` or
`AbstractVector`) under top downward flux `down_top`, Lambertian ground
reflectance `soil_reflect`, and upward ground emission `soil_emit`.

Return the 7-field NamedTuple
`(; down, up, net_layer, absorption, net_soil, ground_absorbed, generated)`.

# Examples
```jldoctest
julia> l1 = emitting_layer(0.9, 0.05, 1.2, UniformEmission(180.0, 180.0));

julia> sol = solve_emitting_column((l1,), 320.0, 0.04, 390.0);

julia> isapprox(320.0 + sol.generated + 390.0, sol.up[1] + sum(sol.absorption) + sol.ground_absorbed; rtol = 1e-13)
true
```
"""
@inline function solve_emitting_column(
    ::Tuple{},
    down_top::Real,
    soil_reflect::Real,
    soil_emit::Real,
)
    FT0 = common_scalar_type(down_top, soil_reflect, soil_emit)
    dt0 = convert(FT0, down_top)
    sr0 = convert(FT0, soil_reflect)
    se0 = convert(FT0, soil_emit)
    u0 = sr0 * dt0 + se0
    return (;
        down = (dt0,),
        up = (u0,),
        net_layer = (),
        absorption = (),
        net_soil = dt0 - u0,
        ground_absorbed = (one(FT0) - sr0) * dt0,
        generated = zero(FT0),
    )
end

@inline function solve_emitting_column(
    layers::Tuple{EmittingLayer,Vararg{EmittingLayer}},
    down_top::Real,
    soil_reflect::Real,
    soil_emit::Real,
)
    FT = common_scalar_type(_precision_operands(layers)..., down_top, soil_reflect, soil_emit)
    return _solve_emitting_column_tuple(
        _convert_tuple(EmittingLayer{FT}, layers),
        convert(FT, down_top),
        convert(FT, soil_reflect),
        convert(FT, soil_emit),
    )
end

@inline _emitting_back_tuple(::Tuple{}, soil_reflect::FT, soil_emit::FT) where {FT<:Real} =
    ((soil_reflect,), (soil_emit,))
@inline function _emitting_back_tuple(
    layers::Tuple{EmittingLayer{FT},Vararg{EmittingLayer{FT}}},
    soil_reflect::FT,
    soil_emit::FT,
) where {FT<:Real}
    e = first(layers)
    p = e.passive
    r_tail, s_tail = _emitting_back_tuple(Base.tail(layers), soil_reflect, soil_emit)
    r_next = first(r_tail)
    s_next = first(s_tail)
    inverse = inv(one(FT) - p.reflection * r_next)
    r_curr = p.reflection + p.transmission^2 * r_next * inverse
    s_curr = e.upward + p.transmission * (s_next + r_next * e.downward) * inverse
    return ((r_curr, r_tail...), (s_curr, s_tail...))
end

@inline _emitting_fwd_tuple(
    ::Tuple{},
    ::Tuple{FT},
    ::Tuple{FT},
    d_curr::FT,
    u_curr::FT,
    gen_acc::FT,
) where {FT<:Real} = ((d_curr,), (u_curr,), (), (), gen_acc)

@inline function _emitting_fwd_tuple(
    layers::Tuple{EmittingLayer{FT},Vararg{EmittingLayer{FT}}},
    reflect::Tuple{FT,FT,Vararg{FT}},
    source::Tuple{FT,FT,Vararg{FT}},
    d_curr::FT,
    u_curr::FT,
    gen_acc::FT,
) where {FT<:Real}
    e = first(layers)
    p = e.passive
    r_rest = Base.tail(reflect)
    s_rest = Base.tail(source)
    r_next = first(r_rest)
    s_next = first(s_rest)
    d_next =
        (p.transmission * d_curr + p.reflection * s_next + e.downward) /
        (one(FT) - p.reflection * r_next)
    u_next = r_next * d_next + s_next
    net_l = (d_curr - u_curr) - (d_next - u_next)
    abs_l = e.generated + net_l
    d_tail, u_tail, net_tail, abs_tail, gen_tot =
        _emitting_fwd_tuple(Base.tail(layers), r_rest, s_rest, d_next, u_next, gen_acc + e.generated)
    return (
        (d_curr, d_tail...),
        (u_curr, u_tail...),
        (net_l, net_tail...),
        (abs_l, abs_tail...),
        gen_tot,
    )
end

@inline function _solve_emitting_column_tuple(
    layers::NTuple{N,EmittingLayer{FT}},
    down_top::FT,
    soil_reflect::FT,
    soil_emit::FT,
) where {N,FT<:Real}
    reflect, source = _emitting_back_tuple(layers, soil_reflect, soil_emit)
    down, up, net_layer, absorption, generated = _emitting_fwd_tuple(
        layers,
        reflect,
        source,
        down_top,
        first(reflect) * down_top + first(source),
        zero(FT),
    )
    return (;
        down,
        up,
        net_layer,
        absorption,
        net_soil = down[end] - up[end],
        ground_absorbed = (one(FT) - soil_reflect) * down[end],
        generated,
    )
end

function solve_emitting_column(
    layers::AbstractVector{<:EmittingLayer},
    down_top::Real,
    soil_reflect::Real,
    soil_emit::Real,
)
    FT = common_scalar_type(layers, down_top, soil_reflect, soil_emit)
    ws = EmittingColumnWorkspace(FT, length(layers))
    return solve_emitting_column!(ws, layers, down_top, soil_reflect, soil_emit)
end

"""
    solve_emission(layers, soil_reflect)

Solve a zero-incident source-only stack of [`EmittingLayer`](@ref)s (`NTuple` or
`AbstractVector`) above passive Lambertian ground reflectance `soil_reflect`.

Return `(; down, up, absorption, ground, generated)` satisfying
`generated == up[1] + sum(absorption) + ground`.

# Examples
```jldoctest
julia> l1 = emitting_layer(0.08, 0.12, 1.5, UniformEmission(0.3, 0.25));

julia> sol = solve_emission((l1,), 0.2);

julia> isapprox(sol.up[1] + sum(sol.absorption) + sol.ground, sol.generated; atol = 1e-14)
true
```
"""
@inline function solve_emission(
    ::Tuple{},
    soil_reflect::Real,
)
    FT0 = common_scalar_type(soil_reflect)
    z0 = zero(FT0)
    return (; down = (z0,), up = (z0,), absorption = (), ground = z0, generated = z0)
end

@inline function solve_emission(
    layers::Tuple{EmittingLayer,Vararg{EmittingLayer}},
    soil_reflect::Real,
)
    FT = common_scalar_type(_precision_operands(layers)..., soil_reflect)
    return _solve_emission_tuple(_convert_tuple(EmittingLayer{FT}, layers), convert(FT, soil_reflect))
end

@inline function _solve_emission_tuple(
    layers::NTuple{N,EmittingLayer{FT}},
    soil::FT,
) where {N,FT<:Real}
    reflect, source = _emitting_back_tuple(layers, soil, zero(FT))
    down, up, _, absorption, generated = _emitting_fwd_tuple(
        layers,
        reflect,
        source,
        zero(FT),
        first(source),
        zero(FT),
    )
    return (;
        down,
        up,
        absorption,
        ground = (one(FT) - soil) * down[end],
        generated,
    )
end

function solve_emission(layers::AbstractVector{<:EmittingLayer}, soil_reflect::Real)
    FT = common_scalar_type(layers, soil_reflect)
    res = solve_emitting_column(layers, zero(FT), convert(FT, soil_reflect), zero(FT))
    return (;
        down = res.down,
        up = res.up,
        absorption = res.absorption,
        ground = res.ground_absorbed,
        generated = res.generated,
    )
end

"""
    point_emission_interior(absorption, backscatter, lai, source_depth, x, upward, downward)

Evaluate internal `(; down, up)` fluxes at observation depth `x ∈ [0, lai]` due
to an integrated planar source `(upward, downward)` at `source_depth ∈ [0, lai]`
with homogeneous zero-flux boundaries.

At `x == source_depth`, return the upper-side limit.

# Examples
```jldoctest
julia> f = point_emission_interior(0.1, 0.08, 1.0, 0.5, 0.25, 0.4, 0.4);

julia> f.up > 0 && f.down >= 0
true
```
"""
@inline function point_emission_interior(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    source_depth::Real,
    x::Real,
    upward::Real,
    downward::Real,
)
    FT = common_scalar_type(absorption, backscatter, lai, source_depth, x, upward, downward)
    abs_f = convert(FT, absorption)
    back_f = convert(FT, backscatter)
    lai_f = convert(FT, lai)
    sd_f = convert(FT, source_depth)
    x_f = convert(FT, x)
    up_f = convert(FT, upward)
    dn_f = convert(FT, downward)
    resp = _point_emission_response(abs_f, back_f, lai_f, sd_f, up_f, dn_f)
    if primal(x_f) <= primal(sd_f)
        return emission_interior(
            abs_f,
            back_f,
            sd_f,
            x_f,
            NoEmission(),
            zero(FT),
            resp.up_at_source,
        )
    else
        return emission_interior(
            abs_f,
            back_f,
            lai_f - sd_f,
            x_f - sd_f,
            NoEmission(),
            resp.down_at_source,
            zero(FT),
        )
    end
end

"""
    emission_interior(absorption, backscatter, lai, x, emission, down_top, up_bottom)

Reconstruct interior `(; down, up)` diffuse fluxes at local depth `x ∈ [0, lai]`
inside an emitting layer subject to boundary fluxes `down_top` and `up_bottom`.

# Examples
```jldoctest
julia> em = UniformEmission(0.25, 0.25);

julia> el = emitting_layer(0.12, 0.08, 1.4, em);

julia> sol = solve_emission((el,), 0.15);

julia> mid = emission_interior(0.12, 0.08, 1.4, 0.7, em, sol.down[1], sol.up[2]);

julia> mid.down > 0 && mid.up > 0
true
```
"""
@inline function emission_interior(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    x::Real,
    emission::Union{NoEmission,UniformEmission},
    down_top::Real,
    up_bottom::Real,
)
    FT = common_scalar_type(
        absorption,
        backscatter,
        lai,
        x,
        _precision_operands(emission)...,
        down_top,
        up_bottom,
    )
    abs_f = convert(FT, absorption)
    back_f = convert(FT, backscatter)
    lai_f = convert(FT, lai)
    x_f = convert(FT, x)
    dt_f = convert(FT, down_top)
    ub_f = convert(FT, up_bottom)
    upper = emitting_layer(abs_f, back_f, x_f, emission)
    lower = emitting_layer(abs_f, back_f, lai_f - x_f, emission)
    p, q = upper.passive, lower.passive
    lower_source = q.transmission * ub_f + lower.upward
    down =
        (p.transmission * dt_f + p.reflection * lower_source + upper.downward) /
        (one(FT) - p.reflection * q.reflection)
    up = q.reflection * down + lower_source
    return (; down, up)
end

@inline function emission_interior(
    absorption::Real,
    backscatter::Real,
    lai::Real,
    x::Real,
    source::QuadratureEmission,
    down_top::Real,
    up_bottom::Real,
)
    FT = common_scalar_type(absorption, backscatter, lai, x, source, down_top, up_bottom)
    abs_f = convert(FT, absorption)
    back_f = convert(FT, backscatter)
    lai_f = convert(FT, lai)
    x_f = convert(FT, x)
    dt_f = convert(FT, down_top)
    ub_f = convert(FT, up_bottom)
    background = emission_interior(abs_f, back_f, lai_f, x_f, NoEmission(), dt_f, ub_f)
    down, up = background.down, background.up
    @inbounds for k in eachindex(source.depths, source.weights, source.upward, source.downward)
        measure = lai_f * convert(FT, source.weights[k])
        f = point_emission_interior(
            abs_f,
            back_f,
            lai_f,
            lai_f * convert(FT, source.depths[k]),
            x_f,
            measure * convert(FT, source.upward[k]),
            measure * convert(FT, source.downward[k]),
        )
        down += f.down
        up += f.up
    end
    return (; down, up)
end
