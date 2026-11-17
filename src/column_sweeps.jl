# ============================================================================
# Adding sweeps of the passive and the emitting column over column storage
# ============================================================================
#
# One sweep per column kind, written once and shared by the vector-workspace
# solvers (`solve_column!`, `solve_emitting_column!`) and the KernelAbstractions
# batch kernels (`batch_solve_columns!`, `batch_solve_emitting_columns!`, in
# column_batch.jl). The two sweeps stay separate recursions: the emitting column
# carries no direct beam and orders its operations differently, so a passive column
# solved as an emitting one would differ in the last bits. The tuple recursions in
# two_stream.jl and emission_transport.jl evaluate the same expressions in the same
# order for one or more layers; they stay because they infer and run allocation free
# inside scalar broadcasts over GPU arrays. Exception, as in v0.1.0: for a passive
# column of ZERO layers the tuple method computes `soil * (direct + diffuse)`, this
# sweep `soil * diffuse + soil * direct`, which can differ by up to 2 ulp.

# Entries of one column, indexed 0, 1, … from the top of the column: entry `s` is
# `data[offset + s]` of a vector or `data[offset + s, b]` of column `b` of a matrix
# (`trailing = (b,)`). No copy and no allocation; isbits for a device array, so it is
# built inside GPU kernels. Indexing is unchecked unless Julia runs with
# `--check-bounds=yes`: `solve_column!` and `solve_emitting_column!` check every
# workspace and layer length, the batch drivers every matrix and vector axis, before
# a sweep starts.
struct ColumnView{A,I<:Tuple}
    data::A
    offset::Int
    trailing::I
end

@inline column_view(v::AbstractVector) = ColumnView(v, firstindex(v), ())
@inline Base.getindex(c::ColumnView, s::Int) = @inbounds c.data[c.offset + s, c.trailing...]
@inline function Base.setindex!(c::ColumnView, x, s::Int)
    @inbounds c.data[c.offset + s, c.trailing...] = x
    return c
end

# Layer `s` of a column, as a record with the field names the sweeps read, every
# scalar of type FT: from a vector of `LayerOperator`s or `EmittingLayer`s (converted
# to FT as the vector solvers always did), or from a `NamedTuple` of `ColumnView`s,
# one per field (the batch kernels, where the fields live in separate matrices).
@inline _layer_at(
    c::NamedTuple{(:reflection, :transmission, :beam_up, :beam_down, :beam_transmission)},
    s::Int,
    ::Type,
) = (;
    reflection = c.reflection[s],
    transmission = c.transmission[s],
    beam_up = c.beam_up[s],
    beam_down = c.beam_down[s],
    beam_transmission = c.beam_transmission[s],
)
@inline _layer_at(
    c::NamedTuple{(:reflection, :transmission, :upward, :downward, :generated)},
    s::Int,
    ::Type,
) = (;
    reflection = c.reflection[s],
    transmission = c.transmission[s],
    upward = c.upward[s],
    downward = c.downward[s],
    generated = c.generated[s],
)
@inline _layer_at(layers::ColumnView{<:AbstractVector{<:LayerOperator}}, s::Int, ::Type{FT}) where {FT} =
    convert(LayerOperator{FT}, layers[s])
@inline function _layer_at(
    layers::ColumnView{<:AbstractVector{<:EmittingLayer}},
    s::Int,
    ::Type{FT},
) where {FT}
    e = convert(EmittingLayer{FT}, layers[s])
    p = e.passive
    return (;
        reflection = p.reflection,
        transmission = p.transmission,
        upward = e.upward,
        downward = e.downward,
        generated = e.generated,
    )
end

#     _passive_column_sweep!(direct, down, up, absorption, layers, n, direct_top, diffuse_top, soil)
#
# Adding solution of a passive column of `n` layers under top direct and diffuse
# fluxes above Lambertian ground reflectance `soil`, written into the column views
# `direct`, `down`, `up` (`n + 1` interfaces) and `absorption` (`n` layers). Layer `s`
# is `_layer_at(layers, s, FT)` with fields `reflection`, `transmission`, `beam_up`,
# `beam_down`, `beam_transmission`. Returns the ground absorption.
#
# The backward sweep keeps the effective interface reflectance in `down` and the
# beam-induced upward source in `up`; the forward sweep overwrites them with fluxes.
@inline function _passive_column_sweep!(
    direct,
    down,
    up,
    absorption,
    layers,
    n::Int,
    direct_top,
    diffuse_top,
    soil,
)
    FT = typeof(direct_top)
    direct[0] = direct_top
    for s in 0:(n - 1)
        op = _layer_at(layers, s, FT)
        direct[s + 1] = direct[s] * op.beam_transmission
    end
    down[n] = soil
    up[n] = soil * direct[n]
    for s in (n - 1):-1:0
        p = _layer_at(layers, s, FT)
        r_next = down[s + 1]
        s_next = up[s + 1]
        b_curr = direct[s]
        den = inv(one(FT) - p.reflection * r_next)
        down[s] = p.reflection + p.transmission^2 * r_next * den
        up[s] =
            p.beam_up * b_curr +
            p.transmission * den * (s_next + r_next * p.beam_down * b_curr)
    end
    r1 = down[0]
    s1 = up[0]
    down[0] = diffuse_top
    up[0] = r1 * diffuse_top + s1
    for s in 0:(n - 1)
        p = _layer_at(layers, s, FT)
        d_curr = down[s]
        u_curr = up[s]
        b_curr = direct[s]
        b_next = direct[s + 1]
        r_next = down[s + 1]
        s_next = up[s + 1]
        val =
            (p.transmission * d_curr + p.reflection * s_next + p.beam_down * b_curr) /
            (one(FT) - p.reflection * r_next)
        u_next = r_next * val + s_next
        down[s + 1] = val
        up[s + 1] = u_next
        absorption[s] = d_curr + b_curr + u_next - val - b_next - u_curr
    end
    return (one(FT) - soil) * (down[n] + direct[n])
end

#     _emitting_column_sweep!(down, up, net_layer, absorption, layers, n, down_top, soil_reflect, soil_emit)
#
# Adding solution of an emitting (cavity) column of `n` layers under top downward
# flux `down_top` above ground reflectance `soil_reflect` and ground emission
# `soil_emit`, written into the column views `down`, `up` (`n + 1` interfaces) and
# `net_layer`, `absorption` (`n` layers). Layer `s` is `_layer_at(layers, s, FT)` with
# fields `reflection`, `transmission`, `upward`, `downward`, `generated`. Returns
# `(; net_soil, ground_absorbed, generated)`.
@inline function _emitting_column_sweep!(
    down,
    up,
    net_layer,
    absorption,
    layers,
    n::Int,
    down_top,
    soil_reflect,
    soil_emit,
)
    FT = typeof(down_top)
    down[n] = soil_reflect
    up[n] = soil_emit
    for s in (n - 1):-1:0
        e = _layer_at(layers, s, FT)
        r_next = down[s + 1]
        s_next = up[s + 1]
        inv_den = inv(one(FT) - e.reflection * r_next)
        down[s] = e.reflection + e.transmission^2 * r_next * inv_den
        up[s] = e.upward + e.transmission * (s_next + r_next * e.downward) * inv_den
    end
    r1 = down[0]
    s1 = up[0]
    down[0] = down_top
    up[0] = r1 * down_top + s1
    generated = zero(FT)
    for s in 0:(n - 1)
        e = _layer_at(layers, s, FT)
        d_curr = down[s]
        u_curr = up[s]
        r_next = down[s + 1]
        s_next = up[s + 1]
        val =
            (e.transmission * d_curr + e.reflection * s_next + e.downward) /
            (one(FT) - e.reflection * r_next)
        u_next = r_next * val + s_next
        down[s + 1] = val
        up[s + 1] = u_next
        net_l = (d_curr - u_curr) - (val - u_next)
        net_layer[s] = net_l
        absorption[s] = e.generated + net_l
        generated += e.generated
    end
    net_soil = down[n] - up[n]
    ground_absorbed = (one(FT) - soil_reflect) * down[n]
    return (; net_soil, ground_absorbed, generated)
end
