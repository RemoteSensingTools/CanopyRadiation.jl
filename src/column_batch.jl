# ============================================================================
# Batch column solvers: one KernelAbstractions work item per column
# ============================================================================
#
# Each work item solves column `b` with the sweep shared with the vector solvers
# (column_sweeps.jl), reading and writing column `b` of every matrix through a
# `ColumnView`. Interface rows start at `if0`, layer rows at `ia0` (outputs) and
# `ir0` (inputs), columns at `b0`: the first indices of the arrays, so `OffsetArray`
# storage works. The same kernel runs on the CPU backend and on GPUs.

@inline _batch_column(A::AbstractMatrix, row0::Int, b::Int) = ColumnView(A, row0, (b,))

@kernel function _batch_solve_columns_kernel!(
    direct_out,
    down_out,
    up_out,
    absorption_out,
    ground_out,
    reflection,
    transmission,
    beam_up,
    beam_down,
    beam_trans,
    direct_top,
    diffuse_top,
    soil,
    if0::Int,
    ia0::Int,
    ir0::Int,
    b0::Int,
    n_layers::Int,
)
    b_rel = @index(Global, Linear)
    b = b0 + b_rel - 1
    layers = (;
        reflection = _batch_column(reflection, ir0, b),
        transmission = _batch_column(transmission, ir0, b),
        beam_up = _batch_column(beam_up, ir0, b),
        beam_down = _batch_column(beam_down, ir0, b),
        beam_transmission = _batch_column(beam_trans, ir0, b),
    )
    @inbounds ground_out[b] = _passive_column_sweep!(
        _batch_column(direct_out, if0, b),
        _batch_column(down_out, if0, b),
        _batch_column(up_out, if0, b),
        _batch_column(absorption_out, ia0, b),
        layers,
        n_layers,
        direct_top[b],
        diffuse_top[b],
        soil[b],
    )
end

"""
    batch_solve_columns!(direct_out, down_out, up_out, absorption_out, ground_out, reflection, transmission, beam_up, beam_down, beam_trans, direct_top, diffuse_top, soil)

Execute zero-scratch backend-portable (`KernelAbstractions`) passive column
solves across `n_batch` independent columns/wavelengths on CPU, `OffsetArray`,
or GPU (`CuArray`).

Layer inputs are `(n_layers, n_batch)` matrices of the [`LayerOperator`](@ref)
fields; interface outputs are `(n_layers + 1, n_batch)`. Every column is solved
by the same adding sweep, in the same arithmetic order, as [`solve_column!`](@ref).
All arrays must share one floating-point element type (`ArgumentError` for a
non-floating-point type).
"""
function batch_solve_columns!(
    direct_out::AbstractMatrix{FT},
    down_out::AbstractMatrix{FT},
    up_out::AbstractMatrix{FT},
    absorption_out::AbstractMatrix{FT},
    ground_out::AbstractVector{FT},
    reflection::AbstractMatrix{FT},
    transmission::AbstractMatrix{FT},
    beam_up::AbstractMatrix{FT},
    beam_down::AbstractMatrix{FT},
    beam_trans::AbstractMatrix{FT},
    direct_top::AbstractVector{FT},
    diffuse_top::AbstractVector{FT},
    soil::AbstractVector{FT},
) where {FT<:Real}
    same_precision(down_out)
    ax_lay, ax_b = axes(reflection)
    n_layers, n_batch = size(reflection)
    axes(transmission) == axes(beam_up) == axes(beam_down) == axes(beam_trans) == (ax_lay, ax_b) &&
        axes(direct_out) == axes(down_out) == axes(up_out) &&
        size(down_out, 1) == n_layers + 1 &&
        axes(down_out, 2) == ax_b &&
        size(absorption_out, 1) == n_layers &&
        axes(absorption_out, 2) == ax_b &&
        axes(ground_out, 1) == axes(direct_top, 1) == axes(diffuse_top, 1) == axes(soil, 1) == ax_b ||
        throw(DimensionMismatch("Batch column array shapes or axes do not align"))
    _disjoint_hemispheric_buffers(
        (direct_out, down_out, up_out, absorption_out, ground_out),
        (reflection, transmission, beam_up, beam_down, beam_trans, direct_top, diffuse_top, soil),
    )
    n_batch == 0 && return nothing
    backend = KernelAbstractions.get_backend(down_out)
    k! = _batch_solve_columns_kernel!(backend)
    k!(
        direct_out,
        down_out,
        up_out,
        absorption_out,
        ground_out,
        reflection,
        transmission,
        beam_up,
        beam_down,
        beam_trans,
        direct_top,
        diffuse_top,
        soil,
        firstindex(down_out, 1),
        firstindex(absorption_out, 1),
        firstindex(reflection, 1),
        firstindex(reflection, 2),
        n_layers;
        ndrange = n_batch,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _batch_solve_emitting_columns_kernel!(
    down_out,
    up_out,
    net_layer_out,
    absorption_out,
    net_soil_out,
    ground_abs_out,
    generated_out,
    reflection,
    transmission,
    emit_up,
    emit_down,
    emit_gen,
    down_top,
    soil_reflect,
    soil_emit,
    if0::Int,
    ia0::Int,
    ir0::Int,
    b0::Int,
    n_layers::Int,
)
    b_rel = @index(Global, Linear)
    b = b0 + b_rel - 1
    layers = (;
        reflection = _batch_column(reflection, ir0, b),
        transmission = _batch_column(transmission, ir0, b),
        upward = _batch_column(emit_up, ir0, b),
        downward = _batch_column(emit_down, ir0, b),
        generated = _batch_column(emit_gen, ir0, b),
    )
    @inbounds begin
        sol = _emitting_column_sweep!(
            _batch_column(down_out, if0, b),
            _batch_column(up_out, if0, b),
            _batch_column(net_layer_out, ia0, b),
            _batch_column(absorption_out, ia0, b),
            layers,
            n_layers,
            down_top[b],
            soil_reflect[b],
            soil_emit[b],
        )
        net_soil_out[b] = sol.net_soil
        ground_abs_out[b] = sol.ground_absorbed
        generated_out[b] = sol.generated
    end
end

"""
    batch_solve_emitting_columns!(down_out, up_out, net_layer_out, absorption_out, net_soil_out, ground_abs_out, generated_out, reflection, transmission, emit_up, emit_down, emit_gen, down_top, soil_reflect, soil_emit)

Execute zero-scratch backend-portable (`KernelAbstractions`) cavity emission and
longwave column solves across `n_batch` independent columns/wavelengths on CPU,
`OffsetArray`, or GPU (`CuArray`).

Layer inputs are `(n_layers, n_batch)` matrices of the [`EmittingLayer`](@ref)
fields (passive reflection and transmission, upward and downward escape,
generated emission); interface outputs are `(n_layers + 1, n_batch)`. Every column
is solved by the same adding sweep, in the same arithmetic order, as
[`solve_emitting_column!`](@ref). All arrays must share one floating-point element
type (`ArgumentError` for a non-floating-point type).
"""
function batch_solve_emitting_columns!(
    down_out::AbstractMatrix{FT},
    up_out::AbstractMatrix{FT},
    net_layer_out::AbstractMatrix{FT},
    absorption_out::AbstractMatrix{FT},
    net_soil_out::AbstractVector{FT},
    ground_abs_out::AbstractVector{FT},
    generated_out::AbstractVector{FT},
    reflection::AbstractMatrix{FT},
    transmission::AbstractMatrix{FT},
    emit_up::AbstractMatrix{FT},
    emit_down::AbstractMatrix{FT},
    emit_gen::AbstractMatrix{FT},
    down_top::AbstractVector{FT},
    soil_reflect::AbstractVector{FT},
    soil_emit::AbstractVector{FT},
) where {FT<:Real}
    same_precision(down_out)
    ax_lay, ax_b = axes(reflection)
    n_layers, n_batch = size(reflection)
    axes(transmission) == axes(emit_up) == axes(emit_down) == axes(emit_gen) == (ax_lay, ax_b) &&
        axes(down_out) == axes(up_out) &&
        size(down_out, 1) == n_layers + 1 &&
        axes(down_out, 2) == ax_b &&
        axes(net_layer_out) == axes(absorption_out) &&
        size(absorption_out, 1) == n_layers &&
        axes(absorption_out, 2) == ax_b &&
        axes(net_soil_out, 1) == axes(ground_abs_out, 1) == axes(generated_out, 1) ==
        axes(down_top, 1) == axes(soil_reflect, 1) == axes(soil_emit, 1) == ax_b ||
        throw(DimensionMismatch("Batch emitting-column array shapes or axes do not align"))
    _disjoint_hemispheric_buffers(
        (down_out, up_out, net_layer_out, absorption_out, net_soil_out, ground_abs_out, generated_out),
        (reflection, transmission, emit_up, emit_down, emit_gen, down_top, soil_reflect, soil_emit),
    )
    n_batch == 0 && return nothing
    backend = KernelAbstractions.get_backend(down_out)
    k! = _batch_solve_emitting_columns_kernel!(backend)
    k!(
        down_out,
        up_out,
        net_layer_out,
        absorption_out,
        net_soil_out,
        ground_abs_out,
        generated_out,
        reflection,
        transmission,
        emit_up,
        emit_down,
        emit_gen,
        down_top,
        soil_reflect,
        soil_emit,
        firstindex(down_out, 1),
        firstindex(absorption_out, 1),
        firstindex(reflection, 1),
        firstindex(reflection, 2),
        n_layers;
        ndrange = n_batch,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
