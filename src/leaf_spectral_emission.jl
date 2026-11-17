"""
    FullEmissionSupport()

Excitation-independent emission support rule (`emission_allowed(...) === true`)
permitting thermally broadened anti-Stokes emission across the entire tabulated
fluorophore spectrum.

# Examples
```jldoctest
julia> using CanopyOptics

julia> emission_allowed(FullEmissionSupport(), 680.0, 690.0)
true
```
"""
struct FullEmissionSupport <: AbstractEmissionSupport end

# A subtype of the CanopyOptics support type, so not a RadiationRecord: singleton
# equality and hashing are Base's; it broadcasts as a scalar.
broadcastable(s::FullEmissionSupport) = Ref(s)

@inline CanopyOptics.emission_allowed(::FullEmissionSupport, ::FT, ::FT) where {FT<:AbstractFloat} =
    true

"""
    AbstractLeafEmission

Supertype for leaf spectral fluorescence escape models
([`NoLeafEmission`](@ref), [`LeafEmissionMatrix`](@ref), and
[`DualLeafEmissionMatrix`](@ref)).
"""
abstract type AbstractLeafEmission <: RadiationRecord end

"""
    NoLeafEmission()

Disabled leaf spectral emission closure producing exact zeros without requiring
precomputed optical escape matrices.

# Examples
```jldoctest
julia> NoLeafEmission()
NoLeafEmission()
```
"""
struct NoLeafEmission <: AbstractLeafEmission end

"""
    LeafEmissionMatrix(front, back, reabsorbed)
    LeafEmissionMatrix(front, back, reabsorbed, out_of_window)

Conditional photon-fate tables after internal fluorescence production.

Rows index emission bins in the active transport window and columns index
excitation channels. Each column sums across `front`, `back`, `reabsorbed`, and
the `out_of_window` long-wavelength / short-wavelength tail fraction to unity:

```math
\\sum_m (\\text{front}_{m,e} + \\text{back}_{m,e} + \\text{reabsorbed}_{m,e}) + \\text{out\\_of\\_window}_e = 1.
```

The three-argument constructor initializes `out_of_window` to zeros sharing
`axes(front, 2)`.

# Examples
```jldoctest
julia> m = LeafEmissionMatrix([0.35;;], [0.25;;], [0.30;;], [0.10]);

julia> validate_leaf_emission(m; atol = 1e-12) === nothing
true
```
"""
struct LeafEmissionMatrix{
    FT<:Real,
    A<:AbstractMatrix{FT},
    V<:AbstractVector{FT},
} <: AbstractLeafEmission
    front::A
    back::A
    reabsorbed::A
    out_of_window::V

    function LeafEmissionMatrix(
        front::A,
        back::A,
        reabsorbed::A,
        out_of_window::V,
    ) where {FT<:Real,A<:AbstractMatrix{FT},V<:AbstractVector{FT}}
        same_precision(front)
        axes(front) == axes(back) == axes(reabsorbed) && !isempty(front) ||
            throw(DimensionMismatch("Photon-fate matrices must share nonempty axes"))
        axes(out_of_window, 1) == axes(front, 2) ||
            throw(DimensionMismatch("out_of_window must share excitation axis axes(front, 2)"))
        return new{FT,A,V}(front, back, reabsorbed, out_of_window)
    end
end

function LeafEmissionMatrix(
    front::A,
    back::A,
    reabsorbed::A,
) where {FT<:Real,A<:AbstractMatrix{FT}}
    same_precision(front)
    axes(front) == axes(back) == axes(reabsorbed) && !isempty(front) ||
        throw(DimensionMismatch("Photon-fate matrices must share nonempty axes"))
    oow = fill!(similar(front, FT, (axes(front, 2),)), zero(FT))
    return LeafEmissionMatrix(front, back, reabsorbed, oow)
end

Base.eltype(::Type{<:LeafEmissionMatrix{FT}}) where {FT} = FT

"""
    DualLeafEmissionMatrix(psii, psi)

Pair Photosystem II (`psii`) and Photosystem I (`psi`) leaf spectral escape
models (`LeafEmissionMatrix` or `NoLeafEmission`) on a shared emission-bin axis.

# Examples
```jldoctest
julia> m2 = LeafEmissionMatrix([0.4;;], [0.3;;], [0.3;;]);

julia> m1 = LeafEmissionMatrix([0.3;;], [0.25;;], [0.25;;], [0.20]);

julia> dm = DualLeafEmissionMatrix(m2, m1);

julia> validate_leaf_emission(dm; atol = 1e-12) === nothing
true
```
"""
struct DualLeafEmissionMatrix{P2<:AbstractLeafEmission,P1<:AbstractLeafEmission} <:
       AbstractLeafEmission
    psii::P2
    psi::P1

    function DualLeafEmissionMatrix(
        psii::P2,
        psi::P1,
    ) where {P2<:AbstractLeafEmission,P1<:AbstractLeafEmission}
        _check_photosystem_pair(psii, psi)
        return new{P2,P1}(psii, psi)
    end
end

# Two active photosystem matrices must share the scalar type and the emission-bin
# axis; a disabled photosystem (NoLeafEmission) pairs with anything.
@inline _check_photosystem_pair(::AbstractLeafEmission, ::AbstractLeafEmission) = nothing
function _check_photosystem_pair(psii::LeafEmissionMatrix, psi::LeafEmissionMatrix)
    eltype(psii) === eltype(psi) ||
        throw(ArgumentError("PSII and PSI LeafEmissionMatrix models must share scalar type"))
    axes(psii.front, 1) == axes(psi.front, 1) ||
        throw(DimensionMismatch("PSII and PSI LeafEmissionMatrix models must share emission-bin axis"))
    return nothing
end

"""
    validate_leaf_emission(model; atol)

Validate that all conditional photon-fate entries lie in `[0, 1]` and every
excitation column sums across `front`, `back`, `reabsorbed`, and `out_of_window`
to unity within `atol`, without clipping or rescaling.
"""
function validate_leaf_emission(
    model::LeafEmissionMatrix{FT};
    atol::Real = sqrt(eps(primal_type(FT))),
) where {FT<:Real}
    P = primal_type(FT)
    atol_p = primal(convert(FT, atol))
    all_finite(atol) && zero(P) <= atol_p < one(P) ||
        throw(ArgumentError("Invalid photon-budget tolerance"))
    nrow, ncol = size(model.front)
    fh = reshape(_host_linear(model.front), nrow, ncol)
    bh = reshape(_host_linear(model.back), nrow, ncol)
    rh = reshape(_host_linear(model.reabsorbed), nrow, ncol)
    oh = _host_linear(model.out_of_window)
    for arr in (fh, bh, rh)
        for x in arr
            all_finite(x) && zero(P) <= primal(x) <= one(P) ||
                throw(ArgumentError("Photon-fate probabilities must be finite and lie in [0, 1]"))
        end
    end
    for x in oh
        all_finite(x) && zero(P) <= primal(x) <= one(P) ||
            throw(ArgumentError("out_of_window probabilities must be finite and lie in [0, 1]"))
    end
    for c in 1:ncol
        s = primal(oh[c])
        for r in 1:nrow
            s += primal(fh[r, c]) + primal(bh[r, c]) + primal(rh[r, c])
        end
        abs(s - one(P)) <= atol_p ||
            throw(ArgumentError("Each internal-photon column must account for all escaped, reabsorbed, and out-of-window photons"))
    end
    return nothing
end

validate_leaf_emission(::NoLeafEmission; atol::Real = 0.0) = nothing

function validate_leaf_emission(dual::DualLeafEmissionMatrix; atol::Real = 1e-6)
    validate_leaf_emission(dual.psii; atol)
    validate_leaf_emission(dual.psi; atol)
    return nothing
end

function _check_disjoint_escape_outputs(outputs, inputs)
    for (i, a) in enumerate(outputs)
        a === nothing && continue
        for b in inputs
            b === nothing && continue
            _arrays_alias(a, b) && throw(ArgumentError("Escape output aliases input storage"))
        end
        for j in 1:(i - 1)
            prev = outputs[j]
            prev === nothing && continue
            _arrays_alias(a, prev) && throw(ArgumentError("Escape outputs alias each other"))
        end
    end
    return nothing
end

function _accumulate_leaf_escape!(
    front::AbstractMatrix{FT},
    back::AbstractMatrix{FT},
    reabsorbed::AbstractMatrix{FT},
    out_of_window::Union{Nothing,AbstractVector{FT}},
    model::LeafEmissionMatrix{FT},
    internal::AbstractMatrix{FT},
    accumulate::Bool,
) where {FT<:Real}
    axes(front) == axes(back) == axes(reabsorbed) ==
        (axes(model.front, 1), axes(internal, 2)) &&
        axes(internal, 1) == axes(model.front, 2) ||
        throw(DimensionMismatch("Spectral photon bins and population arrays must share aligned axes"))
    if out_of_window !== nothing
        axes(out_of_window) == (axes(internal, 2),) ||
            throw(DimensionMismatch("out_of_window vector must share population axis axes(internal, 2)"))
    end
    _check_disjoint_escape_outputs(
        (front, back, reabsorbed, out_of_window),
        (model.front, model.back, model.reabsorbed, model.out_of_window, internal),
    )
    if isempty(front)
        if out_of_window !== nothing && !accumulate
            fill!(out_of_window, zero(FT))
        end
        return nothing
    end
    has_off =
        Base.has_offset_axes(
            front,
            back,
            reabsorbed,
            model.front,
            model.back,
            model.reabsorbed,
            internal,
        ) || (out_of_window !== nothing && Base.has_offset_axes(out_of_window))
    if !has_off
        beta = accumulate ? one(FT) : zero(FT)
        mul!(front, model.front, internal, one(FT), beta)
        mul!(back, model.back, internal, one(FT), beta)
        mul!(reabsorbed, model.reabsorbed, internal, one(FT), beta)
        if out_of_window !== nothing
            mul!(out_of_window, transpose(internal), model.out_of_window, one(FT), beta)
        end
    else
        for p in axes(internal, 2)
            for m in axes(model.front, 1)
                sf = accumulate ? front[m, p] : zero(FT)
                sb = accumulate ? back[m, p] : zero(FT)
                sr = accumulate ? reabsorbed[m, p] : zero(FT)
                @inbounds for e in axes(model.front, 2)
                    q = internal[e, p]
                    sf += model.front[m, e] * q
                    sb += model.back[m, e] * q
                    sr += model.reabsorbed[m, e] * q
                end
                front[m, p] = sf
                back[m, p] = sb
                reabsorbed[m, p] = sr
            end
            if out_of_window !== nothing
                so = accumulate ? out_of_window[p] : zero(FT)
                @inbounds for e in axes(model.front, 2)
                    so += model.out_of_window[e] * internal[e, p]
                end
                out_of_window[p] = so
            end
        end
    end
    return nothing
end

"""
    escape_leaf_emission!(front, back, reabsorbed, model, internal; out_of_window = nothing)

Project internal fluorescence photon rates `internal` (`excitation × population`)
through `model` into caller-owned `(emission × population)` matrices `front`,
`back`, and `reabsorbed` (plus optional per-population `out_of_window`).

Uses BLAS / CuBLAS `mul!` on 1-based arrays and an axes-preserving loop when
offset axes (`OffsetArray`) are supplied.
"""
function escape_leaf_emission!(
    front::AbstractMatrix{FT},
    back::AbstractMatrix{FT},
    reabsorbed::AbstractMatrix{FT},
    model::LeafEmissionMatrix{FT},
    internal::AbstractMatrix{FT};
    out_of_window::Union{Nothing,AbstractVector{FT}} = nothing,
) where {FT<:Real}
    same_precision(front)
    _accumulate_leaf_escape!(front, back, reabsorbed, out_of_window, model, internal, false)
    return out_of_window === nothing ?
           (; front, back, reabsorbed) :
           (; front, back, reabsorbed, out_of_window)
end

function escape_leaf_emission!(
    front::AbstractMatrix{FT},
    back::AbstractMatrix{FT},
    reabsorbed::AbstractMatrix{FT},
    ::NoLeafEmission,
    internal = nothing;
    out_of_window::Union{Nothing,AbstractVector{FT}} = nothing,
) where {FT<:Real}
    same_precision(front)
    axes(front) == axes(back) == axes(reabsorbed) ||
        throw(DimensionMismatch("Output spectra must share axes"))
    if out_of_window !== nothing
        axes(out_of_window) == (axes(front, 2),) ||
            throw(DimensionMismatch("out_of_window vector must match population axis axes(front, 2)"))
    end
    _check_disjoint_escape_outputs((front, back, reabsorbed, out_of_window), ())
    fill!(front, zero(FT))
    fill!(back, zero(FT))
    fill!(reabsorbed, zero(FT))
    out_of_window !== nothing && fill!(out_of_window, zero(FT))
    return out_of_window === nothing ?
           (; front, back, reabsorbed) :
           (; front, back, reabsorbed, out_of_window)
end

"""
    escape_dual_leaf_emission!(front, back, reabsorbed, dual::DualLeafEmissionMatrix, internal_psii, internal_psi; out_of_window = nothing)

Superpose PSII (`internal_psii`) and PSI (`internal_psi`) internal photon rates
into shared leaf-level `front`, `back`, and `reabsorbed` spectra (plus optional
`out_of_window`) using zero-scratch accumulation.
"""
function escape_dual_leaf_emission!(
    front::AbstractMatrix{FT},
    back::AbstractMatrix{FT},
    reabsorbed::AbstractMatrix{FT},
    dual::DualLeafEmissionMatrix,
    internal_psii = nothing,
    internal_psi = nothing;
    out_of_window::Union{Nothing,AbstractVector{FT}} = nothing,
) where {FT<:Real}
    same_precision(front)
    outputs = (front, back, reabsorbed, out_of_window)
    _escape_photosystems!(outputs, dual.psii, dual.psi, internal_psii, internal_psi)
    return out_of_window === nothing ?
           (; front, back, reabsorbed) :
           (; front, back, reabsorbed, out_of_window)
end

# Escape of both photosystems into shared outputs: the first active photosystem
# overwrites, the second accumulates; with none active the outputs are zero.
const _NEED_PSII = "internal_psii matrix required for active PSII LeafEmissionMatrix"
const _NEED_PSI = "internal_psi matrix required for active PSI LeafEmissionMatrix"

@inline _required_internal(::Type{FT}, internal::AbstractMatrix{FT}, _) where {FT} = internal
@inline _required_internal(::Type, _, message) = throw(ArgumentError(message))

@inline function _escape_one!(outputs, model, internal, accumulate)
    front, back, reabsorbed, out_of_window = outputs
    _accumulate_leaf_escape!(front, back, reabsorbed, out_of_window, model, internal, accumulate)
    return nothing
end

function _escape_photosystems!(outputs, psii::LeafEmissionMatrix, psi::LeafEmissionMatrix, i2, i1)
    FT = eltype(first(outputs))
    _escape_one!(outputs, psii, _required_internal(FT, i2, _NEED_PSII), false)
    _escape_one!(outputs, psi, _required_internal(FT, i1, _NEED_PSI), true)
    return nothing
end
function _escape_photosystems!(outputs, psii::LeafEmissionMatrix, ::AbstractLeafEmission, i2, _)
    FT = eltype(first(outputs))
    _escape_one!(outputs, psii, _required_internal(FT, i2, _NEED_PSII), false)
    return nothing
end
function _escape_photosystems!(outputs, ::AbstractLeafEmission, psi::LeafEmissionMatrix, _, i1)
    FT = eltype(first(outputs))
    _escape_one!(outputs, psi, _required_internal(FT, i1, _NEED_PSI), false)
    return nothing
end
function _escape_photosystems!(outputs, ::AbstractLeafEmission, ::AbstractLeafEmission, _, _)
    front, back, reabsorbed, out_of_window = outputs
    escape_leaf_emission!(front, back, reabsorbed, NoLeafEmission(); out_of_window)
    return nothing
end

function _prepare_conditional_window_columns(
    spectrum::TabulatedPhotonSpectrum{FT},
    support::AbstractEmissionSupport,
    excitation_nm::AbstractVector{FT},
    window_indices,
    out_of_window_fraction,
) where {FT<:AbstractFloat}
    isempty(excitation_nm) &&
        throw(ArgumentError("excitation_nm must be nonempty"))
    win = collect(window_indices)
    !isempty(win) &&
        all(i -> i isa Integer && i in eachindex(spectrum.wavelength_nm), win) &&
        issorted(win) &&
        allunique(win) ||
        throw(ArgumentError("window_indices must be nonempty, strictly increasing, unique, and in bounds"))
    n_spec = length(spectrum.wavelength_nm)
    in_win = falses(n_spec)
    for idx in win
        in_win[idx] = true
    end
    ne = length(excitation_nm)
    prob_in = Matrix{FT}(undef, length(win), ne)
    oow = Vector{FT}(undef, ne)
    e0 = firstindex(excitation_nm)
    if any(!, in_win)
        out_of_window_fraction === nothing ||
            throw(ArgumentError("Cannot combine explicit out_of_window_fraction with a proper window_indices subset"))
        for c in 1:ne
            lam_ex = excitation_nm[e0 + c - 1]
            full_prob = emission_probabilities(spectrum, support, lam_ex)
            for (r, idx) in enumerate(win)
                prob_in[r, c] = full_prob[idx]
            end
            tail_sum = zero(FT)
            for i in eachindex(full_prob)
                if !in_win[i]
                    tail_sum += full_prob[i]
                end
            end
            oow[c] = tail_sum
        end
    else
        first_pos_idx = findfirst(>(zero(FT)), spectrum.relative_photon_mass)
        first_pos_nm = spectrum.wavelength_nm[first_pos_idx]
        f_vec = _tail_fractions(out_of_window_fraction, FT, ne, support, excitation_nm, first_pos_nm)
        for c in 1:ne
            lam_ex = excitation_nm[e0 + c - 1]
            full_prob = emission_probabilities(spectrum, support, lam_ex)
            scale_in = one(FT) - f_vec[c]
            for (r, idx) in enumerate(win)
                prob_in[r, c] = scale_in * full_prob[idx]
            end
            oow[c] = f_vec[c]
        end
    end
    return prob_in, oow
end

# Excitation-dependent out-of-window fraction when window_indices covers the whole
# table: none given (zero), one fraction for every excitation, or one per excitation.
_tail_fractions(::Nothing, ::Type{FT}, ne, _, _, _) where {FT} = zeros(FT, ne)
function _tail_fractions(f::Real, ::Type{FT}, ne, support, excitation_nm, first_pos_nm) where {FT}
    f_s = convert(FT, f)
    isfinite(f_s) && zero(FT) <= f_s < one(FT) ||
        throw(ArgumentError("out_of_window_fraction must be finite and lie in [0, 1)"))
    _check_scalar_tail(support, f_s, excitation_nm, first_pos_nm)
    return fill(f_s, ne)
end
function _tail_fractions(f::AbstractVector{FT}, ::Type{FT}, ne, _, _, _) where {FT}
    length(f) == ne ||
        throw(DimensionMismatch("out_of_window_fraction vector must match length(excitation_nm)"))
    for x in f
        isfinite(x) && zero(FT) <= x < one(FT) ||
            throw(ArgumentError("out_of_window_fraction entries must be finite and lie in [0, 1)"))
    end
    return collect(f)
end
_tail_fractions(_, ::Type, _, _, _, _) =
    throw(ArgumentError("Unsupported out_of_window_fraction specification"))

# A nonzero scalar tail fraction is only consistent with strict Stokes emission when
# every excitation lies below the emission spectrum.
_check_scalar_tail(::AbstractEmissionSupport, _, _, _) = nothing
function _check_scalar_tail(::StrictStokes, f_s, excitation_nm, first_pos_nm)
    !iszero(f_s) &&
        any(>=(first_pos_nm), excitation_nm) &&
        throw(ArgumentError("Nonzero scalar out_of_window_fraction is inconsistent with StrictStokes excitation inside the emission spectrum; pass full TabulatedPhotonSpectrum with window_indices or per-excitation vector"))
    return nothing
end

"""
    build_leaf_emission_matrices(spectrum, support, excitation_nm, emission_escapes, births; window_indices = eachindex(spectrum.wavelength_nm), out_of_window_fraction = nothing, atol = ...)
    build_leaf_emission_matrices(spectrum, support, excitation_nm, emission_escapes, front_births, back_births; window_indices = eachindex(spectrum.wavelength_nm), out_of_window_fraction = nothing, atol = ...)

Construct a validated [`LeafEmissionMatrix`](@ref) from a fluorophore
`TabulatedPhotonSpectrum`, an `AbstractEmissionSupport` (`StrictStokes()` or
[`FullEmissionSupport`](@ref)), `excitation_nm`, per-emission-bin leaf escape
operators `emission_escapes` (`RegularPlateEscape` or `DorsiventralPlateEscape`),
and birth-depth profiles (`births` for `n_excite` columns, or `front_births` and
`back_births` for `2 * n_excite` anatomical-face columns).

When `window_indices` selects a proper subset of `spectrum` (such as `640–850 nm`
from a full `600–1200 nm` fluorophore table), `emission_probabilities` is
conditioned on the full table prior to window slicing so in-window probabilities
are never falsely renormalized and `out_of_window` records the exact
excitation-dependent tail fraction.
"""
function build_leaf_emission_matrices(
    spectrum::TabulatedPhotonSpectrum{FT},
    support::AbstractEmissionSupport,
    excitation_nm::AbstractVector{FT},
    emission_escapes::AbstractVector,
    births::AbstractVector{<:AbstractLeafBirthProfile};
    window_indices = eachindex(spectrum.wavelength_nm),
    out_of_window_fraction = nothing,
    atol::FT = FT(256) * eps(FT),
) where {FT<:AbstractFloat}
    prob_in, oow = _prepare_conditional_window_columns(
        spectrum,
        support,
        excitation_nm,
        window_indices,
        out_of_window_fraction,
    )
    n_emit, ne = size(prob_in)
    length(emission_escapes) == n_emit ||
        throw(DimensionMismatch("emission_escapes length must match length(window_indices)"))
    length(births) == ne ||
        throw(DimensionMismatch("births length must match length(excitation_nm)"))
    front = Matrix{FT}(undef, n_emit, ne)
    back = Matrix{FT}(undef, n_emit, ne)
    reabs = Matrix{FT}(undef, n_emit, ne)
    esc0 = firstindex(emission_escapes)
    b0 = firstindex(births)
    for c in 1:ne
        birth = births[b0 + c - 1]
        for r in 1:n_emit
            p = prob_in[r, c]
            fates = leaf_emission_fates(emission_escapes[esc0 + r - 1], birth)
            front[r, c] = p * fates.front
            back[r, c] = p * fates.back
            reabs[r, c] = p * fates.reabsorbed
        end
    end
    model = LeafEmissionMatrix(front, back, reabs, oow)
    validate_leaf_emission(model; atol)
    return model
end

function build_leaf_emission_matrices(
    spectrum::TabulatedPhotonSpectrum{FT},
    support::AbstractEmissionSupport,
    excitation_nm::AbstractVector{FT},
    emission_escapes::AbstractVector,
    front_births::AbstractVector{<:AbstractLeafBirthProfile},
    back_births::AbstractVector{<:AbstractLeafBirthProfile};
    window_indices = eachindex(spectrum.wavelength_nm),
    out_of_window_fraction = nothing,
    atol::FT = FT(256) * eps(FT),
) where {FT<:AbstractFloat}
    prob_in, oow_single = _prepare_conditional_window_columns(
        spectrum,
        support,
        excitation_nm,
        window_indices,
        out_of_window_fraction,
    )
    n_emit, ne = size(prob_in)
    length(emission_escapes) == n_emit ||
        throw(DimensionMismatch("emission_escapes length must match length(window_indices)"))
    length(front_births) == ne && length(back_births) == ne ||
        throw(DimensionMismatch("front_births and back_births must match length(excitation_nm)"))
    front = Matrix{FT}(undef, n_emit, 2 * ne)
    back = Matrix{FT}(undef, n_emit, 2 * ne)
    reabs = Matrix{FT}(undef, n_emit, 2 * ne)
    oow = Vector{FT}(undef, 2 * ne)
    esc0 = firstindex(emission_escapes)
    fb0 = firstindex(front_births)
    bb0 = firstindex(back_births)
    for c in 1:ne
        bf = front_births[fb0 + c - 1]
        bb = back_births[bb0 + c - 1]
        oow[c] = oow_single[c]
        oow[ne + c] = oow_single[c]
        for r in 1:n_emit
            p = prob_in[r, c]
            esc = emission_escapes[esc0 + r - 1]
            ff = leaf_emission_fates(esc, bf)
            fb = leaf_emission_fates(esc, bb)
            front[r, c] = p * ff.front
            back[r, c] = p * ff.back
            reabs[r, c] = p * ff.reabsorbed
            front[r, ne + c] = p * fb.front
            back[r, ne + c] = p * fb.back
            reabs[r, ne + c] = p * fb.reabsorbed
        end
    end
    model = LeafEmissionMatrix(front, back, reabs, oow)
    validate_leaf_emission(model; atol)
    return model
end

# ============================================================================
# Row-wise spectral conversions: output[i, j] = f(input[i, j], row[i])
# ============================================================================
#
# A plain loop on the KernelAbstractions CPU backend (no kernel launch for these
# memory-bound maps), one kernel elsewhere. `f` is `*` (bins from densities), `/`
# (densities from bins) or a `_PerWavelengthEnergy` (energy from photons).

struct _PerWavelengthEnergy{FT}
    coefficient::FT
end
@inline (e::_PerWavelengthEnergy)(photons, wavelength_nm) =
    photons * (e.coefficient / wavelength_nm)

@kernel function _row_map_kernel!(f, output, input, row, r0::Int, c0::Int, s0::Int)
    i, j = @index(Global, NTuple)
    @inbounds output[r0 + i - 1, c0 + j - 1] = f(input[r0 + i - 1, c0 + j - 1], row[s0 + i - 1])
end

function _map_rows!(::KernelAbstractions.CPU, f, output, input, row)
    @inbounds for j in axes(output, 2), i in axes(output, 1)
        output[i, j] = f(input[i, j], row[i])
    end
    return output
end

function _map_rows!(backend, f, output, input, row)
    k! = _row_map_kernel!(backend)
    k!(
        f,
        output,
        input,
        row,
        firstindex(output, 1),
        firstindex(output, 2),
        firstindex(row, 1);
        ndrange = size(output),
    )
    KernelAbstractions.synchronize(backend)
    return output
end

function _check_spectral_row_axes(output, input, measure)
    axes(output) == axes(input) && axes(input, 1) == axes(measure, 1) ||
        throw(DimensionMismatch("Spectral output, input, and row-measure axes must align"))
    return nothing
end

"""
    photon_bins!(output, density, widths_nm)

Convert per-nm photon `density` into integrated photon bins in `output` by
multiplying row `r` by `widths_nm[r]` across CPU, `OffsetArray`, and GPU arrays.
"""
function photon_bins!(
    output::AbstractMatrix{FT},
    density::AbstractMatrix{FT},
    widths_nm::AbstractVector{FT},
) where {FT<:Real}
    same_precision(output)
    _check_spectral_row_axes(output, density, widths_nm)
    isempty(output) && return output
    _map_rows!(KernelAbstractions.get_backend(output), *, output, density, widths_nm)
    return output
end

"""
    photon_density!(output, bins, widths_nm)

Convert integrated photon `bins` into per-nm photon density in `output` by
dividing row `r` by `widths_nm[r]` across CPU, `OffsetArray`, and GPU arrays.
"""
function photon_density!(
    output::AbstractMatrix{FT},
    bins::AbstractMatrix{FT},
    widths_nm::AbstractVector{FT},
) where {FT<:Real}
    same_precision(output)
    _check_spectral_row_axes(output, bins, widths_nm)
    isempty(output) && return output
    _map_rows!(KernelAbstractions.get_backend(output), /, output, bins, widths_nm)
    return output
end

"""
    photon_energy!(output, photons, wavelengths_nm, coefficient)

Convert photon fluxes `photons` to energy fluxes in `output` using each row's
representative wavelength `wavelengths_nm[r]` and caller-supplied conversion
`coefficient` (`N_A h c × (μmol/mol) / (nm/m)`).
"""
function photon_energy!(
    output::AbstractMatrix{FT},
    photons::AbstractMatrix{FT},
    wavelengths_nm::AbstractVector{FT},
    coefficient::FT,
) where {FT<:Real}
    same_precision(output)
    _check_spectral_row_axes(output, photons, wavelengths_nm)
    isempty(output) && return output
    _map_rows!(KernelAbstractions.get_backend(output), _PerWavelengthEnergy(coefficient), output, photons, wavelengths_nm)
    return output
end
