"""
    AbstractSpectralMap{FT<:AbstractFloat}

Supertype for nonnegative point-response spectral interpolation operators
([`IdentitySpectralMap`](@ref) and [`LinearSpectralMap`](@ref)).
"""
abstract type AbstractSpectralMap{FT<:AbstractFloat} end

function _checked_wavelength_grid(grid::AbstractVector{FT}) where {FT<:AbstractFloat}
    isempty(grid) && throw(ArgumentError("Spectral grid cannot be empty"))
    all(x -> isfinite(x) && x > zero(FT), grid) ||
        throw(ArgumentError("Spectral grid wavelengths must be positive and finite"))
    i0 = firstindex(grid)
    all(i -> grid[i] > grid[i - 1], (i0 + 1):lastindex(grid)) ||
        throw(ArgumentError("Spectral grid wavelengths must be strictly increasing"))
    return collect(FT, grid)
end

"""
    IdentitySpectralMap(grid)

Identity spectral map on a copied, validated strictly increasing wavelength
`grid`.

# Examples
```jldoctest
julia> m = IdentitySpectralMap([400.0, 550.0, 700.0]);

julia> source_grid(m) == target_grid(m)
true
```
"""
struct IdentitySpectralMap{FT<:AbstractFloat} <: AbstractSpectralMap{FT}
    wavelengths::Vector{FT}

    IdentitySpectralMap(grid::AbstractVector{FT}) where {FT<:AbstractFloat} =
        new{FT}(_checked_wavelength_grid(grid))
end

Base.eltype(::Type{IdentitySpectralMap{FT}}) where {FT} = FT
Base.eltype(m::IdentitySpectralMap) = eltype(m.wavelengths)
Base.:(==)(a::IdentitySpectralMap, b::IdentitySpectralMap) = a.wavelengths == b.wavelengths
Base.isequal(a::IdentitySpectralMap, b::IdentitySpectralMap) =
    _isequal_array(a.wavelengths, b.wavelengths)
Base.hash(m::IdentitySpectralMap, h::UInt) =
    _hash_array(m.wavelengths, hash(:IdentitySpectralMap, h))

"""
    LinearSpectralMap(anchors, targets)

Nonnegative piecewise-linear interpolation map from `anchors` to `targets`
without extrapolation (`first(anchors) <= first(targets)` and
`last(targets) <= last(anchors)`).

# Examples
```jldoctest
julia> m = LinearSpectralMap([400.0, 700.0], [400.0, 550.0, 700.0]);

julia> source_basis_weights(m)
3×2 Matrix{Float64}:
 1.0  0.0
 0.5  0.5
 0.0  1.0
```
"""
struct LinearSpectralMap{FT<:AbstractFloat} <: AbstractSpectralMap{FT}
    anchors::Vector{FT}
    targets::Vector{FT}
    lower::Vector{Int}
    upper::Vector{Int}
    upper_weight::Vector{FT}

    function LinearSpectralMap(
        anchors::AbstractVector{FT},
        targets::AbstractVector{FT},
    ) where {FT<:AbstractFloat}
        a = _checked_wavelength_grid(anchors)
        t = _checked_wavelength_grid(targets)
        first(t) >= first(a) && last(t) <= last(a) ||
            throw(ArgumentError("Target wavelengths must lie within the anchor grid"))
        lower = Vector{Int}(undef, length(t))
        upper = Vector{Int}(undef, length(t))
        weights = Vector{FT}(undef, length(t))
        for (idx, wavelength) in enumerate(t)
            j = searchsortedlast(a, wavelength)
            exact = a[j] == wavelength
            k = exact ? j : j + 1
            w = exact ? zero(FT) : (wavelength - a[j]) / (a[k] - a[j])
            lower[idx] = j
            upper[idx] = k
            weights[idx] = w
        end
        return new{FT}(a, t, lower, upper, weights)
    end
end

Base.eltype(::Type{LinearSpectralMap{FT}}) where {FT} = FT
Base.eltype(m::LinearSpectralMap) = eltype(m.anchors)
Base.:(==)(a::LinearSpectralMap, b::LinearSpectralMap) =
    a.anchors == b.anchors &&
    a.targets == b.targets &&
    a.lower == b.lower &&
    a.upper == b.upper &&
    a.upper_weight == b.upper_weight

Base.isequal(a::LinearSpectralMap, b::LinearSpectralMap) =
    _isequal_array(a.anchors, b.anchors) &&
    _isequal_array(a.targets, b.targets) &&
    isequal(a.lower, b.lower) &&
    isequal(a.upper, b.upper) &&
    _isequal_array(a.upper_weight, b.upper_weight)

function Base.hash(m::LinearSpectralMap, h::UInt)
    h = hash(:LinearSpectralMap, h)
    h = _hash_array(m.anchors, h)
    h = _hash_array(m.targets, h)
    h = hash(m.lower, h)
    h = hash(m.upper, h)
    return _hash_array(m.upper_weight, h)
end

"""
    source_grid(map::AbstractSpectralMap)

Return the read-only source/anchor wavelength vector of `map`.
"""
source_grid(m::IdentitySpectralMap) = m.wavelengths
source_grid(m::LinearSpectralMap) = m.anchors

"""
    target_grid(map::AbstractSpectralMap)

Return the read-only target/native wavelength vector of `map`.
"""
target_grid(m::IdentitySpectralMap) = m.wavelengths
target_grid(m::LinearSpectralMap) = m.targets

function _check_spectral_map_buffers(dest, m::AbstractSpectralMap{FT}, src) where {FT}
    eltype(dest) === FT && eltype(src) === FT ||
        throw(ArgumentError("Map and response buffers must share identical floating type"))
    ndims(dest) == ndims(src) &&
        ndims(src) > 0 &&
        size(src, 1) == length(source_grid(m)) &&
        size(dest, 1) == length(target_grid(m)) &&
        axes(src)[2:end] == axes(dest)[2:end] ||
        throw(DimensionMismatch("Spectral response buffer dimensions or batch axes differ"))
    _arrays_alias(dest, src) &&
        throw(ArgumentError("Source and destination buffers must not alias"))
    return nothing
end

"""
    map_spectrum!(dest, map::AbstractSpectralMap, src)

Interpolate first-axis point responses from `src` into `dest` along `map`,
preserving all trailing batch dimensions.
"""
function map_spectrum!(dest, m::IdentitySpectralMap, src)
    _check_spectral_map_buffers(dest, m, src)
    copyto!(dest, src)
    return dest
end

function map_spectrum!(dest, m::LinearSpectralMap{FT}, src) where {FT}
    _check_spectral_map_buffers(dest, m, src)
    d0 = firstindex(dest, 1)
    s0 = firstindex(src, 1)
    for j in eachindex(m.targets)
        out = selectdim(dest, 1, d0 + j - 1)
        lo = selectdim(src, 1, s0 + m.lower[j] - 1)
        if m.lower[j] == m.upper[j]
            out .= lo
        else
            hi = selectdim(src, 1, s0 + m.upper[j] - 1)
            w = m.upper_weight[j]
            out .= (one(FT) - w) .* lo .+ w .* hi
        end
    end
    return dest
end

"""
    source_basis_weights(map::AbstractSpectralMap)

Return the `(n_target, n_source)` partition-of-unity weight matrix of `map` for
exact algebraic excitation pullback.
"""
function source_basis_weights(::AbstractSpectralMap)
    throw(ArgumentError("Spectral map has no fixed linear source-basis weights"))
end

function source_basis_weights(
    m::Union{IdentitySpectralMap{FT},LinearSpectralMap{FT}},
) where {FT}
    na = length(source_grid(m))
    weights = zeros(FT, length(target_grid(m)), na)
    return map_spectrum!(weights, m, Matrix{FT}(I, na, na))
end

"""
    prepare_emission_basis(model::LeafEmissionMatrix, excitation_spectra; atol)

Reduce `model` onto normalized conditional excitation basis columns from
nonnegative `(excitation × basis)` `excitation_spectra`, propagating `front`,
`back`, `reabsorbed`, and `out_of_window` while preserving exact column-sum
conservation.

Return `(; totals, active, model, conditional_excitation)` where `model === nothing`
when all basis columns are identically zero.
"""
function prepare_emission_basis(
    model::LeafEmissionMatrix{FT},
    excitation_spectra::AbstractMatrix{FT};
    atol::FT = FT(256) * eps(FT),
) where {FT<:AbstractFloat}
    size(excitation_spectra, 1) == size(model.front, 2) ||
        throw(DimensionMismatch("Basis and optical excitation axes differ"))
    size(excitation_spectra, 2) > 0 ||
        throw(ArgumentError("At least one original basis column is required"))
    spec_h = reshape(_host_linear(excitation_spectra), size(excitation_spectra))
    all(x -> isfinite(x) && x >= zero(FT), spec_h) ||
        throw(ArgumentError("Excitation basis spectra must be finite and nonnegative"))
    validate_leaf_emission(model; atol)
    totals = vec(sum(spec_h; dims = 1))
    all(isfinite, totals) ||
        throw(ArgumentError("Unrepresentable basis excitation integral"))
    active = findall(>(zero(FT)), totals)
    if isempty(active)
        return (;
            totals,
            active,
            model = nothing,
            conditional_excitation = zeros(FT, size(spec_h, 1), 0),
        )
    end
    conditional = Matrix{FT}(spec_h[:, active])
    for col in eachcol(conditional)
        col ./= maximum(col)
        col ./= sum(col)
    end
    fh = reshape(_host_linear(model.front), size(model.front))
    bh = reshape(_host_linear(model.back), size(model.back))
    rh = reshape(_host_linear(model.reabsorbed), size(model.reabsorbed))
    oh = _host_linear(model.out_of_window)
    prepared = LeafEmissionMatrix(
        fh * conditional,
        bh * conditional,
        rh * conditional,
        vec(transpose(oh) * conditional),
    )
    validate_leaf_emission(prepared; atol)
    return (; totals, active, model = prepared, conditional_excitation = conditional)
end

"""
    validate_basis_amplitudes(amplitudes, n_basis)

Validate that `(population × n_basis)` basis `amplitudes` are finite and
nonnegative across all original basis columns (including inactive ones).
"""
function validate_basis_amplitudes(
    amplitudes::AbstractMatrix{FT},
    n_basis::Integer,
) where {FT<:AbstractFloat}
    n_basis > 0 && size(amplitudes, 2) == n_basis ||
        throw(DimensionMismatch("Amplitude columns must match n_basis"))
    for x in _host_linear(amplitudes)
        isfinite(x) && x >= zero(FT) ||
            throw(ArgumentError("Basis amplitudes must be finite and nonnegative"))
    end
    return nothing
end

@inline function _supported_product(a::FT, b::FT) where {FT<:AbstractFloat}
    val = a * b
    isfinite(val) || throw(ArgumentError("Unrepresentable source-basis product"))
    a > zero(FT) &&
        b > zero(FT) &&
        iszero(val) &&
        throw(ArgumentError("Positive source support underflowed"))
    return val
end

"""
    prepare_anchor_basis(map, incident_bins, eta, model, internal_probability, excitation_energy; atol)

Prepare exact algebraic anchor-channel excitation and emission pullback tables
for direct/diffuse boundary `incident_bins` (`n_native × 2`), photosystem
selection fraction `eta` (`n_native`), anatomical-face `model` (`n_emit × 2n_native`),
in-window `internal_probability` (`n_emit × 2n_native`), and `excitation_energy`.

Enforces full photon conservation across in-window `internal_probability` plus
`model.out_of_window` both before and after basis reduction.
"""
function prepare_anchor_basis(
    map::AbstractSpectralMap{FT},
    incident_bins::AbstractMatrix{FT},
    eta::AbstractVector{FT},
    model::LeafEmissionMatrix{FT},
    internal_probability::AbstractMatrix{FT},
    excitation_energy::AbstractVector{FT};
    atol::FT = FT(256) * eps(FT),
) where {FT<:AbstractFloat}
    ne = length(target_grid(map))
    na = length(source_grid(map))
    isfinite(atol) && zero(FT) <= atol < one(FT) ||
        throw(ArgumentError("Invalid source probability tolerance"))
    size(incident_bins) == (ne, 2) &&
        length(eta) == length(excitation_energy) == ne ||
        throw(DimensionMismatch("Native forcing, selection, and energy axes differ"))
    size(model.front, 2) == 2 * ne &&
        size(internal_probability) == size(model.front) ||
        throw(DimensionMismatch("Native anatomical-face optical axes differ"))
    inc_h = reshape(_host_linear(incident_bins), ne, 2)
    eta_h = _host_linear(eta)
    ip_h = reshape(_host_linear(internal_probability), size(internal_probability))
    en_h = _host_linear(excitation_energy)
    fh = reshape(_host_linear(model.front), size(model.front))
    bh = reshape(_host_linear(model.back), size(model.back))
    rh = reshape(_host_linear(model.reabsorbed), size(model.reabsorbed))
    oh = _host_linear(model.out_of_window)
    for vals in (inc_h, eta_h, en_h, ip_h)
        all(x -> isfinite(x) && x >= zero(FT), vals) ||
            throw(ArgumentError("Source preparation inputs must be finite and nonnegative"))
    end
    all(<=(one(FT)), eta_h) ||
        throw(ArgumentError("Selected photosystem fraction exceeds absorbed photons"))
    all(>(zero(FT)), en_h) ||
        throw(ArgumentError("Excitation energy must be positive"))
    validate_leaf_emission(model; atol)
    all(<=(one(FT)), ip_h) &&
        all(x -> abs(x - one(FT)) <= atol, vec(sum(ip_h; dims = 1)) .+ oh) ||
        throw(ArgumentError("In-window internal probability plus out_of_window must sum to unity"))
    all(abs.(ip_h .- (fh .+ bh .+ rh)) .<= atol) ||
        throw(ArgumentError("In-window internal probability disagrees with optical fates"))
    weights = source_basis_weights(map)
    eltype(weights) === FT && size(weights) == (ne, na) ||
        throw(DimensionMismatch("Source-basis map weights differ from native/anchor axes or precision"))
    all(x -> isfinite(x) && x >= zero(FT), weights) &&
        all(x -> abs(x - one(FT)) <= atol, vec(sum(weights; dims = 2))) ||
        throw(ArgumentError("Source basis requires a nonnegative partition-of-unity point map"))
    forcing_scale = [maximum(view(inc_h, :, b)) for b in 1:2]
    forcing_scale = [iszero(x) ? one(FT) : x for x in forcing_scale]
    scaled = Matrix{FT}(undef, ne, 2)
    for b in 1:2, i in 1:ne
        val = inc_h[i, b] / forcing_scale[b]
        inc_h[i, b] > zero(FT) &&
            iszero(val) &&
            throw(ArgumentError("Native forcing dynamic range cannot preserve positive support"))
        scaled[i, b] = val
    end
    labels = [(; anchor = a, boundary = b, receiving_face = f) for f in 1:2 for b in 1:2 for a in 1:na]
    photosystem = zeros(FT, 2 * ne, length(labels))
    absorbed = zeros(FT, length(labels))
    scale = FT[forcing_scale[k.boundary] for k in labels]
    for (k, label) in enumerate(labels), i in 1:ne
        h = _supported_product(weights[i, label.anchor], scaled[i, label.boundary])
        absorbed[k] += h
        photosystem[i + (label.receiving_face - 1) * ne, k] = _supported_product(eta_h[i], h)
    end
    all(isfinite, absorbed) || throw(ArgumentError("Unrepresentable APAR basis integral"))
    prepared = prepare_emission_basis(model, photosystem; atol)
    internal = ip_h * prepared.conditional_excitation
    if prepared.model !== nothing
        all(
            x -> abs(x - one(FT)) <= atol,
            vec(sum(internal; dims = 1)) .+ prepared.model.out_of_window,
        ) || throw(ArgumentError("Prepared internal source columns plus out_of_window must conserve photons"))
    end
    energy = vcat(en_h, en_h)
    precursor = vec(transpose(energy) * prepared.conditional_excitation)
    return (;
        absorbed,
        psii = photosystem,
        scale,
        labels,
        weights,
        forcing = copy(inc_h),
        eta = copy(eta_h),
        anchors = copy(source_grid(map)),
        native = copy(target_grid(map)),
        totals = prepared.totals,
        active = prepared.active,
        model = prepared.model,
        conditional_excitation = prepared.conditional_excitation,
        internal,
        precursor,
    )
end

"""
    anchor_amplitudes(front, back, basis; populations = axes(front, 2))

Assemble `(population × channel)` basis amplitudes from unit-boundary face
responses `front` and `back` of shape `(anchor, population, boundary)` and
`basis` prepared by [`prepare_anchor_basis`](@ref).
"""
function anchor_amplitudes(
    front::AbstractArray{FT,3},
    back::AbstractArray{FT,3},
    basis;
    populations = axes(front, 2),
) where {FT<:AbstractFloat}
    eltype(basis.scale) === FT ||
        throw(ArgumentError("Basis and RT response precision differ"))
    axes(front) == axes(back) &&
        size(front, 1) == length(basis.anchors) &&
        size(front, 3) == 2 ||
        throw(DimensionMismatch("RT response anchor/population/boundary axes differ"))
    indices = collect(populations)
    all(i -> i isa Integer && i in axes(front, 2), indices) ||
        throw(BoundsError(front, indices))
    a0 = firstindex(front, 1)
    b0 = firstindex(front, 3)
    amplitudes = zeros(FT, length(indices), length(basis.labels))
    for (k, label) in enumerate(basis.labels), (row, p) in enumerate(indices)
        arr = label.receiving_face == 1 ? front : back
        resp = arr[a0 + label.anchor - 1, p, b0 + label.boundary - 1]
        isfinite(resp) && resp >= zero(FT) ||
            throw(ArgumentError("Invalid RT response amplitude"))
        amplitudes[row, k] = _supported_product(resp, basis.scale[k])
    end
    validate_basis_amplitudes(amplitudes, length(basis.labels))
    return amplitudes
end
