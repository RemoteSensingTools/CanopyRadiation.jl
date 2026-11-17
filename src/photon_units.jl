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
    _map_rows!(
        KernelAbstractions.get_backend(output),
        _PerWavelengthEnergy(coefficient),
        output,
        photons,
        wavelengths_nm,
    )
    return output
end
