"""
    AngularEmissionBoundary(reflection, emission, mu, weights, fourier_order)

Represent top-of-canopy boundary response `U = emission + reflection * D` for
one Fourier azimuth order `fourier_order >= 0` on discrete-ordinate polar
cosines `mu` and quadrature `weights`.

# Examples
```jldoctest
julia> b = AngularEmissionBoundary([0.1;;], [0.5;;], [0.7], [1.0], 0);

julia> b.fourier_order
0
```
"""
struct AngularEmissionBoundary{
    FT<:Real,
    R<:AbstractMatrix{FT},
    S<:AbstractMatrix{FT},
    V<:AbstractVector{FT},
} <: RadiationRecord
    reflection::R
    emission::S
    mu::V
    weights::V
    fourier_order::Int

    function AngularEmissionBoundary(
        reflection::R,
        emission::S,
        mu::V,
        weights::V,
        fourier_order::Integer,
    ) where {FT<:Real,R<:AbstractMatrix{FT},S<:AbstractMatrix{FT},V<:AbstractVector{FT}}
        same_precision(mu)
        n = length(mu)
        n > 0 &&
            axes(mu) == axes(weights) &&
            axes(reflection) == (axes(mu, 1), axes(mu, 1)) &&
            axes(emission, 1) == axes(mu, 1) ||
            throw(DimensionMismatch("Boundary operators, sources, and ordinates must align"))
        fourier_order >= 0 || throw(ArgumentError("Fourier order must be nonnegative"))
        return new{FT,R,S,V}(reflection, emission, mu, weights, Int(fourier_order))
    end
end

Base.eltype(::Type{<:AngularEmissionBoundary{FT}}) where {FT} = FT

"""
    AbstractAtmosphericTransfer

Supertype for source-free atmospheric operators above an
[`AngularEmissionBoundary`](@ref) ([`VacuumAtmosphere`](@ref) and
[`AngularAtmosphericTransfer`](@ref)).
"""
abstract type AbstractAtmosphericTransfer <: RadiationRecord end

"""
    VacuumAtmosphere()

Transparent atmosphere with zero downward backscatter and unit upward
transmission.

# Examples
```jldoctest
julia> VacuumAtmosphere()
VacuumAtmosphere()
```
"""
struct VacuumAtmosphere <: AbstractAtmosphericTransfer end

"""
    AngularAtmosphericTransfer(return_down, transmit_up, mu, weights, fourier_order)

Source-free atmosphere seen from the top of the canopy (`D = return_down * U`,
`TOA = transmit_up * U`) on discrete-ordinate polar cosines `mu` and `weights`.

# Examples
```jldoctest
julia> atm = AngularAtmosphericTransfer([0.05;;], [0.9;;], [0.7], [1.0], 0);

julia> atm.fourier_order
0
```
"""
struct AngularAtmosphericTransfer{
    FT<:Real,
    R<:AbstractMatrix{FT},
    T<:AbstractMatrix{FT},
    V<:AbstractVector{FT},
} <: AbstractAtmosphericTransfer
    return_down::R
    transmit_up::T
    mu::V
    weights::V
    fourier_order::Int

    function AngularAtmosphericTransfer(
        return_down::R,
        transmit_up::T,
        mu::V,
        weights::V,
        fourier_order::Integer,
    ) where {FT<:Real,R<:AbstractMatrix{FT},T<:AbstractMatrix{FT},V<:AbstractVector{FT}}
        same_precision(mu)
        n = length(mu)
        n > 0 &&
            axes(mu) == axes(weights) &&
            axes(return_down) == axes(transmit_up) == (axes(mu, 1), axes(mu, 1)) ||
            throw(DimensionMismatch("Atmospheric operators and ordinates must align"))
        fourier_order >= 0 || throw(ArgumentError("Fourier order must be nonnegative"))
        return new{FT,R,T,V}(return_down, transmit_up, mu, weights, Int(fourier_order))
    end
end

Base.eltype(::Type{<:AngularAtmosphericTransfer{FT}}) where {FT} = FT

# Operator of the promoted scalar type: as it is, or converted elementwise (plain
# operators coupled with dual ones).
@inline _with_eltype(::Type{FT}, A::AbstractArray{FT}) where {FT} = A
@inline _with_eltype(::Type{FT}, A::AbstractArray) where {FT} = convert.(FT, A)

function _check_shared_ordinate_grid(
    canopy::AngularEmissionBoundary,
    atmosphere::AngularAtmosphericTransfer,
)
    canopy.fourier_order == atmosphere.fourier_order &&
        axes(canopy.mu) == axes(atmosphere.mu) &&
        primal.(canopy.mu) == primal.(atmosphere.mu) &&
        primal.(canopy.weights) == primal.(atmosphere.weights) ||
        throw(ArgumentError("Coupling requires identical ordinate grids, weights, and Fourier order"))
    return nothing
end

"""
    couple_emission(canopy::AngularEmissionBoundary, atmosphere::AbstractAtmosphericTransfer)

Solve coupled canopy-atmosphere multiple scattering (`U = (I - R_c R_a) \\ e`,
`D = R_a U`, `TOA = T_a U`), returning `(; canopy_up, canopy_down, top_up)`.

Supports compatible mixed primal and automatic-differentiation scalar types.

# Examples
```jldoctest
julia> b = AngularEmissionBoundary([0.2;;], [1.0;;], [0.7], [1.0], 0);

julia> atm = AngularAtmosphericTransfer([0.1;;], [0.8;;], [0.7], [1.0], 0);

julia> c = couple_emission(b, atm);

julia> isapprox(c.canopy_up[1, 1], 1.0 / (1.0 - 0.02); atol = 1e-15)
true
```
"""
function couple_emission(canopy::AngularEmissionBoundary, atmosphere::AngularAtmosphericTransfer)
    FT = common_scalar_type(canopy, atmosphere)
    _check_shared_ordinate_grid(canopy, atmosphere)
    rc = _with_eltype(FT, canopy.reflection)
    em = _with_eltype(FT, canopy.emission)
    ra = _with_eltype(FT, atmosphere.return_down)
    ta = _with_eltype(FT, atmosphere.transmit_up)
    factor = lu(I - rc * ra)
    canopy_up = factor \ em
    canopy_down = ra * canopy_up
    top_up = ta * canopy_up
    return (; canopy_up, canopy_down, top_up)
end

function couple_emission(canopy::AngularEmissionBoundary, ::VacuumAtmosphere)
    canopy_up = copy(canopy.emission)
    canopy_down = zero(canopy.emission)
    top_up = copy(canopy.emission)
    return (; canopy_up, canopy_down, top_up)
end

"""
    one_way_emission(canopy::AngularEmissionBoundary, atmosphere::AbstractAtmosphericTransfer)

Evaluate single-pass upward atmospheric transmission `T_a * emission` ignoring
downward atmospheric return.

# Examples
```jldoctest
julia> b = AngularEmissionBoundary([0.2;;], [1.0;;], [0.7], [1.0], 0);

julia> one_way_emission(b, VacuumAtmosphere())
1×1 Matrix{Float64}:
 1.0
```
"""
function one_way_emission(canopy::AngularEmissionBoundary, atmosphere::AngularAtmosphericTransfer)
    FT = common_scalar_type(canopy, atmosphere)
    _check_shared_ordinate_grid(canopy, atmosphere)
    em = _with_eltype(FT, canopy.emission)
    ta = _with_eltype(FT, atmosphere.transmit_up)
    return ta * em
end

one_way_emission(canopy::AngularEmissionBoundary, ::VacuumAtmosphere) = copy(canopy.emission)
