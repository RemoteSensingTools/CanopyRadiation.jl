# ============================================================================
# Scalar hooks and the precision policy
# ============================================================================
#
# The five hooks are exact for plain floating-point numbers. A number type that
# carries derivative information overloads them (for ForwardDiff.Dual a package
# extension, or the tests when no extension is present); the code converted to this
# policy (two_stream.jl, emission_transport.jl, column_sweeps.jl, column_batch.jl)
# inspects the internals of a scalar type nowhere else. The remaining files still use
# the reflection helpers in scalars.jl.

"""
    CanopyRadiation.primal_type(::Type{T}) -> Type

Floating-point type that carries the value of scalar type `T`. The default returns
`T`; a derivative-carrying number type returns the floating-point type of its value
(for `ForwardDiff.Dual{Tag,Float64,N}` this is `Float64`).

The precision policy compares `primal_type` of every scalar that meets in a record
or kernel: they must agree and be an `AbstractFloat`.

# Examples
```jldoctest
julia> CanopyRadiation.primal_type(Float32)
Float32
```
"""
primal_type(::Type{T}) where {T} = T

"""
    CanopyRadiation.primal(x)

Value of scalar `x` without derivative information, in the units of `x`. Used for
the comparisons that select a numerical branch or validate an input (thin-slab norm,
series cut-offs, leaf optical bounds, depths) and for hashing, so neither depends on
how a number type compares derivative components.

# Examples
```jldoctest
julia> CanopyRadiation.primal(2.5f0)
2.5f0
```
"""
primal(x) = x

"""
    CanopyRadiation.all_finite(x) -> Bool

Whether the value and every derivative component of scalar `x` are finite. Used by
the validated constructors ([`NativeTwoStream`](@ref), [`QuadratureEmission`](@ref))
and by [`foursail_canopy_rates`](@ref).

# Examples
```jldoctest
julia> (CanopyRadiation.all_finite(1.5), CanopyRadiation.all_finite(NaN))
(true, false)
```
"""
all_finite(x) = isfinite(x)

"""
    CanopyRadiation.exact_zero(x) -> Bool

Whether the value and every derivative component of scalar `x` are zero (either sign).
[`layer_operator`](@ref) uses it for the empty layer: a leaf area index that is an
exact zero returns the identity operator, while a zero value with a nonzero
derivative component evaluates the slab and keeps the right derivative.

# Examples
```jldoctest
julia> (CanopyRadiation.exact_zero(-0.0), CanopyRadiation.exact_zero(1.0e-12))
(true, false)
```
"""
exact_zero(x) = iszero(x)

"""
    CanopyRadiation.scalar_equal(compare, x, y) -> Bool

Whether scalars `x` and `y` are equal under `compare` (`==` or `isequal`) in value
**and** in every derivative component; a plain number has zero derivative components.
Record equality (`==`, `isequal`) is built on it, so it does not depend on how a
number type defines `==` (ForwardDiff 0.10 compares values only, 1.x also partials).
The default is `compare(x, y)`.

# Examples
```jldoctest
julia> (CanopyRadiation.scalar_equal(==, -0.0, 0.0), CanopyRadiation.scalar_equal(isequal, -0.0, 0.0))
(true, false)
```
"""
scalar_equal(compare, x, y) = compare(x, y)::Bool

# ============================================================================
# Precision policy, in one place: every scalar that meets in a record or a kernel
# must carry the same floating-point type. Float64 with a Float64-valued dual is
# accepted (and promotes); Float32 with Float64 is not; integers, rationals,
# irrationals and booleans are not radiation scalars. Records and arrays take part
# through their element type. The checks depend only on types, so they fold away in
# compiled kernels (also on GPUs); the messages are static.
# ============================================================================

const _NOT_FLOAT = "canopy radiation scalars must be floating point"
const _MIXED = "mixed floating-point precision in canopy radiation"

# Scalar type of a kernel input: a scalar's own type, the element type of a record
# or an array, the element type of the records in an array or a homogeneous tuple.
@inline _scalar_type(x) = _element_scalar_type(eltype(typeof(x)))
@inline _element_scalar_type(::Type{T}) where {T} = T
@inline _float_type(x) = primal_type(_scalar_type(x))
@inline _is_float(::Type{P}) where {P} = P <: AbstractFloat

@inline function same_precision(a, rest...)
    P = _float_type(a)
    _is_float(P) || throw(ArgumentError(_NOT_FLOAT))
    _same_precision(P, rest...)
    return nothing
end
@inline _same_precision(::Type) = nothing
@inline function _same_precision(::Type{P}, b, rest...) where {P}
    Q = _float_type(b)
    _is_float(Q) || throw(ArgumentError(_NOT_FLOAT))
    Q === P || throw(ArgumentError(_MIXED))
    return _same_precision(P, rest...)
end

# Scalar type every input of a kernel is converted to: checks the policy, then
# promotes the element types (Float64 with a Float64 dual gives the dual).
@inline function common_scalar_type(a, rest...)
    same_precision(a, rest...)
    return promote_type(_scalar_type(a), map(_scalar_type, rest)...)
end

# Validity checks on primal values; derivative components only need to be finite.
@inline function _checked(x, bound_ok, message)
    all_finite(x) && bound_ok(primal(x)) || throw(ArgumentError(message))
    return x
end
@inline _positive(x) = x > zero(x)
@inline _nonnegative(x) = x >= zero(x)
@inline _unit_interval(x) = zero(x) <= x <= one(x)
