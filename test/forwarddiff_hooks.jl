# Scalar hooks for ForwardDiff dual numbers.
#
# The package extension CanopyRadiationForwardDiffExt defines these hooks when
# ForwardDiff is loaded. If the extension is not part of the package (it is a weak
# dependency awaiting a decision), the tests define the same hooks here, so the AD
# tests do not depend on it.

using CanopyRadiation
using ForwardDiff

if Base.get_extension(CanopyRadiation, :CanopyRadiationForwardDiffExt) === nothing
    @eval module TestForwardDiffHooks
    using ForwardDiff: Dual, value, partials
    import CanopyRadiation: primal_type, primal, all_finite, exact_zero, scalar_equal

    primal_type(::Type{<:Dual{T,V}}) where {T,V} = primal_type(V)
    primal(d::Dual) = primal(value(d))
    all_finite(d::Dual) = all_finite(value(d)) && all(all_finite, partials(d).values)
    exact_zero(d::Dual) = exact_zero(value(d)) && all(exact_zero, partials(d).values)

    scalar_equal(compare, a::Dual{T}, b::Dual{T}) where {T} =
        scalar_equal(compare, value(a), value(b)) &&
        _partials_equal(compare, partials(a).values, partials(b).values)
    scalar_equal(compare, a::Dual, b::Dual) =
        _constant_equal(compare, a, b) && _constant_equal(compare, b, a)
    scalar_equal(compare, a::Dual, b::Real) = _constant_equal(compare, a, b)
    scalar_equal(compare, a::Real, b::Dual) = _constant_equal(compare, b, a)

    _constant_equal(compare, a::Dual, b) =
        scalar_equal(compare, value(a), b) &&
        all(p -> scalar_equal(compare, p, zero(p)), partials(a).values)
    _partials_equal(compare, p::NTuple{N}, q::NTuple{N}) where {N} =
        all(map((x, y) -> scalar_equal(compare, x, y), p, q))
    _partials_equal(compare, p::Tuple, q::Tuple) = false
    end
end
