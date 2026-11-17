# Scalar hooks for ForwardDiff dual numbers.
#
# The package extension CanopyRadiationForwardDiffExt defines these hooks whenever
# ForwardDiff is loaded, so with the extension present this file defines nothing. The
# code below runs only if the extension is absent (for example when the commit that
# adds the ForwardDiff weak dependency is dropped); it is a copy of the extension.

using CanopyRadiation
using ForwardDiff

if Base.get_extension(CanopyRadiation, :CanopyRadiationForwardDiffExt) === nothing
    @eval module TestForwardDiffHooks
    using ForwardDiff: Dual, value, partials
    import CanopyRadiation: primal_type, primal, all_finite, exact_zero, scalar_equal

    primal_type(::Type{<:Dual{T,V}}) where {T,V} = primal_type(V)
    primal(d::Dual) = primal(value(d))
    all_finite(d::Dual) = all_finite(value(d)) && all(all_finite, partials(d))
    exact_zero(d::Dual) = exact_zero(value(d)) && all(exact_zero, partials(d))

    # Equal values and equal derivative components; a plain number has zero derivatives.
    # Duals of different tags (nested differentiation) are compared as constants. The
    # partials are read through their public AbstractVector interface.
    scalar_equal(compare, a::Dual{T}, b::Dual{T}) where {T} =
        scalar_equal(compare, value(a), value(b)) &&
        _partials_equal(compare, partials(a), partials(b))
    scalar_equal(compare, a::Dual, b::Dual) =
        _constant_equal(compare, a, b) && _constant_equal(compare, b, a)
    scalar_equal(compare, a::Dual, b::Real) = _constant_equal(compare, a, b)
    scalar_equal(compare, a::Real, b::Dual) = _constant_equal(compare, b, a)

    _constant_equal(compare, a::Dual, b) =
        scalar_equal(compare, value(a), b) && all(p -> scalar_equal(compare, p, zero(p)), partials(a))
    _partials_equal(compare, p, q) =
        length(p) == length(q) && all(i -> scalar_equal(compare, p[i], q[i]), eachindex(p, q))
    end
end
