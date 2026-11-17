const UnsupportedDiscreteScalar = Union{Bool,Integer,Rational,AbstractIrrational}

@inline primal_float_type(::Type{T}) where {T<:AbstractFloat} = T
@inline function primal_float_type(::Type{T}) where {T<:Real}
    T <: UnsupportedDiscreteScalar && return T
    isconcretetype(T) || return Any
    fieldcount(T) == 0 && return Any
    V = fieldtype(T, 1)
    V === T && return Any
    return primal_float_type(V)
end

@inline function is_continuous_scalar(::Type{T}) where {T}
    isconcretetype(T) || return false
    T <: Real || return false
    T <: UnsupportedDiscreteScalar && return false
    P = primal_float_type(T)
    return isconcretetype(P) && (P <: AbstractFloat)
end

@inline primal_scalar(x::AbstractFloat) = x
@inline function primal_scalar(x::T) where {T<:Real}
    is_continuous_scalar(T) || throw(MethodError(primal_scalar, (x,)))
    return primal_scalar(getfield(x, 1))
end

@generated function finite_scalar(x::T) where {T}
    if T <: AbstractFloat
        return :(isfinite(x))
    elseif T <: Tuple
        N = fieldcount(T)
        N == 0 && return :(true)
        expr = :(finite_scalar(x[1]))
        for i in 2:N
            expr = :($expr && finite_scalar(x[$i]))
        end
        return expr
    elseif isconcretetype(T)
        nf = fieldcount(T)
        nf == 0 && return :(true)
        expr = :(finite_scalar(getfield(x, 1)))
        for i in 2:nf
            expr = :($expr && finite_scalar(getfield(x, $i)))
        end
        return expr
    else
        return :(isfinite(x))
    end
end

@generated function exact_zero_scalar(x::T) where {T}
    if T <: AbstractFloat
        return :(iszero(x))
    elseif T <: Tuple
        N = fieldcount(T)
        N == 0 && return :(true)
        expr = :(exact_zero_scalar(x[1]))
        for i in 2:N
            expr = :($expr && exact_zero_scalar(x[$i]))
        end
        return expr
    elseif isconcretetype(T)
        nf = fieldcount(T)
        nf == 0 && return :(true)
        expr = :(exact_zero_scalar(getfield(x, 1)))
        for i in 2:nf
            expr = :($expr && exact_zero_scalar(getfield(x, $i)))
        end
        return expr
    else
        return :(iszero(x))
    end
end

@inline function require_continuous_scalar(::Type{T}, f, args::Tuple) where {T}
    is_continuous_scalar(T) || throw(MethodError(f, args))
    return nothing
end

@inline function compatible_scalar(::Type{A}, ::Type{B}) where {A<:Real,B<:Real}
    is_continuous_scalar(A) && is_continuous_scalar(B) || return Union{}
    primal_float_type(A) === primal_float_type(B) || return Union{}
    FT = promote_type(A, B)
    return is_continuous_scalar(FT) ? FT : Union{}
end

@inline function promote_compatible_scalar(f, x::T) where {T<:Real}
    require_continuous_scalar(T, f, (x,))
    return T
end

@inline function promote_compatible_scalar(f, x::A, y::B, rest::Real...) where {A<:Real,B<:Real}
    FT = compatible_scalar(A, B)
    FT === Union{} && throw(MethodError(f, (x, y, rest...)))
    return _promote_compatible_tail(f, FT, (x, y, rest...), rest...)
end

@inline _promote_compatible_tail(f, ::Type{FT}, orig::Tuple) where {FT} = FT
@inline function _promote_compatible_tail(
    f,
    ::Type{FT},
    orig::Tuple,
    z::C,
    rest::Real...,
) where {FT,C<:Real}
    FT2 = compatible_scalar(FT, C)
    FT2 === Union{} && throw(MethodError(f, orig))
    return _promote_compatible_tail(f, FT2, orig, rest...)
end

@inline _canonical_zero_float(x::T) where {T<:AbstractFloat} = ifelse(iszero(x), zero(T), x)

@inline function _hash_scalar(x::T, h::UInt) where {T}
    if T <: AbstractFloat
        return hash(_canonical_zero_float(x), h)
    elseif is_continuous_scalar(T)
        return hash(_canonical_zero_float(primal_scalar(x)), h)
    else
        return hash(x, h)
    end
end

function _hash_array(A::AbstractArray, h::UInt)
    h = hash(axes(A), h)
    host = _host_linear(A)
    for x in host
        h = _hash_scalar(x, h)
    end
    return h
end

function _isequal_array(A::AbstractArray, B::AbstractArray)
    axes(A) == axes(B) || return false
    ha = _host_linear(A)
    hb = _host_linear(B)
    @inbounds for i in eachindex(ha, hb)
        isequal(ha[i], hb[i]) || return false
    end
    return true
end

@inline function neumaier_add(sum::FT, comp::FT, term::FT) where {FT<:Real}
    t = sum + term
    comp += ifelse(abs(sum) >= abs(term), (sum - t) + term, (term - t) + sum)
    return t, comp
end
