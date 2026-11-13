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

@inline _host_linear(A::Array) = A
@inline _host_linear(A::SubArray{<:Any,<:Any,<:Array}) = A
@inline _host_linear(A::AbstractArray{T}) where {T} = copyto!(Vector{T}(undef, length(A)), A)

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

"""
    unit_sin_from_cos(mu)

Evaluate `sqrt(max(0, (1 - mu) * (1 + mu)))` cleanly across primal floats and
automatic-differentiation dual numbers.

When `(1 - mu) * (1 + mu)` is identically zero (`exact_zero_scalar`) or has a
nonpositive primal value (such as `mu = ±1` at zenith illumination or nadir
viewing), return `zero(typeof(mu))` directly without invoking `sqrt` at zero,
avoiding singular `NaN` tangents when differentiating with respect to other
parameters.

# Examples
```jldoctest
julia> CanopyRadiation.unit_sin_from_cos(0.6)
0.8

julia> CanopyRadiation.unit_sin_from_cos(1.0)
0.0
```
"""
@inline function unit_sin_from_cos(mu::FT) where {FT<:Real}
    require_continuous_scalar(FT, unit_sin_from_cos, (mu,))
    P = primal_float_type(FT)
    s2 = (one(FT) - mu) * (one(FT) + mu)
    if exact_zero_scalar(s2) || primal_scalar(s2) <= zero(P)
        return zero(FT)
    end
    return sqrt(s2)
end

@inline function neumaier_add(sum::FT, comp::FT, term::FT) where {FT<:Real}
    t = sum + term
    comp += ifelse(abs(sum) >= abs(term), (sum - t) + term, (term - t) + sum)
    return t, comp
end

@inline function even_cosh_c0(q::FT) where {FT<:Real}
    r = one(FT)
    r = one(FT) + q * r * inv(FT(240))
    r = one(FT) + q * r * inv(FT(182))
    r = one(FT) + q * r * inv(FT(132))
    r = one(FT) + q * r * inv(FT(90))
    r = one(FT) + q * r * inv(FT(56))
    r = one(FT) + q * r * inv(FT(30))
    r = one(FT) + q * r * inv(FT(12))
    r = one(FT) + q * r * inv(FT(2))
    return r
end

@inline function even_sinh_s0(q::FT) where {FT<:Real}
    r = one(FT)
    r = one(FT) + q * r * inv(FT(272))
    r = one(FT) + q * r * inv(FT(210))
    r = one(FT) + q * r * inv(FT(156))
    r = one(FT) + q * r * inv(FT(110))
    r = one(FT) + q * r * inv(FT(72))
    r = one(FT) + q * r * inv(FT(42))
    r = one(FT) + q * r * inv(FT(20))
    r = one(FT) + q * r * inv(FT(6))
    return r
end

@inline function even_cosh_m1_s1(q::FT) where {FT<:Real}
    r = one(FT)
    r = one(FT) + q * r * inv(FT(306))
    r = one(FT) + q * r * inv(FT(240))
    r = one(FT) + q * r * inv(FT(182))
    r = one(FT) + q * r * inv(FT(132))
    r = one(FT) + q * r * inv(FT(90))
    r = one(FT) + q * r * inv(FT(56))
    r = one(FT) + q * r * inv(FT(30))
    r = one(FT) + q * r * inv(FT(12))
    return inv(FT(2)) * r
end

@inline _index_sets_overlap(i::Integer, j::Integer) = i == j
@inline _index_sets_overlap(i::Integer, r::AbstractRange) = i in r
@inline _index_sets_overlap(r::AbstractRange, j::Integer) = j in r
@inline _index_sets_overlap(r1::AbstractRange, r2::AbstractRange) = !isempty(intersect(r1, r2))
@inline _index_sets_overlap(_, _) = true

function _unwrap_nonview_parent(A::AbstractArray)
    curr = A
    while !(curr isa SubArray)
        p = parent(curr)
        p === curr && return curr
        curr = p
    end
    return curr
end

function _arrays_alias(a::AbstractArray, b::AbstractArray)
    (isempty(a) || isempty(b)) && return false
    ua = _unwrap_nonview_parent(a)
    ub = _unwrap_nonview_parent(b)
    ua === ub && return true
    ra = ua isa SubArray ? _unwrap_nonview_parent(parent(ua)) : ua
    rb = ub isa SubArray ? _unwrap_nonview_parent(parent(ub)) : ub
    if (ra isa SubArray) || (rb isa SubArray)
        while parent(ra) !== ra
            ra = parent(ra)
        end
        while parent(rb) !== rb
            rb = parent(rb)
        end
        return ra === rb
    end
    ra === rb || return false
    ia = ua isa SubArray ? parentindices(ua) : axes(ra)
    ib = ub isa SubArray ? parentindices(ub) : axes(rb)
    length(ia) == length(ib) || return true
    for d in eachindex(ia, ib)
        _index_sets_overlap(ia[d], ib[d]) || return false
    end
    return true
end

function _disjoint_hemispheric_buffers(outputs, inputs)
    for (i, a) in enumerate(outputs)
        any(b -> _arrays_alias(a, b), inputs) &&
            throw(ArgumentError("Output buffer aliases source input"))
        any(b -> _arrays_alias(a, b), outputs[1:(i - 1)]) &&
            throw(ArgumentError("Output buffers alias each other"))
    end
    return nothing
end
