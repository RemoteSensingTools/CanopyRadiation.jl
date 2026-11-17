# ============================================================================
# Shared behaviour of the records in two_stream.jl and emission_transport.jl:
# value equality, hashing, broadcasting as a scalar, and conversion to a scalar
# type of the same precision
# ============================================================================

# Internal supertype of every record defined in two_stream.jl and
# emission_transport.jl (two-stream methods, layer rates and operators, volume
# emissions, emitting layers). Element types are declared per record type.
abstract type RadiationRecord end

broadcastable(x::RadiationRecord) = Ref(x)

# A record inside an array or a tuple contributes its element type to the precision
# policy, so a vector of layer operators is checked like its scalars.
@inline _element_scalar_type(::Type{R}) where {R<:RadiationRecord} = eltype(R)

# Operands of the precision check for a tuple of records: the tuple itself when all
# records have one type (checked once, however long the column), else each record.
@inline _precision_operands(t::Tuple{T,Vararg{T}}) where {T} = (t,)
@inline _precision_operands(t::Tuple) = t

@inline _fields(x::T) where {T} = ntuple(i -> getfield(x, i), Val(fieldcount(T)))
@inline _same_kind(a, b) = nameof(typeof(a)) === nameof(typeof(b))

# Host-memory linear view of an array: arrays in host memory as they are, device
# arrays copied once (equality and hashing only, never in a kernel).
@inline _host_linear(A::Array) = A
@inline _host_linear(A::SubArray{<:Any,<:Any,<:Array}) = A
@inline _host_linear(A::AbstractArray{T}) where {T} = copyto!(Vector{T}(undef, length(A)), A)

# Componentwise comparison through the `scalar_equal` hook: value and derivative
# components, identical on every ForwardDiff version. Tuple recursion keeps the
# result a `Bool` and the call free of allocations for scalar records. Array fields
# (QuadratureEmission) compare equal axes and then every element.
@inline _all_equal(compare, ::Tuple{}, ::Tuple{}) = true
@inline _all_equal(compare, ::Tuple, ::Tuple) = false
@inline _all_equal(compare, a::Tuple{Any,Vararg{Any,N}}, b::Tuple{Any,Vararg{Any,N}}) where {N} =
    _value_equal(compare, first(a), first(b)) &&
    _all_equal(compare, Base.tail(a), Base.tail(b))
@inline _value_equal(compare, x::RadiationRecord, y::RadiationRecord) =
    _record_equal(compare, x, y)
@inline _value_equal(compare, x::AbstractArray, y::AbstractArray) =
    axes(x) == axes(y) && _elements_equal(compare, _host_linear(x), _host_linear(y))
@inline _value_equal(compare, x, y) = scalar_equal(compare, x, y)
@inline _record_equal(compare, a, b) =
    _same_kind(a, b) && _all_equal(compare, _fields(a), _fields(b))

function _elements_equal(compare, a, b)
    for (x, y) in zip(a, b)
        scalar_equal(compare, x, y) || return false
    end
    return true
end

Base.:(==)(a::RadiationRecord, b::RadiationRecord) = _record_equal(==, a, b)
Base.isequal(a::RadiationRecord, b::RadiationRecord) = _record_equal(isequal, a, b)

# Hash the primal value with the zero sign normalised, so records that compare equal
# (`-0.0 == 0.0`, or a dual with zero derivative and a plain float) hash equal. Array
# fields hash their axes, then every element.
Base.hash(a::RadiationRecord, h::UInt) = _hash_values(_fields(a), hash(nameof(typeof(a)), h))
@inline _hash_values(::Tuple{}, h::UInt) = h
@inline _hash_values(t::Tuple, h::UInt) = _hash_values(Base.tail(t), _hash_value(first(t), h))
@inline _hash_value(x::RadiationRecord, h::UInt) = hash(x, h)
function _hash_value(x::AbstractArray, h::UInt)
    h = hash(axes(x), h)
    for v in _host_linear(x)
        h = _hash_value(v, h)
    end
    return h
end
@inline _hash_value(x, h::UInt) = (p = primal(x); hash(ifelse(iszero(p), zero(p), p), h))

# The same kind of record with every scalar converted to the scalar type of `R`,
# which must have the same precision (Float64 to a Float64 dual, not Float32 to
# Float64). Used when a record is stored in a container of the promoted type or
# enters a kernel whose other inputs are duals.
@inline _convert_record(::Type{R}, x::R) where {R} = x
@inline function _convert_record(::Type{R}, x) where {R}
    same_precision(zero(eltype(R)), x)
    return R(ntuple(i -> convert(fieldtype(R, i), getfield(x, i)), Val(fieldcount(R)))...)
end

# Every record of a tuple converted to record type `R` (no work when all already are).
@inline _convert_tuple(::Type{R}, t::Tuple{Vararg{R}}) where {R} = t
@inline _convert_tuple(::Type{R}, t::Tuple) where {R} =
    (convert(R, first(t)), _convert_tuple(R, Base.tail(t))...)
