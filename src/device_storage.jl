"""
    adapt_radiation_storage(target, x)

Transfer array-backed radiation objects (`QuadratureEmission`,
`AngularEmissionBoundary`, `AngularAtmosphericTransfer`, `LeafEmissionMatrix`,
`DualLeafEmissionMatrix`, `LayerHemisphericSource`,
`DualLayerHemisphericSource`, `ColumnFluxWorkspace`, `EmittingColumnWorkspace`)
onto array container type `target` (such as `Array` or `CuArray` when `CUDA` is
loaded).

Every array field is transferred (ranges and static vectors become arrays of the
target type with the same element type), nested records are transferred
recursively, and scalar fields are kept. Objects are rebuilt through their
constructors, so dimension checks run again; `QuadratureEmission` and
`LayerHemisphericSource` skip their value checks (`validate = false`), which would
otherwise read device memory element by element.
"""
adapt_radiation_storage(::Type{Array}, x::Array) = x
adapt_radiation_storage(::Type{Array}, x::AbstractArray) = Array(x)

# The objects that hold arrays (or are disabled placeholders for one) and can move
# between host and device storage.
const _DeviceStorable = Union{
    QuadratureEmission,
    AngularEmissionBoundary,
    VacuumAtmosphere,
    AngularAtmosphericTransfer,
    NoLeafEmission,
    LeafEmissionMatrix,
    DualLeafEmissionMatrix,
    LayerHemisphericSource,
    DualLayerHemisphericSource,
    ColumnFluxWorkspace,
    EmittingColumnWorkspace,
}

function adapt_radiation_storage(::Type{T}, x::_DeviceStorable) where {T<:AbstractArray}
    return _rebuild(x, map(field -> _adapt_field(T, field), _fields(x)))
end

@inline _adapt_field(::Type{T}, x::AbstractArray) where {T} = adapt_radiation_storage(T, x)
@inline _adapt_field(::Type{T}, x::_DeviceStorable) where {T} = adapt_radiation_storage(T, x)
@inline _adapt_field(::Type, x) = x

# Rebuild an object of the same kind from new field values through its positional
# constructor; the two records with value checks on every element skip them.
@inline _rebuild(x, fields::Tuple) = _constructor_of(typeof(x))(fields...)
@inline _rebuild(::QuadratureEmission, fields::Tuple) = QuadratureEmission(fields...; validate = false)
@inline _rebuild(::LayerHemisphericSource, fields::Tuple) =
    LayerHemisphericSource(fields...; validate = false)
@inline _constructor_of(::Type{T}) where {T} = Base.typename(T).wrapper
