"""
    adapt_radiation_storage(target, x)

Transfer array-backed radiation objects (`QuadratureEmission`,
`AngularEmissionBoundary`, `AngularAtmosphericTransfer`, `LeafEmissionMatrix`,
`DualLeafEmissionMatrix`, `LayerHemisphericSource`,
`DualLayerHemisphericSource`, `ColumnFluxWorkspace`, `EmittingColumnWorkspace`)
onto array container type `target` (such as `Array` or `CuArray` when `CUDA` is
loaded).
"""
adapt_radiation_storage(::Type{Array}, x::Array) = x
adapt_radiation_storage(::Type{Array}, x::AbstractArray) = Array(x)
adapt_radiation_storage(::Type{T}, x::NoLeafEmission) where {T<:AbstractArray} = x
adapt_radiation_storage(::Type{T}, x::VacuumAtmosphere) where {T<:AbstractArray} = x

function adapt_radiation_storage(
    ::Type{T},
    b::AngularEmissionBoundary,
) where {T<:AbstractArray}
    return AngularEmissionBoundary(
        adapt_radiation_storage(T, b.reflection),
        adapt_radiation_storage(T, b.emission),
        adapt_radiation_storage(T, b.mu),
        adapt_radiation_storage(T, b.weights),
        b.fourier_order,
    )
end

function adapt_radiation_storage(
    ::Type{T},
    a::AngularAtmosphericTransfer,
) where {T<:AbstractArray}
    return AngularAtmosphericTransfer(
        adapt_radiation_storage(T, a.return_down),
        adapt_radiation_storage(T, a.transmit_up),
        adapt_radiation_storage(T, a.mu),
        adapt_radiation_storage(T, a.weights),
        a.fourier_order,
    )
end

function adapt_radiation_storage(::Type{T}, q::QuadratureEmission) where {T<:AbstractArray}
    return QuadratureEmission(
        adapt_radiation_storage(T, q.depths),
        adapt_radiation_storage(T, q.weights),
        adapt_radiation_storage(T, q.upward),
        adapt_radiation_storage(T, q.downward);
        validate = false,
    )
end

function adapt_radiation_storage(::Type{T}, m::LeafEmissionMatrix) where {T<:AbstractArray}
    return LeafEmissionMatrix(
        adapt_radiation_storage(T, m.front),
        adapt_radiation_storage(T, m.back),
        adapt_radiation_storage(T, m.reabsorbed),
        adapt_radiation_storage(T, m.out_of_window),
    )
end

function adapt_radiation_storage(
    ::Type{T},
    d::DualLeafEmissionMatrix,
) where {T<:AbstractArray}
    return DualLeafEmissionMatrix(
        adapt_radiation_storage(T, d.psii),
        adapt_radiation_storage(T, d.psi),
    )
end

function adapt_radiation_storage(
    ::Type{T},
    s::LayerHemisphericSource,
) where {T<:AbstractArray}
    return LayerHemisphericSource(
        s.lai,
        adapt_radiation_storage(T, s.fractional_depths),
        adapt_radiation_storage(T, s.weights),
        adapt_radiation_storage(T, s.plus),
        adapt_radiation_storage(T, s.minus),
        adapt_radiation_storage(T, s.physical_channels);
        validate = false,
    )
end

function adapt_radiation_storage(
    ::Type{T},
    d::DualLayerHemisphericSource,
) where {T<:AbstractArray}
    return DualLayerHemisphericSource(
        adapt_radiation_storage(T, d.psii),
        adapt_radiation_storage(T, d.psi),
    )
end

function adapt_radiation_storage(
    ::Type{T},
    ws::ColumnFluxWorkspace,
) where {T<:AbstractArray}
    return ColumnFluxWorkspace(
        adapt_radiation_storage(T, ws.direct),
        adapt_radiation_storage(T, ws.down),
        adapt_radiation_storage(T, ws.up),
        adapt_radiation_storage(T, ws.absorption),
    )
end

function adapt_radiation_storage(
    ::Type{T},
    ws::EmittingColumnWorkspace,
) where {T<:AbstractArray}
    return EmittingColumnWorkspace(
        adapt_radiation_storage(T, ws.down),
        adapt_radiation_storage(T, ws.up),
        adapt_radiation_storage(T, ws.net_layer),
        adapt_radiation_storage(T, ws.absorption),
    )
end
