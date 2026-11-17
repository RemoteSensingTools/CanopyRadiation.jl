"""
    hemispheric_moment(law::AbstractEmissionAngularLaw, mu, weighted, depth_idx, channel_idx)

Reduce orientation-weighted internal channel strength `weighted[:, depth_idx, channel_idx]`
with leaf-normal cosines `mu` under `law` into `(; plus, minus)` hemisphere
moments at `(depth_idx, channel_idx)`, respecting generic array axes.
"""
@inline function hemispheric_moment(
    law::AbstractEmissionAngularLaw,
    mu::AbstractVector,
    weighted::AbstractArray{<:Real,3},
    depth_idx::Integer,
    channel_idx::Integer,
)
    FT = promote_compatible_scalar(
        hemispheric_moment,
        _law_scalar_probe(law, zero(eltype(mu))),
        zero(eltype(mu)),
        zero(eltype(weighted)),
    )
    @boundscheck begin
        axes(weighted, 1) == axes(mu, 1) ||
            throw(DimensionMismatch("weighted orientation axis must match axes(mu, 1)"))
        (depth_idx in axes(weighted, 2) && channel_idx in axes(weighted, 3)) ||
            throw(BoundsError(weighted, (:, depth_idx, channel_idx)))
    end
    plus, minus = zero(FT), zero(FT)
    @inbounds for o in eachindex(mu)
        e = leaf_hemispheric_emission(law, mu[o], weighted[o, depth_idx, channel_idx], zero(FT))
        plus += convert(FT, e.upward)
        minus += convert(FT, e.downward)
    end
    return (; plus, minus)
end

"""
    hemispheric_bin(plus, minus, front, back, depth_idx)

Expand conditional channel escapes `front` and `back` against pre-integrated
hemisphere moments `plus[depth_idx, :]` and `minus[depth_idx, :]` into
`(; up, down)` hemispheric source strengths at `depth_idx`.
"""
@inline function hemispheric_bin(
    plus::AbstractMatrix,
    minus::AbstractMatrix,
    front::AbstractVector,
    back::AbstractVector,
    depth_idx::Integer,
)
    FT = promote_compatible_scalar(
        hemispheric_bin,
        zero(eltype(plus)),
        zero(eltype(minus)),
        zero(eltype(front)),
        zero(eltype(back)),
    )
    @boundscheck begin
        axes(plus) == axes(minus) &&
            axes(front, 1) == axes(back, 1) == axes(plus, 2) ||
            throw(DimensionMismatch("hemispheric_bin plus, minus, front, and back axes must align"))
        depth_idx in axes(plus, 1) ||
            throw(BoundsError(plus, (depth_idx, :)))
    end
    u, v = zero(FT), zero(FT)
    @inbounds for c in axes(plus, 2)
        fc = convert(FT, front[c])
        bc = convert(FT, back[c])
        p_dc = convert(FT, plus[depth_idx, c])
        m_dc = convert(FT, minus[depth_idx, c])
        u += fc * p_dc + bc * m_dc
        v += fc * m_dc + bc * p_dc
    end
    return (; up = u, down = v)
end

@kernel function _hemispheric_moments_kernel!(
    plus,
    minus,
    law,
    mu,
    weighted,
    o0::Int,
    no::Int,
    d0::Int,
    c0::Int,
)
    d_rel, c_rel = @index(Global, NTuple)
    FT = eltype(plus)
    d_idx = d0 + d_rel - 1
    c_idx = c0 + c_rel - 1
    p_acc = zero(FT)
    m_acc = zero(FT)
    @inbounds for o_step in 0:(no - 1)
        o_idx = o0 + o_step
        e = leaf_hemispheric_emission(law, mu[o_idx], weighted[o_idx, d_idx, c_idx], zero(FT))
        p_acc += convert(FT, e.upward)
        m_acc += convert(FT, e.downward)
    end
    @inbounds begin
        plus[d_idx, c_idx] = p_acc
        minus[d_idx, c_idx] = m_acc
    end
end

"""
    hemispheric_source_moments!(plus, minus, law::AbstractEmissionAngularLaw, mu, weighted)

Compute `(depth × channel)` hemisphere moments `plus` and `minus` from leaf
normal cosines `mu` (`[-1, 1]`) and nonnegative `(orientation × depth × channel)`
`weighted` internal sources across CPU, `OffsetArray`, and GPU arrays.
"""
function hemispheric_source_moments!(
    plus::AbstractMatrix{FT},
    minus::AbstractMatrix{FT},
    law::AbstractEmissionAngularLaw,
    mu::AbstractVector,
    weighted::AbstractArray{<:Real,3},
) where {FT<:Real}
    require_continuous_scalar(FT, hemispheric_source_moments!, (plus, minus, law, mu, weighted))
    FT_req = promote_compatible_scalar(
        hemispheric_source_moments!,
        zero(FT),
        _law_scalar_probe(law, zero(FT)),
        zero(eltype(mu)),
        zero(eltype(weighted)),
    )
    FT_req === FT ||
        throw(MethodError(hemispheric_source_moments!, (plus, minus, law, mu, weighted)))
    axes(weighted, 1) == axes(mu, 1) &&
        axes(plus) == axes(minus) == (axes(weighted, 2), axes(weighted, 3)) ||
        throw(DimensionMismatch("Hemisphere moment axes must align with weighted[orientation, depth, channel]"))
    _disjoint_hemispheric_buffers((plus, minus), (mu, weighted))
    P = primal_float_type(FT)
    for x in _host_linear(mu)
        finite_scalar(x) && -one(P) <= primal_scalar(x) <= one(P) ||
            throw(ArgumentError("Leaf normal cosines mu must be finite and lie in [-1, 1]"))
    end
    for x in _host_linear(weighted)
        finite_scalar(x) && primal_scalar(x) >= zero(P) ||
            throw(ArgumentError("Weighted internal source must be finite and nonnegative"))
    end
    isempty(plus) && return plus, minus
    backend = KernelAbstractions.get_backend(plus)
    if backend isa KernelAbstractions.CPU
        @inbounds for c in axes(weighted, 3), d in axes(weighted, 2)
            h = hemispheric_moment(law, mu, weighted, d, c)
            plus[d, c] = convert(FT, h.plus)
            minus[d, c] = convert(FT, h.minus)
        end
    else
        k! = _hemispheric_moments_kernel!(backend)
        k!(
            plus,
            minus,
            law,
            mu,
            weighted,
            firstindex(mu, 1),
            length(mu),
            firstindex(plus, 1),
            firstindex(plus, 2);
            ndrange = size(plus),
        )
        KernelAbstractions.synchronize(backend)
    end
    return plus, minus
end

@kernel function _expand_hemispheric_kernel!(
    up,
    down,
    plus,
    minus,
    front,
    back,
    d0::Int,
    c0::Int,
    nc::Int,
)
    d_rel = @index(Global, Linear)
    FT = eltype(up)
    d_idx = d0 + d_rel - 1
    u_acc = zero(FT)
    v_acc = zero(FT)
    @inbounds for c_step in 0:(nc - 1)
        c_idx = c0 + c_step
        fc = convert(FT, front[c_idx])
        bc = convert(FT, back[c_idx])
        p_dc = convert(FT, plus[d_idx, c_idx])
        m_dc = convert(FT, minus[d_idx, c_idx])
        u_acc += fc * p_dc + bc * m_dc
        v_acc += fc * m_dc + bc * p_dc
    end
    @inbounds begin
        up[d_idx] = u_acc
        down[d_idx] = v_acc
    end
end

"""
    expand_hemispheric_source!(up, down, plus, minus, front, back)

Expand one emission bin's conditional face escapes `front` and `back` against
hemisphere moments `plus` and `minus` into depth-resolved hemispheric sources
`up` and `down` across CPU, `OffsetArray`, and GPU arrays.
"""
function expand_hemispheric_source!(
    up::AbstractVector{FT},
    down::AbstractVector{FT},
    plus::AbstractMatrix,
    minus::AbstractMatrix,
    front::AbstractVector,
    back::AbstractVector,
) where {FT<:Real}
    require_continuous_scalar(
        FT,
        expand_hemispheric_source!,
        (up, down, plus, minus, front, back),
    )
    FT_req = promote_compatible_scalar(
        expand_hemispheric_source!,
        zero(FT),
        zero(eltype(plus)),
        zero(eltype(minus)),
        zero(eltype(front)),
        zero(eltype(back)),
    )
    FT_req === FT ||
        throw(MethodError(expand_hemispheric_source!, (up, down, plus, minus, front, back)))
    axes(plus) == axes(minus) &&
        axes(up) == axes(down) == (axes(plus, 1),) &&
        axes(front) == axes(back) == (axes(plus, 2),) ||
        throw(DimensionMismatch("Hemisphere expansion axes must align"))
    _disjoint_hemispheric_buffers((up, down), (plus, minus, front, back))
    P = primal_float_type(FT)
    for arr in (plus, minus, front, back)
        for x in _host_linear(arr)
            finite_scalar(x) && primal_scalar(x) >= zero(P) ||
                throw(ArgumentError("Hemisphere expansion inputs must be finite and nonnegative"))
        end
    end
    isempty(up) && return up, down
    backend = KernelAbstractions.get_backend(up)
    if backend isa KernelAbstractions.CPU
        @inbounds for d in axes(plus, 1)
            h = hemispheric_bin(plus, minus, front, back, d)
            up[d] = convert(FT, h.up)
            down[d] = convert(FT, h.down)
        end
    else
        k! = _expand_hemispheric_kernel!(backend)
        k!(
            up,
            down,
            plus,
            minus,
            front,
            back,
            firstindex(plus, 1),
            firstindex(plus, 2),
            size(plus, 2);
            ndrange = length(up),
        )
        KernelAbstractions.synchronize(backend)
    end
    return up, down
end

"""
    LayerHemisphericSource(lai, fractional_depths, weights, plus, minus, physical_channels)

Store pre-reduced unit-interval `fractional_depths ∈ [0, 1]`, quadrature
`weights`, `(depth × channel)` hemisphere moments `plus` and `minus` (per unit
`LAI`), and ground-area integrated `physical_channels` (`μmol` internal photons
per m² ground per s) for one canopy layer and one photosystem class.

Storing `fractional_depths` directly in `[0, 1]` avoids singular division by
`lai` at empty layers (`lai == 0`) and preserves smooth automatic
differentiation with respect to `lai`.

# Examples
```jldoctest
julia> lhs = LayerHemisphericSource(1.5, [0.5], [1.0], [0.4;;], [0.4;;], [1.2]);

julia> lhs.lai
1.5
```
"""
struct LayerHemisphericSource{
    FT<:Real,
    V<:AbstractVector{FT},
    M<:AbstractMatrix{FT},
}
    lai::FT
    fractional_depths::V
    weights::V
    plus::M
    minus::M
    physical_channels::V

    function LayerHemisphericSource(
        lai::FT,
        fractional_depths::V,
        weights::V,
        plus::M,
        minus::M,
        physical_channels::V;
        validate::Bool = true,
    ) where {FT<:Real,V<:AbstractVector{FT},M<:AbstractMatrix{FT}}
        require_continuous_scalar(
            FT,
            LayerHemisphericSource,
            (lai, fractional_depths, weights, plus, minus, physical_channels),
        )
        axes(fractional_depths) == axes(weights) ||
            throw(DimensionMismatch("fractional_depths and weights must share axes"))
        axes(plus) == axes(minus) ==
            (axes(fractional_depths, 1), axes(physical_channels, 1)) ||
            throw(DimensionMismatch("plus/minus must have (depth, channel) axes matching fractional_depths and physical_channels"))
        if validate
            P = primal_float_type(FT)
            finite_scalar(lai) && primal_scalar(lai) >= zero(P) ||
                throw(ArgumentError("lai must be finite and nonnegative"))
            for x in _host_linear(fractional_depths)
                finite_scalar(x) && zero(P) <= primal_scalar(x) <= one(P) ||
                    throw(ArgumentError("fractional_depths must be finite and lie in [0, 1]"))
            end
            for arr in (weights, plus, minus, physical_channels)
                for x in _host_linear(arr)
                    finite_scalar(x) && primal_scalar(x) >= zero(P) ||
                        throw(ArgumentError("LayerHemisphericSource weights, moments, and channels must be finite and nonnegative"))
                end
            end
        end
        return new{FT,V,M}(
            lai,
            fractional_depths,
            weights,
            plus,
            minus,
            physical_channels,
        )
    end
end

Base.eltype(::Type{<:LayerHemisphericSource{FT}}) where {FT} = FT
Base.eltype(s::LayerHemisphericSource) = typeof(s.lai)

Base.:(==)(a::LayerHemisphericSource, b::LayerHemisphericSource) =
    a.lai == b.lai &&
    axes(a.fractional_depths) == axes(b.fractional_depths) &&
    axes(a.plus) == axes(b.plus) &&
    a.fractional_depths == b.fractional_depths &&
    a.weights == b.weights &&
    a.plus == b.plus &&
    a.minus == b.minus &&
    a.physical_channels == b.physical_channels

Base.isequal(a::LayerHemisphericSource, b::LayerHemisphericSource) =
    isequal(a.lai, b.lai) &&
    _isequal_array(a.fractional_depths, b.fractional_depths) &&
    _isequal_array(a.weights, b.weights) &&
    _isequal_array(a.plus, b.plus) &&
    _isequal_array(a.minus, b.minus) &&
    _isequal_array(a.physical_channels, b.physical_channels)

function Base.hash(s::LayerHemisphericSource, h::UInt)
    h = hash(:LayerHemisphericSource, h)
    h = _hash_scalar(s.lai, h)
    h = _hash_array(s.fractional_depths, h)
    h = _hash_array(s.weights, h)
    h = _hash_array(s.plus, h)
    h = _hash_array(s.minus, h)
    return _hash_array(s.physical_channels, h)
end

"""
    DualLayerHemisphericSource(psii, psi)

Pair Photosystem II (`psii`) and Photosystem I (`psi`)
[`LayerHemisphericSource`](@ref) objects sharing identical `lai`,
`fractional_depths`, and depth `weights` (while permitting distinct active
basis channel counts).

# Examples
```jldoctest
julia> s2 = LayerHemisphericSource(1.5, [0.5], [1.0], [0.4;;], [0.4;;], [1.2]);

julia> s1 = LayerHemisphericSource(1.5, [0.5], [1.0], [0.1;;], [0.1;;], [0.3]);

julia> ds = DualLayerHemisphericSource(s2, s1);

julia> ds.psii.lai == ds.psi.lai
true
```
"""
struct DualLayerHemisphericSource{
    L2<:LayerHemisphericSource,
    L1<:LayerHemisphericSource,
}
    psii::L2
    psi::L1

    function DualLayerHemisphericSource(
        psii::L2,
        psi::L1,
    ) where {L2<:LayerHemisphericSource,L1<:LayerHemisphericSource}
        eltype(psii) === eltype(psi) ||
            throw(ArgumentError("PSII and PSI LayerHemisphericSource objects must share scalar type"))
        psii.lai == psi.lai &&
            axes(psii.fractional_depths) == axes(psi.fractional_depths) &&
            _host_linear(psii.fractional_depths) == _host_linear(psi.fractional_depths) &&
            _host_linear(psii.weights) == _host_linear(psi.weights) ||
            throw(ArgumentError("PSII and PSI LayerHemisphericSource objects must share lai, fractional_depths, and weights"))
        return new{L2,L1}(psii, psi)
    end
end

Base.eltype(::Type{<:DualLayerHemisphericSource{L2}}) where {L2} = eltype(L2)
Base.eltype(d::DualLayerHemisphericSource) = eltype(d.psii)
Base.:(==)(a::DualLayerHemisphericSource, b::DualLayerHemisphericSource) =
    (a.psii == b.psii) && (a.psi == b.psi)
Base.isequal(a::DualLayerHemisphericSource, b::DualLayerHemisphericSource) =
    isequal(a.psii, b.psii) && isequal(a.psi, b.psi)

function Base.hash(d::DualLayerHemisphericSource, h::UInt)
    h = hash(:DualLayerHemisphericSource, h)
    h = hash(d.psii, h)
    return hash(d.psi, h)
end

@inline _model_scalar_probe(::Nothing, fallback) = fallback
@inline _model_scalar_probe(::NoLeafEmission, fallback) = fallback
@inline _model_scalar_probe(m::LeafEmissionMatrix, _) = zero(eltype(m))
@inline function _model_scalar_probe(d::DualLeafEmissionMatrix, fallback)
    d.psii isa LeafEmissionMatrix && return zero(eltype(d.psii))
    d.psi isa LeafEmissionMatrix && return zero(eltype(d.psi))
    return fallback
end

@inline function _layers_scalar_type(f, layers::AbstractVector, fallback::Type = Float64)
    T = eltype(eltype(layers))
    isconcretetype(T) && return T
    isempty(layers) && return fallback
    FT = eltype(layers[firstindex(layers)])
    require_continuous_scalar(FT, f, (layers,))
    for k in eachindex(layers)
        FT = promote_compatible_scalar(f, zero(FT), zero(eltype(layers[k])))
    end
    return FT
end

@inline _validate_layer_model_channels(::LayerHemisphericSource, ::Union{Nothing,NoLeafEmission}) =
    nothing
@inline function _validate_layer_model_channels(
    src::LayerHemisphericSource,
    model::LeafEmissionMatrix,
)
    axes(src.plus, 2) == axes(src.physical_channels, 1) ==
        axes(model.front, 2) == axes(model.out_of_window, 1) ||
        throw(DimensionMismatch("LayerHemisphericSource channel axis must match LeafEmissionMatrix excitation-channel axis"))
    return nothing
end

@inline _validate_dual_layer_model_channels(::DualLayerHemisphericSource, ::NoLeafEmission) =
    nothing
@inline function _validate_dual_layer_model_channels(
    dsrc::DualLayerHemisphericSource,
    dual_model::DualLeafEmissionMatrix,
)
    _validate_layer_model_channels(dsrc.psii, dual_model.psii)
    _validate_layer_model_channels(dsrc.psi, dual_model.psi)
    return nothing
end

function _eval_single_layer_bin!(
    up_buf::AbstractVector{FT_buf},
    down_buf::AbstractVector{FT_buf},
    src::LayerHemisphericSource,
    model::Union{Nothing,NoLeafEmission,LeafEmissionMatrix},
    bin::Integer;
    accumulate::Bool = false,
) where {FT_buf<:Real}
    if model === nothing || model isa NoLeafEmission
        if !accumulate
            fill!(up_buf, zero(FT_buf))
            fill!(down_buf, zero(FT_buf))
        end
        return (; internal = zero(FT_buf), reabsorbed = zero(FT_buf), escaped = zero(FT_buf))
    end
    _validate_layer_model_channels(src, model)
    axes(up_buf, 1) == axes(down_buf, 1) == axes(src.plus, 1) ||
        throw(DimensionMismatch("Layer hemispheric source depth buffer axes must match axes(src.plus, 1)"))
    bin in axes(model.front, 1) || throw(BoundsError(model.front, (bin, :)))
    for d in axes(src.plus, 1)
        u_acc = accumulate ? up_buf[d] : zero(FT_buf)
        v_acc = accumulate ? down_buf[d] : zero(FT_buf)
        @inbounds for c in axes(src.plus, 2)
            fc = convert(FT_buf, model.front[bin, c])
            bc = convert(FT_buf, model.back[bin, c])
            p_dc = convert(FT_buf, src.plus[d, c])
            m_dc = convert(FT_buf, src.minus[d, c])
            u_acc += fc * p_dc + bc * m_dc
            v_acc += fc * m_dc + bc * p_dc
        end
        up_buf[d] = u_acc
        down_buf[d] = v_acc
    end
    internal = zero(FT_buf)
    reabs = zero(FT_buf)
    escaped = zero(FT_buf)
    @inbounds for c in axes(src.physical_channels, 1)
        pc = convert(FT_buf, src.physical_channels[c])
        fc = convert(FT_buf, model.front[bin, c])
        bc = convert(FT_buf, model.back[bin, c])
        rc = convert(FT_buf, model.reabsorbed[bin, c])
        esc_c = fc + bc
        escaped += esc_c * pc
        reabs += rc * pc
        internal += (esc_c + rc) * pc
    end
    return (; internal, reabsorbed = reabs, escaped)
end

"""
    evaluate_hemispheric_sif_bin(layers, emission_model, absorption_rates, backscatter_rates, soil_reflect, bin::Integer)

Evaluate canopy hemispheric fluorescence transport and layer photon-fate budgets
for emission `bin` across single-photosystem (`AbstractVector{<:LayerHemisphericSource}`
with `LeafEmissionMatrix` / `NoLeafEmission`) or dual-photosystem
(`AbstractVector{<:DualLayerHemisphericSource}` with [`DualLeafEmissionMatrix`](@ref)
or `NoLeafEmission`) inputs, supporting mixed primal optical matrices and
automatic-differentiation dual layer sources.

Return `(; down, up, absorption, ground, generated, internal, within_leaf, escaped)`.
"""
function evaluate_hemispheric_sif_bin(
    layers::AbstractVector{<:LayerHemisphericSource},
    model::Union{Nothing,NoLeafEmission,LeafEmissionMatrix},
    absorption_rates::AbstractVector,
    backscatter_rates::AbstractVector,
    soil_reflect::Real,
    bin::Integer,
)
    nl = length(layers)
    length(absorption_rates) == nl && length(backscatter_rates) == nl ||
        throw(DimensionMismatch("Layer rate vectors must match length(layers)"))
    if model isa LeafEmissionMatrix
        bin in axes(model.front, 1) || throw(BoundsError(model.front, (bin, :)))
    end
    FT_fallback = typeof(_model_scalar_probe(model, zero(eltype(absorption_rates))))
    FT_lay = _layers_scalar_type(evaluate_hemispheric_sif_bin, layers, FT_fallback)
    FT_src = promote_compatible_scalar(
        evaluate_hemispheric_sif_bin,
        zero(FT_lay),
        _model_scalar_probe(model, zero(FT_lay)),
    )
    FT_out = promote_compatible_scalar(
        evaluate_hemispheric_sif_bin,
        zero(FT_src),
        zero(eltype(absorption_rates)),
        zero(eltype(backscatter_rates)),
        soil_reflect,
    )
    emitting = Vector{EmittingLayer{FT_out}}(undef, nl)
    internal = Vector{FT_out}(undef, nl)
    within_leaf = Vector{FT_out}(undef, nl)
    escaped = Vector{FT_out}(undef, nl)
    l0 = firstindex(layers)
    a0 = firstindex(absorption_rates)
    b0 = firstindex(backscatter_rates)
    for s in 0:(nl - 1)
        src = layers[l0 + s]
        _validate_layer_model_channels(src, model)
        up_d = similar(src.fractional_depths, FT_src)
        dn_d = similar(src.fractional_depths, FT_src)
        bud = _eval_single_layer_bin!(up_d, dn_d, src, model, bin; accumulate = false)
        internal[s + 1] = convert(FT_out, bud.internal)
        within_leaf[s + 1] = convert(FT_out, bud.reabsorbed)
        escaped[s + 1] = convert(FT_out, bud.escaped)
        emitting[s + 1] = quadrature_emitting_layer(
            absorption_rates[a0 + s],
            backscatter_rates[b0 + s],
            src.lai,
            src.fractional_depths,
            src.weights,
            up_d,
            dn_d,
        )
    end
    flux = solve_emission(emitting, convert(FT_out, soil_reflect))
    return (;
        down = flux.down,
        up = flux.up,
        absorption = flux.absorption,
        ground = flux.ground,
        generated = flux.generated,
        internal,
        within_leaf,
        escaped,
    )
end

function evaluate_hemispheric_sif_bin(
    layers::AbstractVector{<:DualLayerHemisphericSource},
    dual_model::Union{DualLeafEmissionMatrix,NoLeafEmission},
    absorption_rates::AbstractVector,
    backscatter_rates::AbstractVector,
    soil_reflect::Real,
    bin::Integer,
)
    dm = dual_model isa NoLeafEmission ?
         DualLeafEmissionMatrix(NoLeafEmission(), NoLeafEmission()) : dual_model
    nl = length(layers)
    length(absorption_rates) == nl && length(backscatter_rates) == nl ||
        throw(DimensionMismatch("Layer rate vectors must match length(layers)"))
    if dm.psii isa LeafEmissionMatrix
        bin in axes(dm.psii.front, 1) ||
            throw(BoundsError(dm.psii.front, (bin, :)))
    end
    if dm.psi isa LeafEmissionMatrix
        bin in axes(dm.psi.front, 1) ||
            throw(BoundsError(dm.psi.front, (bin, :)))
    end
    FT_fallback = typeof(_model_scalar_probe(dm, zero(eltype(absorption_rates))))
    FT_lay = _layers_scalar_type(evaluate_hemispheric_sif_bin, layers, FT_fallback)
    FT_src = promote_compatible_scalar(
        evaluate_hemispheric_sif_bin,
        zero(FT_lay),
        _model_scalar_probe(dm, zero(FT_lay)),
    )
    FT_out = promote_compatible_scalar(
        evaluate_hemispheric_sif_bin,
        zero(FT_src),
        zero(eltype(absorption_rates)),
        zero(eltype(backscatter_rates)),
        soil_reflect,
    )
    emitting = Vector{EmittingLayer{FT_out}}(undef, nl)
    internal = Vector{FT_out}(undef, nl)
    within_leaf = Vector{FT_out}(undef, nl)
    escaped = Vector{FT_out}(undef, nl)
    l0 = firstindex(layers)
    a0 = firstindex(absorption_rates)
    b0 = firstindex(backscatter_rates)
    for s in 0:(nl - 1)
        dsrc = layers[l0 + s]
        _validate_dual_layer_model_channels(dsrc, dm)
        up_d = similar(dsrc.psii.fractional_depths, FT_src)
        dn_d = similar(dsrc.psii.fractional_depths, FT_src)
        b2 = _eval_single_layer_bin!(
            up_d,
            dn_d,
            dsrc.psii,
            dm.psii,
            bin;
            accumulate = false,
        )
        b1 = _eval_single_layer_bin!(
            up_d,
            dn_d,
            dsrc.psi,
            dm.psi,
            bin;
            accumulate = true,
        )
        internal[s + 1] = convert(FT_out, b2.internal + b1.internal)
        within_leaf[s + 1] = convert(FT_out, b2.reabsorbed + b1.reabsorbed)
        escaped[s + 1] = convert(FT_out, b2.escaped + b1.escaped)
        emitting[s + 1] = quadrature_emitting_layer(
            absorption_rates[a0 + s],
            backscatter_rates[b0 + s],
            dsrc.psii.lai,
            dsrc.psii.fractional_depths,
            dsrc.psii.weights,
            up_d,
            dn_d,
        )
    end
    flux = solve_emission(emitting, convert(FT_out, soil_reflect))
    return (;
        down = flux.down,
        up = flux.up,
        absorption = flux.absorption,
        ground = flux.ground,
        generated = flux.generated,
        internal,
        within_leaf,
        escaped,
    )
end

"""
    evaluate_hemispheric_sif_spectrum(layers, emission_model, absorption_matrix, backscatter_matrix, soil_spectrum)

Evaluate canopy hemispheric fluorescence spectra across all `n_emit` bins, where
`absorption_matrix` and `backscatter_matrix` have `(n_emit, n_layers)` shape and
`soil_spectrum` has length `n_emit`.

Return `(; top, ground, generated, layer_absorption, internal, within_leaf, escaped, out_of_window)`
recording the full 5-fate column photon ledger (`top + sum(layer_absorption) + ground ≈ sum(escaped)`,
`sum(internal) + sum(out_of_window) == total_created_photons`).
"""
function evaluate_hemispheric_sif_spectrum(
    layers::AbstractVector,
    emission_model::Union{LeafEmissionMatrix,DualLeafEmissionMatrix,NoLeafEmission},
    absorption_matrix::AbstractMatrix,
    backscatter_matrix::AbstractMatrix,
    soil_spectrum::AbstractVector,
)
    nl = length(layers)
    n_emit = length(soil_spectrum)
    size(absorption_matrix) == (n_emit, nl) &&
        size(backscatter_matrix) == (n_emit, nl) ||
        throw(DimensionMismatch("Rate matrices must have (n_emit, n_layers) shape"))
    is_single = if isempty(layers)
        (eltype(layers) <: LayerHemisphericSource) ||
            (eltype(layers) === Any && !(emission_model isa DualLeafEmissionMatrix))
    else
        all(l -> l isa LayerHemisphericSource, layers)
    end
    is_dual = if isempty(layers)
        (eltype(layers) <: DualLayerHemisphericSource) ||
            (eltype(layers) === Any && (emission_model isa DualLeafEmissionMatrix))
    else
        all(l -> l isa DualLayerHemisphericSource, layers)
    end
    (is_single && emission_model isa Union{LeafEmissionMatrix,NoLeafEmission}) ||
        (is_dual && emission_model isa Union{DualLeafEmissionMatrix,NoLeafEmission}) ||
        throw(ArgumentError("Layer source types and leaf emission model must both be single-photosystem or both be dual-photosystem"))

    FT_fallback = typeof(_model_scalar_probe(emission_model, zero(eltype(soil_spectrum))))
    FT_lay = _layers_scalar_type(evaluate_hemispheric_sif_spectrum, layers, FT_fallback)
    FT = promote_compatible_scalar(
        evaluate_hemispheric_sif_spectrum,
        zero(FT_lay),
        _model_scalar_probe(emission_model, zero(FT_lay)),
        zero(eltype(absorption_matrix)),
        zero(eltype(backscatter_matrix)),
        zero(eltype(soil_spectrum)),
    )
    l0 = firstindex(layers)
    for k in 1:nl
        lay = layers[l0 + k - 1]
        promote_compatible_scalar(evaluate_hemispheric_sif_spectrum, zero(FT), zero(eltype(lay)))
        if is_single
            _validate_layer_model_channels(lay, emission_model)
        else
            _validate_dual_layer_model_channels(lay, emission_model)
        end
    end

    layers_typed = if is_single
        eltype(layers) <: LayerHemisphericSource ? layers :
        LayerHemisphericSource[layers[l0 + k - 1] for k in 1:nl]
    else
        eltype(layers) <: DualLayerHemisphericSource ? layers :
        DualLayerHemisphericSource[layers[l0 + k - 1] for k in 1:nl]
    end

    top = Vector{FT}(undef, n_emit)
    ground = Vector{FT}(undef, n_emit)
    generated = Vector{FT}(undef, n_emit)
    layer_absorption = Matrix{FT}(undef, n_emit, nl)
    internal = Matrix{FT}(undef, n_emit, nl)
    within_leaf = Matrix{FT}(undef, n_emit, nl)
    escaped = Matrix{FT}(undef, n_emit, nl)
    r_ax = collect(axes(absorption_matrix, 1))
    s_ax = collect(axes(soil_spectrum, 1))
    b_ax = if emission_model isa LeafEmissionMatrix
        collect(axes(emission_model.front, 1))
    elseif emission_model isa DualLeafEmissionMatrix &&
           emission_model.psii isa LeafEmissionMatrix
        collect(axes(emission_model.psii.front, 1))
    elseif emission_model isa DualLeafEmissionMatrix &&
           emission_model.psi isa LeafEmissionMatrix
        collect(axes(emission_model.psi.front, 1))
    else
        collect(1:n_emit)
    end
    length(b_ax) == n_emit ||
        throw(DimensionMismatch("Emission model bin count must match length(soil_spectrum)"))
    for m in 1:n_emit
        r_idx = r_ax[m]
        res = evaluate_hemispheric_sif_bin(
            layers_typed,
            emission_model,
            view(absorption_matrix, r_idx, :),
            view(backscatter_matrix, r_idx, :),
            soil_spectrum[s_ax[m]],
            b_ax[m],
        )
        top[m] = res.up[firstindex(res.up)]
        ground[m] = res.ground
        generated[m] = res.generated
        for k in 1:nl
            layer_absorption[m, k] = res.absorption[firstindex(res.absorption) + k - 1]
            internal[m, k] = res.internal[k]
            within_leaf[m, k] = res.within_leaf[k]
            escaped[m, k] = res.escaped[k]
        end
    end
    out_of_window = zeros(FT, nl)
    for k in 1:nl
        lay = layers_typed[firstindex(layers_typed) + k - 1]
        if lay isa LayerHemisphericSource && emission_model isa LeafEmissionMatrix
            acc = zero(FT)
            @inbounds for c in axes(lay.physical_channels, 1)
                acc += convert(FT, emission_model.out_of_window[c]) *
                       convert(FT, lay.physical_channels[c])
            end
            out_of_window[k] = acc
        elseif lay isa DualLayerHemisphericSource &&
               emission_model isa DualLeafEmissionMatrix
            acc = zero(FT)
            if emission_model.psii isa LeafEmissionMatrix
                @inbounds for c in axes(lay.psii.physical_channels, 1)
                    acc += convert(FT, emission_model.psii.out_of_window[c]) *
                           convert(FT, lay.psii.physical_channels[c])
                end
            end
            if emission_model.psi isa LeafEmissionMatrix
                @inbounds for c in axes(lay.psi.physical_channels, 1)
                    acc += convert(FT, emission_model.psi.out_of_window[c]) *
                           convert(FT, lay.psi.physical_channels[c])
                end
            end
            out_of_window[k] = acc
        end
    end
    return (;
        top,
        ground,
        generated,
        layer_absorption,
        internal,
        within_leaf,
        escaped,
        out_of_window,
    )
end

@kernel function _batch_moments_kernel!(
    plus,
    minus,
    law,
    mu,
    weighted,
    o0::Int,
    no::Int,
    d0::Int,
    c0::Int,
    b0::Int,
)
    d_rel, c_rel, b_rel = @index(Global, NTuple)
    FT = eltype(plus)
    d = d0 + d_rel - 1
    c = c0 + c_rel - 1
    b = b0 + b_rel - 1
    p_acc = zero(FT)
    m_acc = zero(FT)
    @inbounds for o_step in 0:(no - 1)
        o = o0 + o_step
        e = leaf_hemispheric_emission(law, mu[o], weighted[o, d, c, b], zero(FT))
        p_acc += convert(FT, e.upward)
        m_acc += convert(FT, e.downward)
    end
    @inbounds begin
        plus[d, c, b] = p_acc
        minus[d, c, b] = m_acc
    end
end

"""
    batch_hemispheric_moments!(plus, minus, law, mu, weighted)

Compute batched `(n_depth × n_channel × n_batch)` hemisphere moments `plus` and
`minus` from `(n_orient × n_depth × n_channel × n_batch)` `weighted` sources on
CPU, `OffsetArray`, or GPU (`CuArray`).
"""
function batch_hemispheric_moments!(
    plus::AbstractArray{FT,3},
    minus::AbstractArray{FT,3},
    law::AbstractEmissionAngularLaw,
    mu::AbstractVector,
    weighted::AbstractArray{<:Real,4},
) where {FT<:Real}
    require_continuous_scalar(FT, batch_hemispheric_moments!, (plus, minus, law, mu, weighted))
    FT_req = promote_compatible_scalar(
        batch_hemispheric_moments!,
        zero(FT),
        _law_scalar_probe(law, zero(FT)),
        zero(eltype(mu)),
        zero(eltype(weighted)),
    )
    FT_req === FT ||
        throw(MethodError(batch_hemispheric_moments!, (plus, minus, law, mu, weighted)))
    axes(weighted, 1) == axes(mu, 1) &&
        axes(plus) == axes(minus) == (axes(weighted, 2), axes(weighted, 3), axes(weighted, 4)) ||
        throw(DimensionMismatch("Batch hemisphere moment dimensions or axes do not align"))
    _disjoint_hemispheric_buffers((plus, minus), (mu, weighted))
    isempty(plus) && return plus, minus
    backend = KernelAbstractions.get_backend(plus)
    k! = _batch_moments_kernel!(backend)
    k!(
        plus,
        minus,
        law,
        mu,
        weighted,
        firstindex(mu, 1),
        length(mu),
        firstindex(plus, 1),
        firstindex(plus, 2),
        firstindex(plus, 3);
        ndrange = size(plus),
    )
    KernelAbstractions.synchronize(backend)
    return plus, minus
end

@kernel function _batch_expand_sources_kernel!(
    up,
    down,
    plus,
    minus,
    front,
    back,
    d0::Int,
    m0::Int,
    c0::Int,
    nc::Int,
)
    d_rel, m_rel = @index(Global, NTuple)
    FT = eltype(up)
    d = d0 + d_rel - 1
    m = m0 + m_rel - 1
    u_acc = zero(FT)
    v_acc = zero(FT)
    @inbounds for c_step in 0:(nc - 1)
        c = c0 + c_step
        fc = convert(FT, front[m, c])
        bc = convert(FT, back[m, c])
        p_dc = convert(FT, plus[d, c])
        m_dc = convert(FT, minus[d, c])
        u_acc += fc * p_dc + bc * m_dc
        v_acc += fc * m_dc + bc * p_dc
    end
    @inbounds begin
        up[d, m] = u_acc
        down[d, m] = v_acc
    end
end

"""
    batch_expand_hemispheric_sources!(up, down, plus, minus, front, back)

Expand `(n_emit × n_channel)` conditional face tables `front` and `back` against
`(n_depth × n_channel)` hemisphere moments `plus` and `minus` into
`(n_depth × n_emit)` sources `up` and `down` on CPU, `OffsetArray`, or GPU
(`CuArray`).
"""
function batch_expand_hemispheric_sources!(
    up::AbstractMatrix{FT},
    down::AbstractMatrix{FT},
    plus::AbstractMatrix,
    minus::AbstractMatrix,
    front::AbstractMatrix,
    back::AbstractMatrix,
) where {FT<:Real}
    require_continuous_scalar(
        FT,
        batch_expand_hemispheric_sources!,
        (up, down, plus, minus, front, back),
    )
    FT_req = promote_compatible_scalar(
        batch_expand_hemispheric_sources!,
        zero(FT),
        zero(eltype(plus)),
        zero(eltype(minus)),
        zero(eltype(front)),
        zero(eltype(back)),
    )
    FT_req === FT ||
        throw(MethodError(batch_expand_hemispheric_sources!, (up, down, plus, minus, front, back)))
    axes(plus) == axes(minus) &&
        axes(front) == axes(back) &&
        axes(plus, 2) == axes(front, 2) &&
        axes(up) == axes(down) == (axes(plus, 1), axes(front, 1)) ||
        throw(DimensionMismatch("Batch hemisphere source expansion dimensions or axes do not align"))
    _disjoint_hemispheric_buffers((up, down), (plus, minus, front, back))
    isempty(up) && return up, down
    backend = KernelAbstractions.get_backend(up)
    k! = _batch_expand_sources_kernel!(backend)
    k!(
        up,
        down,
        plus,
        minus,
        front,
        back,
        firstindex(up, 1),
        firstindex(up, 2),
        firstindex(plus, 2),
        size(plus, 2);
        ndrange = size(up),
    )
    KernelAbstractions.synchronize(backend)
    return up, down
end

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
