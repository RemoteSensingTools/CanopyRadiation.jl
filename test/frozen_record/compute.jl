# Shared machinery for the CanopyRadiation frozen numerical record.
#
# The reference file reference.toml was produced from CanopyRadiation v0.1.0
# (commit 33f015b) by generate.jl. It must NOT be regenerated from refactored
# code: regeneration is a deliberate, reviewed act (see generate.jl).
#
# This file is included by generate.jl, by test/frozen_record.jl and by
# test/run_cuda.jl. All inputs come from inputs.toml; this file contains no
# numeric input values.

module FrozenRecord

using CanopyRadiation
using TOML

export load_fixture, load_reference, compute_record, compare_record, record_value_count

const FIXTURE_PATH = joinpath(@__DIR__, "inputs.toml")
const REFERENCE_PATH = joinpath(@__DIR__, "reference.toml")
const HOST_TOLERANCES_PATH = joinpath(@__DIR__, "host_tolerances.toml")
const METADATA_FIELDS = ("entry_points",)

"""
    RecordEntry

One recorded output: `key` names the code path and case (shared by every entry
point that runs the same path), `entry_point` names the public function that
produced `fields`, and `fields` maps field names to an `FT` scalar or a
`Vector{FT}` (tuples and arrays are stored column-major).
"""
struct RecordEntry
    key::String
    entry_point::String
    fields::Vector{Pair{String,Any}}
end

load_fixture(path::AbstractString = FIXTURE_PATH) = TOML.parsefile(path)
load_reference(path::AbstractString = REFERENCE_PATH) = TOML.parsefile(path)

function precision_type(name::AbstractString)
    name == "Float32" && return Float32
    name == "Float64" && return Float64
    error("unknown precision $(repr(name)) in fixture")
end

# Inputs are decimal Float64 literals in the fixture, rounded once to FT.
function fl(::Type{FT}, x) where {FT}
    x isa Float64 || error("fixture value $(repr(x)) must be a Float64 literal")
    return FT(x)
end
fl(::Type{FT}, xs::AbstractVector) where {FT} = FT[fl(FT, x) for x in xs]

# ---------------------------------------------------------------------------
# Fixture accessors
# ---------------------------------------------------------------------------
function rates_of(::Type{FT}, fx, name::AbstractString) where {FT}
    r = fx["rates"][name]
    return CanopyRates(
        fl(FT, r["absorption"]),
        fl(FT, r["backscatter"]),
        fl(FT, r["beam_up"]),
        fl(FT, r["beam_down"]),
        fl(FT, r["extinction"]),
    )
end

method_names(fx) = sort!(collect(keys(fx["methods"])))

function method_of(::Type{FT}, fx, name::AbstractString) where {FT}
    m = fx["methods"][name]
    m["kind"] == "analytic" && return AnalyticTwoStream()
    m["kind"] == "native" && return NativeTwoStream(fl(FT, m["thin_norm"]), m["source_terms"])
    error("unknown method kind $(repr(m["kind"]))")
end

function layer_case(fx, name::AbstractString)
    for c in fx["layer_cases"]
        c["name"] == name && return c
    end
    error("unknown layer case $(repr(name))")
end

function layer_inputs(::Type{FT}, fx, name::AbstractString) where {FT}
    c = layer_case(fx, name)
    return rates_of(FT, fx, c["rates"]), fl(FT, c["lai"])
end

function emission_of(::Type{FT}, fx, name::AbstractString) where {FT}
    e = fx["emissions"][name]
    e["kind"] == "none" && return NoEmission()
    e["kind"] == "uniform" && return UniformEmission(fl(FT, e["upward"]), fl(FT, e["downward"]))
    e["kind"] == "quadrature" && return QuadratureEmission(
        fl(FT, e["depths"]),
        fl(FT, e["weights"]),
        fl(FT, e["upward"]),
        fl(FT, e["downward"]),
    )
    error("unknown emission kind $(repr(e["kind"]))")
end

function law_of(::Type{FT}, fx, name::AbstractString) where {FT}
    l = fx["hemispheric"]["laws"][name]
    l["kind"] == "lambertian" && return LambertianFaces()
    l["kind"] == "mixed" && return MixedLambertianFaces(fl(FT, l["lambertian_fraction"]))
    error("unknown law kind $(repr(l["kind"]))")
end

function shaped(::Type{FT}, values, shape) where {FT}
    return reshape(fl(FT, values), Tuple(Int.(shape)))
end

function emitting_layer_of(::Type{FT}, fx, layer::AbstractString, emission::AbstractString) where {FT}
    r, lai = layer_inputs(FT, fx, layer)
    return emitting_layer(r.absorption, r.backscatter, lai, emission_of(FT, fx, emission))
end

# ---------------------------------------------------------------------------
# Flattening of returned objects into named fields
# ---------------------------------------------------------------------------
function flatten_fields!(out, ::Type{FT}, prefix::String, x) where {FT}
    if x isa AbstractFloat
        x isa FT || error("field $prefix has type $(typeof(x)), expected $FT")
        push!(out, prefix => x)
    elseif x isa Bool
        push!(out, prefix => x)
    elseif x isa Integer
        push!(out, prefix => Int(x))
    elseif x isa AbstractArray && eltype(x) !== Union{} && eltype(x) <: Bool
        push!(out, prefix => Bool[y for y in x])
    elseif x isa AbstractArray && eltype(x) !== Union{} && eltype(x) <: Integer
        push!(out, prefix => Int[y for y in x])
    elseif x isa Tuple || x isa AbstractArray
        v = FT[]
        for y in x
            y isa FT || error("field $prefix element has type $(typeof(y)), expected $FT")
            push!(v, y)
        end
        push!(out, prefix => v)
    elseif x isa NamedTuple
        for (k, y) in pairs(x)
            flatten_fields!(out, FT, isempty(prefix) ? string(k) : "$prefix.$k", y)
        end
    else
        for k in fieldnames(typeof(x))
            flatten_fields!(out, FT, isempty(prefix) ? string(k) : "$prefix.$k", getfield(x, k))
        end
    end
    return out
end

fields_of(::Type{FT}, x) where {FT} = flatten_fields!(Pair{String,Any}[], FT, "", x)

# ---------------------------------------------------------------------------
# Batch input builders (shared with the CUDA script). `to` converts host
# arrays to the target array type (identity on CPU).
# ---------------------------------------------------------------------------
function passive_column_operators(::Type{FT}, fx, col) where {FT}
    method = method_of(FT, fx, col["method"])
    ops = LayerOperator{FT}[]
    for name in col["layers"]
        r, lai = layer_inputs(FT, fx, name)
        push!(ops, layer_operator(method, r, lai))
    end
    return ops
end

function passive_scenario(::Type{FT}, fx, name::AbstractString) where {FT}
    s = fx["passive_scenarios"][name]
    return fl(FT, s["direct"]), fl(FT, s["diffuse"]), fl(FT, s["soil"])
end

function passive_batch_inputs(::Type{FT}, fx, col) where {FT}
    ops = passive_column_operators(FT, fx, col)
    scen = col["scenarios"]
    nl, nb = length(ops), length(scen)
    layer_matrix(f) = FT[f(ops[l]) for l in 1:nl, _ in 1:nb]
    tops = [passive_scenario(FT, fx, s) for s in scen]
    return (;
        reflection = layer_matrix(op -> op.reflection),
        transmission = layer_matrix(op -> op.transmission),
        beam_up = layer_matrix(op -> op.beam_up),
        beam_down = layer_matrix(op -> op.beam_down),
        beam_trans = layer_matrix(op -> op.beam_transmission),
        direct_top = FT[t[1] for t in tops],
        diffuse_top = FT[t[2] for t in tops],
        soil = FT[t[3] for t in tops],
        n_layers = nl,
        n_batch = nb,
    )
end

function run_passive_batch(::Type{FT}, inp, to = identity, zeros_like = zeros) where {FT}
    nl, nb = inp.n_layers, inp.n_batch
    dir = zeros_like(FT, nl + 1, nb)
    dn = zeros_like(FT, nl + 1, nb)
    up = zeros_like(FT, nl + 1, nb)
    ab = zeros_like(FT, nl, nb)
    gr = zeros_like(FT, nb)
    batch_solve_columns!(
        dir, dn, up, ab, gr,
        to(inp.reflection), to(inp.transmission), to(inp.beam_up), to(inp.beam_down),
        to(inp.beam_trans), to(inp.direct_top), to(inp.diffuse_top), to(inp.soil),
    )
    return Array(dir), Array(dn), Array(up), Array(ab), Array(gr)
end

function emitting_column_layers(::Type{FT}, fx, col) where {FT}
    return EmittingLayer{FT}[
        emitting_layer_of(FT, fx, l["layer"], l["emission"]) for l in col["layers"]
    ]
end

function emitting_scenario(::Type{FT}, fx, name::AbstractString) where {FT}
    s = fx["emitting_scenarios"][name]
    return fl(FT, s["down_top"]), fl(FT, s["soil_reflect"]), fl(FT, s["soil_emit"])
end

function emitting_batch_inputs(::Type{FT}, fx, col) where {FT}
    els = emitting_column_layers(FT, fx, col)
    scen = col["scenarios"]
    nl, nb = length(els), length(scen)
    layer_matrix(f) = FT[f(els[l]) for l in 1:nl, _ in 1:nb]
    tops = [emitting_scenario(FT, fx, s) for s in scen]
    return (;
        reflection = layer_matrix(e -> e.passive.reflection),
        transmission = layer_matrix(e -> e.passive.transmission),
        emit_up = layer_matrix(e -> e.upward),
        emit_down = layer_matrix(e -> e.downward),
        emit_gen = layer_matrix(e -> e.generated),
        down_top = FT[t[1] for t in tops],
        soil_reflect = FT[t[2] for t in tops],
        soil_emit = FT[t[3] for t in tops],
        n_layers = nl,
        n_batch = nb,
    )
end

function run_emitting_batch(::Type{FT}, inp, to = identity, zeros_like = zeros) where {FT}
    nl, nb = inp.n_layers, inp.n_batch
    dn = zeros_like(FT, nl + 1, nb)
    up = zeros_like(FT, nl + 1, nb)
    net = zeros_like(FT, nl, nb)
    ab = zeros_like(FT, nl, nb)
    ns = zeros_like(FT, nb)
    ga = zeros_like(FT, nb)
    gen = zeros_like(FT, nb)
    batch_solve_emitting_columns!(
        dn, up, net, ab, ns, ga, gen,
        to(inp.reflection), to(inp.transmission), to(inp.emit_up), to(inp.emit_down),
        to(inp.emit_gen), to(inp.down_top), to(inp.soil_reflect), to(inp.soil_emit),
    )
    return Array(dn), Array(up), Array(net), Array(ab), Array(ns), Array(ga), Array(gen)
end

"""
    replicate_batch(inp, replicas)

Tile every batch column of a passive or emitting batch input `replicas` times
(column `b` of the original appears at columns `b, b + n_batch, ...`).
"""
function replicate_batch(inp::NamedTuple, replicas::Integer)
    tile(x::AbstractMatrix) = repeat(x, 1, replicas)
    tile(x::AbstractVector) = repeat(x, replicas)
    tile(x) = x
    out = map(tile, inp)
    return merge(out, (; n_batch = inp.n_batch * replicas))
end

function hemispheric_inputs(::Type{FT}, fx) where {FT}
    h = fx["hemispheric"]
    return (;
        mu = fl(FT, h["mu"]),
        weighted = shaped(FT, h["weighted"], h["weighted_shape"]),
        front = shaped(FT, h["front"], h["emission_shape"]),
        back = shaped(FT, h["back"], h["emission_shape"]),
    )
end

function run_batch_moments(::Type{FT}, law, inp, to = identity, zeros_like = zeros) where {FT}
    _, nd, nc, nb = size(inp.weighted)
    p = zeros_like(FT, nd, nc, nb)
    m = zeros_like(FT, nd, nc, nb)
    batch_hemispheric_moments!(p, m, law, to(inp.mu), to(inp.weighted))
    return Array(p), Array(m)
end

function run_batch_expand(::Type{FT}, plus, minus, inp, to = identity, zeros_like = zeros) where {FT}
    nd = size(plus, 1)
    ne = size(inp.front, 1)
    u = zeros_like(FT, nd, ne)
    d = zeros_like(FT, nd, ne)
    batch_expand_hemispheric_sources!(u, d, to(plus), to(minus), to(inp.front), to(inp.back))
    return Array(u), Array(d)
end

# Scalar-path hemispheric moments assembled from hemispheric_moment calls.
function scalar_moments(::Type{FT}, law, mu, w3) where {FT}
    nd, nc = size(w3, 2), size(w3, 3)
    p = zeros(FT, nd, nc)
    m = zeros(FT, nd, nc)
    for c in 1:nc, d in 1:nd
        h = hemispheric_moment(law, mu, w3, d, c)
        p[d, c] = h.plus
        m[d, c] = h.minus
    end
    return p, m
end

# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
"""
    compute_record(FT, fx) -> Vector{RecordEntry}

Run every recorded public entry point on the fixture `fx` in precision `FT`.
Entries that run the same code path on the same case share a `key`.
"""
function compute_record(::Type{FT}, fx) where {FT}
    P = string(FT)
    rec = RecordEntry[]
    add!(key, entry, x) = push!(rec, RecordEntry("$P/$key", entry, fields_of(FT, x)))
    add_fields!(key, entry, fields) = push!(rec, RecordEntry("$P/$key", entry, fields))
    mnames = method_names(fx)

    # Two-stream layer operators and diffuse response
    for case in fx["layer_cases"]
        r, lai = layer_inputs(FT, fx, case["name"])
        for mname in mnames
            add!("layer_operator/$mname/$(case["name"])", "layer_operator",
                layer_operator(method_of(FT, fx, mname), r, lai))
        end
        add!("diffuse_response/$(case["name"])", "diffuse_response",
            diffuse_response(r.absorption, r.backscatter, lai))
    end

    for case in fx["interior_flux_cases"]
        r, lai = layer_inputs(FT, fx, case["layer"])
        for mname in mnames
            add!("interior_flux/$mname/$(case["name"])", "interior_flux",
                interior_flux(
                    method_of(FT, fx, mname), r, lai, fl(FT, case["x"]),
                    fl(FT, case["down_top"]), fl(FT, case["up_bottom"]), fl(FT, case["beam_top"]),
                ))
        end
    end

    # Passive column: tuple, vector (two entry points), batch
    for col in fx["passive_columns"]
        cname = col["name"]
        ops = passive_column_operators(FT, fx, col)
        tup = Tuple(ops)
        for sname in col["scenarios"]
            dir, dif, soil = passive_scenario(FT, fx, sname)
            add!("passive/tuple/$cname/$sname", "solve_column(::Tuple)",
                solve_column(tup, dir, dif, soil))
            add!("passive/vector/$cname/$sname", "solve_column(::Vector)",
                solve_column(ops, dir, dif, soil))
            ws = ColumnFluxWorkspace(FT, length(ops))
            add!("passive/vector/$cname/$sname", "solve_column!(::ColumnFluxWorkspace)",
                solve_column!(ws, ops, dir, dif, soil))
        end
        dir_o, dn_o, up_o, ab_o, gr_o = run_passive_batch(FT, passive_batch_inputs(FT, fx, col))
        for (b, sname) in enumerate(col["scenarios"])
            add!("passive/batch/$cname/$sname", "batch_solve_columns!",
                (; direct = dir_o[:, b], down = dn_o[:, b], up = up_o[:, b],
                    absorption = ab_o[:, b], ground = gr_o[b]))
        end
    end

    # Emitting layers
    for case in fx["emitting_layer_cases"]
        add!("emitting_layer/$(case["layer"])/$(case["emission"])", "emitting_layer",
            emitting_layer_of(FT, fx, case["layer"], case["emission"]))
    end

    # Emitting column: tuple, vector (two entry points), batch; solve_emission
    for col in fx["emitting_columns"]
        cname = col["name"]
        els = emitting_column_layers(FT, fx, col)
        tup = Tuple(els)
        for sname in col["scenarios"]
            dt, sr, se = emitting_scenario(FT, fx, sname)
            add!("emitting/tuple/$cname/$sname", "solve_emitting_column(::Tuple)",
                solve_emitting_column(tup, dt, sr, se))
            add!("emitting/vector/$cname/$sname", "solve_emitting_column(::Vector)",
                solve_emitting_column(els, dt, sr, se))
            ws = EmittingColumnWorkspace(FT, length(els))
            add!("emitting/vector/$cname/$sname", "solve_emitting_column!(::EmittingColumnWorkspace)",
                solve_emitting_column!(ws, els, dt, sr, se))
            add!("emission_only/tuple/$cname/$sname", "solve_emission(::Tuple)",
                solve_emission(tup, sr))
            add!("emission_only/vector/$cname/$sname", "solve_emission(::Vector)",
                solve_emission(els, sr))
        end
        dn_o, up_o, net_o, ab_o, ns_o, ga_o, gen_o =
            run_emitting_batch(FT, emitting_batch_inputs(FT, fx, col))
        for (b, sname) in enumerate(col["scenarios"])
            add!("emitting/batch/$cname/$sname", "batch_solve_emitting_columns!",
                (; down = dn_o[:, b], up = up_o[:, b], net_layer = net_o[:, b],
                    absorption = ab_o[:, b], net_soil = ns_o[b], ground_absorbed = ga_o[b],
                    generated = gen_o[b]))
        end
    end

    # Point sources and interior reconstruction
    for case in fx["point_emission_cases"]
        r, lai = layer_inputs(FT, fx, case["layer"])
        sd, x = fl(FT, case["source_depth"]), fl(FT, case["x"])
        qu, qd = fl(FT, case["upward"]), fl(FT, case["downward"])
        add!("point_emission_response/$(case["name"])", "point_emission_response",
            point_emission_response(r.absorption, r.backscatter, lai, sd, qu, qd))
        add!("point_emission_interior/$(case["name"])", "point_emission_interior",
            point_emission_interior(r.absorption, r.backscatter, lai, sd, x, qu, qd))
    end
    for case in fx["emission_interior_cases"]
        r, lai = layer_inputs(FT, fx, case["layer"])
        add!("emission_interior/$(case["name"])", "emission_interior",
            emission_interior(
                r.absorption, r.backscatter, lai, fl(FT, case["x"]),
                emission_of(FT, fx, case["emission"]),
                fl(FT, case["down_top"]), fl(FT, case["up_bottom"]),
            ))
    end

    # Hemispheric moments and expansion
    hin = hemispheric_inputs(FT, fx)
    w3 = hin.weighted[:, :, :, 1]
    ne = size(hin.front, 1)
    for lname in sort!(collect(keys(fx["hemispheric"]["laws"])))
        law = law_of(FT, fx, lname)
        p, m = scalar_moments(FT, law, hin.mu, w3)
        add!("hemispheric_moment/scalar/$lname", "hemispheric_moment", (; plus = p, minus = m))
        p2, m2 = similar(p), similar(m)
        hemispheric_source_moments!(p2, m2, law, hin.mu, w3)
        add!("hemispheric_moment/scalar/$lname", "hemispheric_source_moments!", (; plus = p2, minus = m2))
        pb, mb = run_batch_moments(FT, law, hin)
        add!("hemispheric_moment/batch/$lname", "batch_hemispheric_moments!", (; plus = pb, minus = mb))

        nd = size(p, 1)
        u, d = zeros(FT, nd, ne), zeros(FT, nd, ne)
        for e in 1:ne, k in 1:nd
            h = hemispheric_bin(p, m, hin.front[e, :], hin.back[e, :], k)
            u[k, e] = h.up
            d[k, e] = h.down
        end
        add!("hemispheric_bin/scalar/$lname", "hemispheric_bin", (; up = u, down = d))
        u2, d2 = similar(u), similar(d)
        for e in 1:ne
            ue, de = zeros(FT, nd), zeros(FT, nd)
            expand_hemispheric_source!(ue, de, p, m, hin.front[e, :], hin.back[e, :])
            u2[:, e] = ue
            d2[:, e] = de
        end
        add!("hemispheric_bin/scalar/$lname", "expand_hemispheric_source!", (; up = u2, down = d2))
        ub, db = run_batch_expand(FT, p, m, hin)
        add!("hemispheric_bin/batch/$lname", "batch_expand_hemispheric_sources!", (; up = ub, down = db))
    end
    return rec
end

# ---------------------------------------------------------------------------
# Bit strings and comparison
# ---------------------------------------------------------------------------
bitstring_hex(x::Float64) = "0x" * string(reinterpret(UInt64, x); base = 16, pad = 16)
bitstring_hex(x::Float32) = "0x" * string(reinterpret(UInt32, x); base = 16, pad = 8)

function from_hex(::Type{FT}, s::AbstractString) where {FT}
    U = FT === Float64 ? UInt64 : FT === Float32 ? UInt32 : error("unsupported $FT")
    (startswith(s, "0x") && length(s) == 2 + 2 * sizeof(U)) ||
        error("malformed bit string $(repr(s)) for $FT")
    return reinterpret(FT, parse(U, s[3:end]; base = 16))
end

as_vector(x::AbstractFloat) = [x]
as_vector(x::AbstractVector) = x
as_vector(x::Integer) = [x]
is_exact_field(v) = v isa Integer || (v isa AbstractVector && eltype(v) <: Integer)

"""Distance between `a` and reference `b` in units of `eps(b)` (the spacing at the reference value)."""
ulps_of_value(a, b) = a == b ? 0.0 : Float64(abs(a - b)) / Float64(eps(abs(b)))

"""
    compare_record(rec, ref; tolerances = Dict()) -> (mismatches, missing, unmatched, checked, tolerated)

Compare computed entries `rec` bit-for-bit with the parsed reference `ref`.
`mismatches` lists human-readable descriptions; `missing` lists computed keys
absent from the reference; `unmatched` lists reference keys never computed
(only for precisions present in `rec`); `checked` counts compared values.
`tolerances` maps a reference key to a tolerance in units of `eps` of the
reference value; differences within it are listed in `tolerated` instead of
`mismatches`. It is empty unless the caller opts in (see host_tolerances.toml).
"""
function compare_record(rec::Vector{RecordEntry}, ref::AbstractDict; tolerances::AbstractDict = Dict{String,Float64}())
    mismatches = String[]
    tolerated = String[]
    missing_keys = String[]
    seen = Set{String}()
    entries_seen = Dict{String,Set{String}}()
    checked = 0
    for e in rec
        push!(seen, e.key)
        if !haskey(ref, e.key)
            push!(missing_keys, e.key)
            continue
        end
        expected = ref[e.key]
        entry_points = get(expected, "entry_points", String[])
        e.entry_point in entry_points ||
            push!(mismatches, "$(e.key): entry point $(e.entry_point) not listed in reference")
        push!(get!(entries_seen, e.key, Set{String}()), e.entry_point)
        names_got = Set(first.(e.fields))
        for k in keys(expected)
            k in METADATA_FIELDS && continue
            k in names_got || push!(mismatches, "$(e.key) [$(e.entry_point)]: field $k not returned")
        end
        for (fname, val) in e.fields
            if fname in METADATA_FIELDS || !haskey(expected, fname)
                push!(mismatches, "$(e.key) [$(e.entry_point)]: field $fname not in reference")
                continue
            end
            exp_raw = expected[fname]
            if is_exact_field(val)
                # Integer and Bool fields (indices, flags) are compared exactly.
                checked += length(as_vector(val))
                isequal(exp_raw, val) ||
                    push!(mismatches, "$(e.key) [$(e.entry_point)]: $fname reference $(repr(exp_raw)), got $(repr(val))")
                continue
            end
            FT = val isa AbstractVector ? eltype(val) : typeof(val)
            exp_vals = FT[from_hex(FT, s) for s in (exp_raw isa AbstractString ? [exp_raw] : exp_raw)]
            got_vals = as_vector(val)
            if (exp_raw isa AbstractString) != (val isa AbstractFloat) || length(exp_vals) != length(got_vals)
                push!(mismatches, "$(e.key) [$(e.entry_point)]: field $fname shape differs " *
                                  "(reference $(length(exp_vals)), got $(length(got_vals)))")
                continue
            end
            for i in eachindex(exp_vals, got_vals)
                checked += 1
                g, x = got_vals[i], exp_vals[i]
                if bitstring_hex(g) != bitstring_hex(x)
                    msg = "$(e.key) [$(e.entry_point)]: $fname[$i] reference $(repr(x)) " *
                          "($(bitstring_hex(x))), got $(repr(g)) ($(bitstring_hex(g))), " *
                          "$(ulps_of_value(g, x)) eps of reference value"
                    tol = get(tolerances, e.key, nothing)
                    if tol !== nothing && isfinite(g) && within_tolerance(tol, g, x, expected)
                        push!(tolerated, msg)
                    else
                        push!(mismatches, msg)
                    end
                end
            end
        end
    end
    for (k, eps_seen) in entries_seen
        for ep in ref[k]["entry_points"]
            ep in eps_seen || push!(mismatches, "$k: reference entry point $ep was not recomputed")
        end
    end
    precisions = Set(first(split(e.key, '/')) for e in rec)
    unmatched = sort!([k for k in keys(ref)
                       if k != "meta" && first(split(k, '/')) in precisions && !(k in seen)])
    return mismatches, missing_keys, unmatched, checked, tolerated
end

"""
    host_tolerances(path = HOST_TOLERANCES_PATH) -> Dict{String,Float64}

Per-key tolerances (in eps of the reference value) that apply on the current
host. Sections whose names start with `no_native_fma` apply only when
`Core.Intrinsics.have_fma(Float32)` is false. On hosts with native FMA the
result is empty and every value must match bit for bit.
"""
function host_tolerances(path::AbstractString = HOST_TOLERANCES_PATH)
    tol = Dict{String,Any}()
    t = TOML.parsefile(path)
    if !Core.Intrinsics.have_fma(Float32)
        for (name, sec) in t
            startswith(name, "no_native_fma") || continue
            mode = get(sec, "tolerance_mode", "value")
            for k in sec["keys"]
                if mode == "value"
                    tol[k] = max(get(tol, k, 0.0), Float64(sec["tolerance_eps"]))
                elseif mode == "table_scale"
                    haskey(tol, k) && error("key $k listed in more than one tolerance section")
                    tol[k] = (; eps = Float64(sec["tolerance_eps"]), mode = :table_scale)
                else
                    error("unknown tolerance_mode $(repr(mode)) in [$name]")
                end
            end
        end
    end
    return tol
end

"""
    within_tolerance(tol, got, reference, table)

`tol::Real`: |got - reference| <= tol * eps(reference) (eps of the value).
`tol` with `mode = :table_scale`: |got - reference| <= tol.eps * eps(FT) * scale,
where scale is the largest magnitude among the float values of the reference
`table` (for cancellation residues such as the absorption of a non-absorbing
layer, whose own eps is meaningless).
"""
within_tolerance(tol::Real, g, x, table) = ulps_of_value(g, x) <= tol
function within_tolerance(tol::NamedTuple, g::FT, x::FT, table) where {FT}
    tol.mode === :table_scale || error("unknown tolerance mode $(tol.mode)")
    scale = zero(FT)
    for (f, v) in table
        f in METADATA_FIELDS && continue
        for s in (v isa AbstractString ? (v,) : v)
            s isa AbstractString && (scale = max(scale, abs(from_hex(FT, s))))
        end
    end
    return Float64(abs(g - x)) <= tol.eps * Float64(eps(FT)) * Float64(scale)
end

"""Number of distinct recorded values (one per key, not per entry point)."""
function record_value_count(rec::Vector{RecordEntry})
    seen = Set{String}()
    n = 0
    for e in rec
        e.key in seen && continue
        push!(seen, e.key)
        n += sum(length(as_vector(v)) for (_, v) in e.fields; init = 0)
    end
    return n
end

include("compute_api.jl")
include("compute_regimes.jl")

end # module FrozenRecord
