# Frozen numerical record, extension 2: numerical regimes named by the
# refactor reviews (low sun in deep columns, darkness, soil edges, 4SAIL
# chain, layer-edge interior reconstruction, differing empty-column paths,
# asymmetric photosystems, photon conversion rounding).
#
# Included inside module FrozenRecord (compute.jl). reference_regimes.toml was
# produced from the unmodified CanopyRadiation v0.1.0 source (commit 33f015b)
# by generate.jl --part=regimes and must NOT be regenerated from refactored
# code; regeneration is a deliberate, reviewed act. All inputs come from the
# regime_* sections of inputs.toml (plus existing sections they reference);
# this file contains no numeric input values.

const REFERENCE_REGIMES_PATH = joinpath(@__DIR__, "reference_regimes.toml")

function regime_rates(::Type{FT}, fx, name) where {FT}
    r = fx["regime_rates"][name]
    return CanopyRates(fl(FT, r["absorption"]), fl(FT, r["backscatter"]), fl(FT, r["beam_up"]),
        fl(FT, r["beam_down"]), fl(FT, r["extinction"]))
end

function regime_passive_scenario(::Type{FT}, fx, name) where {FT}
    s = fx["regime_passive_scenarios"][name]
    return fl(FT, s["direct"]), fl(FT, s["diffuse"]), fl(FT, s["soil"])
end

function regime_emitting_scenario(::Type{FT}, fx, name) where {FT}
    s = fx["regime_emitting_scenarios"][name]
    return fl(FT, s["down_top"]), fl(FT, s["soil_reflect"]), fl(FT, s["soil_emit"])
end

"""Left-to-right sum, independent of Base's pairwise/SIMD summation."""
leftsum(::Type{FT}, xs) where {FT} = foldl(+, xs; init = zero(FT))

function passive_batch_from(::Type{FT}, ops::Vector{LayerOperator{FT}}, tops) where {FT}
    nl, nb = length(ops), length(tops)
    layer_matrix(f) = FT[f(ops[l]) for l in 1:nl, _ in 1:nb]
    return (;
        reflection = layer_matrix(op -> op.reflection),
        transmission = layer_matrix(op -> op.transmission),
        beam_up = layer_matrix(op -> op.beam_up),
        beam_down = layer_matrix(op -> op.beam_down),
        beam_trans = layer_matrix(op -> op.beam_transmission),
        direct_top = FT[t[1] for t in tops], diffuse_top = FT[t[2] for t in tops], soil = FT[t[3] for t in tops],
        n_layers = nl, n_batch = nb)
end

function emitting_batch_from(::Type{FT}, els::Vector{EmittingLayer{FT}}, tops) where {FT}
    nl, nb = length(els), length(tops)
    layer_matrix(f) = FT[f(els[l]) for l in 1:nl, _ in 1:nb]
    return (;
        reflection = layer_matrix(e -> e.passive.reflection),
        transmission = layer_matrix(e -> e.passive.transmission),
        emit_up = layer_matrix(e -> e.upward), emit_down = layer_matrix(e -> e.downward),
        emit_gen = layer_matrix(e -> e.generated),
        down_top = FT[t[1] for t in tops], soil_reflect = FT[t[2] for t in tops], soil_emit = FT[t[3] for t in tops],
        n_layers = nl, n_batch = nb)
end

"""Selected interfaces and layers of a deep passive solution (see inputs.toml)."""
function deep_selection(::Type{FT}, direct, down, up, absorption, ground, n, tp) where {FT}
    idx = (1, tp, tp + 1, n + 1)
    return (;
        direct = FT[direct[i] for i in idx], down = FT[down[i] for i in idx], up = FT[up[i] for i in idx],
        absorption_thick = absorption[tp], absorption_sum = leftsum(FT, absorption), ground)
end

function manufactured_geometry(::Type{FT}, fx) where {FT}
    gx = fx["api_foursail_geometry"]
    return CanopyOptics.FourSAILGeometry((fl(FT, gx[k]) for k in
        ("mu_s", "mu_o", "dphi", "dso", "ks", "ko", "sob", "sof", "sdb", "sdf", "dob", "dof", "ddb", "ddf"))...)
end

function regime_photosystems(::Type{FT}, fx) where {FT}
    p = fx["regime_photosystems"]
    m(prefix) = LeafEmissionMatrix(
        fmat(FT, p["$(prefix)_front"], p["$(prefix)_shape"]), fmat(FT, p["$(prefix)_back"], p["$(prefix)_shape"]),
        fmat(FT, p["$(prefix)_reabsorbed"], p["$(prefix)_shape"]), fl(FT, p["$(prefix)_out_of_window"]))
    psii, psi_eq, psi_un = m("psii"), m("psi_equal"), m("psi_unequal")
    n_pop = length(p["internal_psi_unequal"])
    return (; psii, psi_equal = psi_eq, psi_unequal = psi_un,
        internal_psii = fmat(FT, p["internal_psii"], (size(psii.front, 2), n_pop)),
        internal_psi_equal = fmat(FT, p["internal_psi_equal"], (size(psi_eq.front, 2), n_pop)),
        internal_psi_unequal = fmat(FT, p["internal_psi_unequal"], (size(psi_un.front, 2), n_pop)))
end

"""
    compute_regime_record(FT, fx) -> Vector{RecordEntry}

Record the regime cases of inputs.toml (sections prefixed `regime_`).
"""
function compute_regime_record(::Type{FT}, fx) where {FT}
    P = string(FT)
    rec = RecordEntry[]
    add!(key, entry, x) = push!(rec, RecordEntry("$P/regimes/$key", entry, fields_of(FT, x)))
    rl = fx["regime_layers"]
    thin, thick = fl(FT, rl["thin_lai"]), fl(FT, rl["thick_lai"])

    # ---- low sun: layer operators of the thick layer at each extinction
    for rname in sort!(collect(keys(fx["regime_rates"]))), mname in method_names(fx)
        add!("layer_operator/$mname/$(rname)_thick", "layer_operator",
            layer_operator(method_of(FT, fx, mname), regime_rates(FT, fx, rname), thick))
    end

    # ---- low sun: deep passive columns on all three paths
    for col in fx["regime_deep_columns"]
        rates = regime_rates(FT, fx, col["rates"])
        n, tp = Int(col["n_layers"]), Int(col["thick_position"])
        tops = [regime_passive_scenario(FT, fx, s) for s in col["scenarios"]]
        for mname in col["methods"]
            method = method_of(FT, fx, mname)
            op_thin, op_thick = layer_operator(method, rates, thin), layer_operator(method, rates, thick)
            ops = LayerOperator{FT}[l == tp ? op_thick : op_thin for l in 1:n]
            tup = Tuple(ops)
            base = "deep_column/$(col["name"])/$mname"
            for (sname, (d, f, s)) in zip(col["scenarios"], tops)
                t = solve_column(tup, d, f, s)
                add!("$base/tuple/$sname", "solve_column(::Tuple)",
                    deep_selection(FT, t.direct, t.down, t.up, t.absorption, t.ground, n, tp))
                v = solve_column(ops, d, f, s)
                add!("$base/vector/$sname", "solve_column(::Vector)",
                    deep_selection(FT, v.direct, v.down, v.up, v.absorption, v.ground, n, tp))
                w = solve_column!(ColumnFluxWorkspace(FT, n), ops, d, f, s)
                add!("$base/vector/$sname", "solve_column!(::ColumnFluxWorkspace)",
                    deep_selection(FT, w.direct, w.down, w.up, w.absorption, w.ground, n, tp))
            end
            dir, dn, up, ab, gr = run_passive_batch(FT, passive_batch_from(FT, ops, tops))
            for (b, sname) in enumerate(col["scenarios"])
                add!("$base/batch/$sname", "batch_solve_columns!",
                    deep_selection(FT, dir[:, b], dn[:, b], up[:, b], ab[:, b], gr[b], n, tp))
            end
        end
    end

    # ---- 4SAIL chain into one-layer columns with soil exactly 0 and 1
    ch = fx["regime_foursail_chain"]
    leaf = only(c for c in fx["api_foursail_cases"] if c["name"] == ch["leaf"])
    rates = foursail_canopy_rates(manufactured_geometry(FT, fx), fl(FT, leaf["reflectance"]), fl(FT, leaf["transmittance"]))
    tops = [regime_passive_scenario(FT, fx, s) for s in ch["scenarios"]]
    for mname in method_names(fx)
        ops = LayerOperator{FT}[layer_operator(method_of(FT, fx, mname), rates, fl(FT, ch["lai"]))]
        for (sname, (d, f, s)) in zip(ch["scenarios"], tops)
            add!("foursail_chain/$mname/tuple/$sname", "solve_column(::Tuple)", solve_column(Tuple(ops), d, f, s))
            add!("foursail_chain/$mname/vector/$sname", "solve_column(::Vector)", solve_column(ops, d, f, s))
            add!("foursail_chain/$mname/vector/$sname", "solve_column!(::ColumnFluxWorkspace)",
                solve_column!(ColumnFluxWorkspace(FT, 1), ops, d, f, s))
        end
        dir, dn, up, ab, gr = run_passive_batch(FT, passive_batch_from(FT, ops, tops))
        for (b, sname) in enumerate(ch["scenarios"])
            add!("foursail_chain/$mname/batch/$sname", "batch_solve_columns!",
                (; direct = dir[:, b], down = dn[:, b], up = up[:, b], absorption = ab[:, b], ground = gr[b]))
        end
    end

    # ---- empty passive column where the tuple and vector paths differ
    empty_tops = [(fl(FT, c["direct"]), fl(FT, c["diffuse"]), fl(FT, c["soil"])) for c in fx["regime_empty_cases"]]
    for (c, (d, f, s)) in zip(fx["regime_empty_cases"], empty_tops)
        add!("empty_column/tuple/$(c["name"])", "solve_column(::Tuple{})", solve_column((), d, f, s))
        add!("empty_column/vector/$(c["name"])", "solve_column(::Vector)", solve_column(LayerOperator{FT}[], d, f, s))
        add!("empty_column/vector/$(c["name"])", "solve_column!(::ColumnFluxWorkspace)",
            solve_column!(ColumnFluxWorkspace(FT, 0), LayerOperator{FT}[], d, f, s))
    end
    dir, dn, up, ab, gr = run_passive_batch(FT, passive_batch_from(FT, LayerOperator{FT}[], empty_tops))
    for (b, c) in enumerate(fx["regime_empty_cases"])
        add!("empty_column/batch/$(c["name"])", "batch_solve_columns!",
            (; direct = dir[:, b], down = dn[:, b], up = up[:, b], absorption = ab[:, b], ground = gr[b]))
    end

    # ---- emitting column: zero sources, soil_reflect exactly 0 and 1
    for col in fx["regime_emitting_columns"]
        els = EmittingLayer{FT}[emitting_layer_of(FT, fx, l["layer"], l["emission"]) for l in col["layers"]]
        tops = [regime_emitting_scenario(FT, fx, s) for s in col["scenarios"]]
        base = "emitting_column/$(col["name"])"
        for (sname, (dt, sr, se)) in zip(col["scenarios"], tops)
            add!("$base/tuple/$sname", "solve_emitting_column(::Tuple)", solve_emitting_column(Tuple(els), dt, sr, se))
            add!("$base/vector/$sname", "solve_emitting_column(::Vector)", solve_emitting_column(els, dt, sr, se))
            add!("$base/vector/$sname", "solve_emitting_column!(::EmittingColumnWorkspace)",
                solve_emitting_column!(EmittingColumnWorkspace(FT, length(els)), els, dt, sr, se))
        end
        dn, up, net, ab, ns, ga, gen = run_emitting_batch(FT, emitting_batch_from(FT, els, tops))
        for (b, sname) in enumerate(col["scenarios"])
            add!("$base/batch/$sname", "batch_solve_emitting_columns!",
                (; down = dn[:, b], up = up[:, b], net_layer = net[:, b], absorption = ab[:, b],
                    net_soil = ns[b], ground_absorbed = ga[b], generated = gen[b]))
        end
    end

    # ---- interior reconstruction at the layer edges x = 0 and x = lai
    for c in fx["regime_interior_edge_cases"]
        r, lai = layer_inputs(FT, fx, c["layer"])
        for mname in method_names(fx), (edge, x) in (("x0", zero(FT)), ("xlai", lai))
            add!("interior_flux_edge/$mname/$(c["name"])/$edge", "interior_flux",
                interior_flux(method_of(FT, fx, mname), r, lai, x, fl(FT, c["down_top"]), fl(FT, c["up_bottom"]),
                    fl(FT, c["beam_top"])))
        end
    end
    for c in fx["regime_point_edge_cases"]
        r, lai = layer_inputs(FT, fx, c["layer"])
        pos(where) = where == "top" ? zero(FT) : where == "bottom" ? lai : error("unknown edge $(repr(where))")
        add!("point_emission_interior_edge/$(c["name"])", "point_emission_interior",
            point_emission_interior(r.absorption, r.backscatter, lai, pos(c["source_at"]), pos(c["observer_at"]),
                fl(FT, c["upward"]), fl(FT, c["downward"])))
    end
    ee = fx["regime_emission_edge"]
    r, lai = layer_inputs(FT, fx, ee["layer"])
    for ename in sort!(collect(keys(fx["emissions"]))), (edge, x) in (("x0", zero(FT)), ("xlai", lai))
        add!("emission_interior_edge/$ename/$edge", "emission_interior",
            emission_interior(r.absorption, r.backscatter, lai, x, emission_of(FT, fx, ename),
                fl(FT, ee["down_top"]), fl(FT, ee["up_bottom"])))
    end

    # ---- asymmetric photosystems: dual escapes
    ps = regime_photosystems(FT, fx)
    n_emit, n_pop = size(ps.psii.front, 1), size(ps.internal_psii, 2)
    escape_cases = (
        ("equal/psii_psi", DualLeafEmissionMatrix(ps.psii, ps.psi_equal), ps.internal_psi_equal),
        ("equal/psii_none", DualLeafEmissionMatrix(ps.psii, NoLeafEmission()), ps.internal_psi_equal),
        ("equal/none_psi", DualLeafEmissionMatrix(NoLeafEmission(), ps.psi_equal), ps.internal_psi_equal),
        ("unequal/psii_psi", DualLeafEmissionMatrix(ps.psii, ps.psi_unequal), ps.internal_psi_unequal),
    )
    for (name, d, internal_psi) in escape_cases
        f, b, rr, o = escape_outputs(FT, n_emit, n_pop)
        add!("escape_dual_leaf_emission/$name", "escape_dual_leaf_emission!",
            escape_dual_leaf_emission!(f, b, rr, d, ps.internal_psii, internal_psi; out_of_window = o))
    end

    # ---- SIF: zero-LAI layer, zero absorption, PSII 2 channels / PSI 1 channel
    sx = fx["regime_sif"]
    lai_v = fl(FT, sx["lai"])
    depths, weights = fl(FT, sx["fractional_depths"]), fl(FT, sx["weights"])
    mk(ps_name, k, nch) = LayerHemisphericSource(lai_v[k], copy(depths), copy(weights),
        fmat(FT, sx["$(ps_name)_plus"][k], (length(depths), nch)),
        fmat(FT, sx["$(ps_name)_minus"][k], (length(depths), nch)),
        fl(FT, sx["$(ps_name)_physical_channels"][k]))
    n2, n1 = size(ps.psii.front, 2), size(ps.psi_unequal.front, 2)
    layers = [DualLayerHemisphericSource(mk("psii", k, n2), mk("psi", k, n1)) for k in eachindex(lai_v)]
    dual = DualLeafEmissionMatrix(ps.psii, ps.psi_unequal)
    for bin in Int.(sx["bins"])
        add!("evaluate_hemispheric_sif_bin/dual_unequal/bin$bin", "evaluate_hemispheric_sif_bin(dual)",
            evaluate_hemispheric_sif_bin(layers, dual, fl(FT, sx["absorption_rates"]), fl(FT, sx["backscatter_rates"]),
                fl(FT, sx["soil_reflect"]), bin))
    end
    add!("evaluate_hemispheric_sif_spectrum/dual_unequal", "evaluate_hemispheric_sif_spectrum",
        evaluate_hemispheric_sif_spectrum(layers, dual, fmat(FT, sx["absorption_matrix"], sx["rates_shape"]),
            fmat(FT, sx["backscatter_matrix"], sx["rates_shape"]), fl(FT, sx["soil_spectrum"])))

    # ---- photon conversions with non-benign rounding
    ph = fx["regime_photon"]
    dens = fmat(FT, ph["density"], ph["density_shape"])
    widths, wl = fl(FT, ph["widths_nm"]), fl(FT, ph["wavelengths_nm"])
    bins = photon_bins!(similar(dens), dens, widths)
    add!("photon_bins", "photon_bins!", (; output = bins))
    add!("photon_density", "photon_density!", (; output = photon_density!(similar(dens), bins, widths)))
    add!("photon_energy", "photon_energy!",
        (; output = photon_energy!(similar(dens), bins, wl, fl(FT, ph["energy_coefficient"]))))
    return rec
end
