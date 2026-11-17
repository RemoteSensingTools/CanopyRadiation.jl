# Frozen numerical record, extension: remaining public API.
#
# Included inside module FrozenRecord (compute.jl). The reference files
# reference_api.toml and reference_canopyoptics.toml were produced from the
# unmodified CanopyRadiation v0.1.0 source (commit 33f015b) by generate.jl and
# must NOT be regenerated from refactored code; regeneration is a deliberate,
# reviewed act. All inputs come from inputs.toml (sections prefixed `api_` and
# `canopyoptics_`); this file contains no numeric input values.

using CanopyOptics: CanopyOptics
using OffsetArrays: OffsetArray

const REFERENCE_API_PATH = joinpath(@__DIR__, "reference_api.toml")
const REFERENCE_CANOPYOPTICS_PATH = joinpath(@__DIR__, "reference_canopyoptics.toml")

fx_shape(shape) = Tuple(Int.(shape))

# population_coordinates fields that are recorded: the index mapping and flags
# plus the one computed coordinate. The remaining fields (layer, depth_lai,
# depth_weight, cos_inclination, azimuth) are input elements selected by the
# recorded indices.
const POPULATION_FIELDS = (:depth_index, :orientation_index, :sunlit, :mirrored_azimuth, :column_depth_lai)

# prepare_anchor_basis fields that are copies of its inputs (not recorded);
# `labels` is recorded as three integer vectors instead.
const ANCHOR_BASIS_INPUT_COPIES = (:labels, :forcing, :eta, :anchors, :native)
fmat(::Type{FT}, values, shape) where {FT} = reshape(fl(FT, values), fx_shape(shape))

function api_law_list(::Type{FT}, fx) where {FT}
    names = sort!(collect(keys(fx["hemispheric"]["laws"])))
    return [(n, law_of(FT, fx, n)) for n in names]
end

layout_of(name) = name == "paired" ? PairedIllumination() :
                  name == "blocked" ? BlockedIllumination() : error("unknown layout $(repr(name))")

function population_inputs(::Type{FT}, fx, layout_name) where {FT}
    p = fx["api_population"]
    g = PopulationQuadrature(
        p["layer"], fl(FT, p["lai"]), fl(FT, p["lai_above"]),
        fl(FT, p["depths"]), fl(FT, p["depth_weights"]),
        fl(FT, p["cos_inclinations"]), fl(FT, p["azimuths"]),
        layout_of(layout_name),
    )
    return (; g, weights = fl(FT, p["weights"]), front = fl(FT, p["front"]), back = fl(FT, p["back"]),
        ray_mu = fl(FT, p["ray_mu"]), ray_phi = fl(FT, p["ray_phi"]))
end

function leaf_emission_models(::Type{FT}, fx) where {FT}
    e = fx["api_leaf_emission"]
    s = e["shape"]
    m(prefix, oow_key) = LeafEmissionMatrix(
        fmat(FT, e["$(prefix)_front"], s), fmat(FT, e["$(prefix)_back"], s),
        fmat(FT, e["$(prefix)_reabsorbed"], s), fl(FT, e[oow_key]))
    return (; psii = m("psii", "psii_out_of_window"), psi = m("psi", "psi_out_of_window"),
        bad = m("psii", "bad_out_of_window"),
        three_arg = LeafEmissionMatrix(fmat(FT, e["psii_front"], s), fmat(FT, e["psii_back"], s),
            fmat(FT, e["psii_reabsorbed"], s)),
        internal_psii = fmat(FT, e["internal_psii"], e["internal_shape"]),
        internal_psi = fmat(FT, e["internal_psi"], e["internal_shape"]))
end

dual_combinations(models) = (
    ("psii_psi", DualLeafEmissionMatrix(models.psii, models.psi)),
    ("psii_none", DualLeafEmissionMatrix(models.psii, NoLeafEmission())),
    ("none_psi", DualLeafEmissionMatrix(NoLeafEmission(), models.psi)),
    ("none_none", DualLeafEmissionMatrix(NoLeafEmission(), NoLeafEmission())),
)

function escape_outputs(::Type{FT}, n_emit, n_pop) where {FT}
    return zeros(FT, n_emit, n_pop), zeros(FT, n_emit, n_pop), zeros(FT, n_emit, n_pop), zeros(FT, n_pop)
end

function anchor_inputs(::Type{FT}, fx) where {FT}
    a = fx["api_anchor_basis"]
    s = a["model_shape"]
    front, back, reab = fmat(FT, a["front"], s), fmat(FT, a["back"], s), fmat(FT, a["reabsorbed"], s)
    return (;
        map = LinearSpectralMap(fl(FT, a["anchors"]), fl(FT, a["native"])),
        incident = fmat(FT, a["incident_bins"], a["incident_shape"]),
        eta = fl(FT, a["eta"]),
        model = LeafEmissionMatrix(front, back, reab, fl(FT, a["out_of_window"])),
        internal_probability = front .+ back .+ reab,
        energy = fl(FT, a["excitation_energy"]),
        response_front = fmat(FT, a["response_front"], a["response_shape"]),
        response_back = fmat(FT, a["response_back"], a["response_shape"]),
        subset = Int.(a["population_subset"]),
    )
end

function sif_layers(::Type{FT}, fx) where {FT}
    s = fx["api_sif"]
    lai = fl(FT, s["lai"])
    depths, weights = fl(FT, s["fractional_depths"]), fl(FT, s["weights"])
    ms = s["moment_shape"]
    mk(ps, k) = LayerHemisphericSource(
        lai[k], copy(depths), copy(weights),
        fmat(FT, s["$(ps)_plus"][k], ms), fmat(FT, s["$(ps)_minus"][k], ms),
        fl(FT, s["$(ps)_physical_channels"][k]))
    psii = [mk("psii", k) for k in eachindex(lai)]
    psi = [mk("psi", k) for k in eachindex(lai)]
    dual = [DualLayerHemisphericSource(psii[k], psi[k]) for k in eachindex(lai)]
    return (; psii, psi, dual,
        absorption = fl(FT, s["absorption_rates"]), backscatter = fl(FT, s["backscatter_rates"]),
        soil = fl(FT, s["soil_reflect"]),
        absorption_matrix = fmat(FT, s["absorption_matrix"], s["rates_shape"]),
        backscatter_matrix = fmat(FT, s["backscatter_matrix"], s["rates_shape"]),
        soil_spectrum = fl(FT, s["soil_spectrum"]))
end

"""Run `f`; return `true` if it returns normally and `false` if it throws an ArgumentError."""
function accepted(f)
    try
        f()
        return true
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
end

"""
    compute_api_record(FT, fx) -> Vector{RecordEntry}

Record every remaining public entry point that does not depend on CanopyOptics
computations (inputs are manufactured in the fixture).
"""
function compute_api_record(::Type{FT}, fx) where {FT}
    P = string(FT)
    rec = RecordEntry[]
    add!(key, entry, x) = push!(rec, RecordEntry("$P/$key", entry, fields_of(FT, x)))
    laws = api_law_list(FT, fx)

    # ---- face optics
    for c in fx["api_source_projection_cases"]
        add!("source_projection/$(c["name"])", "source_projection",
            (; value = source_projection(fl(FT, c["leaf_cosine"]), fl(FT, c["leaf_azimuth"]),
                fl(FT, c["source_cosine"]), fl(FT, c["source_azimuth"]))))
    end
    for c in fx["api_beam_face_cases"]
        add!("beam_face_absorption/$(c["name"])", "beam_face_absorption",
            beam_face_absorption(fl(FT, c["projection"]), fl(FT, c["beam_normal"]), fl(FT, c["absorptance"])))
    end
    for c in fx["api_isotropic_face_cases"]
        add!("isotropic_face_absorption/$(c["name"])", "isotropic_face_absorption",
            isotropic_face_absorption(fl(FT, c["leaf_cosine"]), fl(FT, c["down"]), fl(FT, c["up"]),
                fl(FT, c["absorptance"])))
    end
    for c in fx["api_plate_cases"]
        plate = (; traversal_transmission = fl(FT, c["traversal_transmission"]),
            traversal_absorptance = fl(FT, c["traversal_absorptance"]),
            interface_entry_transmission = fl(FT, c["interface_entry_transmission"]),
            interface_exit_transmission = fl(FT, c["interface_exit_transmission"]))
        add!("regular_plate_passive/$(c["name"])", "regular_plate_passive", regular_plate_passive(plate))
    end
    gx = fx["api_foursail_geometry"]
    geom = CanopyOptics.FourSAILGeometry((fl(FT, gx[k]) for k in
        ("mu_s", "mu_o", "dphi", "dso", "ks", "ko", "sob", "sof", "sdb", "sdf", "dob", "dof", "ddb", "ddf"))...)
    for c in fx["api_foursail_cases"]
        r, t = fl(FT, c["reflectance"]), fl(FT, c["transmittance"])
        if haskey(c, "clumping")
            add!("foursail_canopy_rates/clumped/$(c["name"])", "foursail_canopy_rates(geom, r, t, clumping)",
                foursail_canopy_rates(geom, r, t, fl(FT, c["clumping"])))
        else
            add!("foursail_canopy_rates/plain/$(c["name"])", "foursail_canopy_rates(geom, r, t)",
                foursail_canopy_rates(geom, r, t))
        end
    end

    # ---- population quadrature and angular emission laws
    for lname in fx["api_population"]["layouts"]
        pin = population_inputs(FT, fx, lname)
        g = pin.g
        coords = [population_coordinates(g, k) for k in 1:length(g)]
        add!("population_coordinates/$lname", "population_coordinates",
            (; n_populations = length(g),
                (f => [getfield(c, f) for c in coords] for f in POPULATION_FIELDS)...))
        nd = length(g.depths)
        for (law_name, law) in laws
            for step in 1:nd
                add!("depth_hemispheric_source/$law_name/$lname/depth$step", "depth_hemispheric_source",
                    depth_hemispheric_source(law, g.layout, g.cos_inclinations, pin.weights, pin.front,
                        pin.back, step, g.depth_weights[step]))
                vals = [depth_angular_source(law, g.layout, g.cos_inclinations, g.azimuths, pin.weights,
                    pin.front, pin.back, step, g.depth_weights[step], pin.ray_mu[r], pin.ray_phi[r])
                        for r in eachindex(pin.ray_mu)]
                add!("depth_angular_source/$law_name/$lname/depth$step", "depth_angular_source",
                    (; per_ray = vals))
            end
            up, dn = zeros(FT, nd), zeros(FT, nd)
            assemble_hemispheric_emission!(up, dn, law, g, pin.weights, pin.front, pin.back)
            add!("assemble_hemispheric_emission/$law_name/$lname", "assemble_hemispheric_emission!",
                (; up, down = dn))
            src = zeros(FT, length(pin.ray_mu), nd)
            assemble_emission!(src, law, g, pin.weights, pin.front, pin.back, pin.ray_mu, pin.ray_phi)
            add!("assemble_emission/$law_name/$lname", "assemble_emission!", (; source = src))
        end
    end
    for (law_name, law) in laws
        for c in fx["api_leaf_hemispheric_cases"]
            add!("leaf_hemispheric_emission/$law_name/$(c["name"])", "leaf_hemispheric_emission",
                leaf_hemispheric_emission(law, fl(FT, c["leaf_cosine"]), fl(FT, c["front"]), fl(FT, c["back"])))
        end
        for c in fx["api_leaf_angular_cases"]
            add!("leaf_angular_emission/$law_name/$(c["name"])", "leaf_angular_emission",
                (; value = leaf_angular_emission(law, fl(FT, c["leaf_cosine"]), fl(FT, c["leaf_azimuth"]),
                    fl(FT, c["ray_cosine"]), fl(FT, c["ray_azimuth"]), fl(FT, c["front"]), fl(FT, c["back"]);
                    mirrored_azimuth = c["mirrored_azimuth"]::Bool)))
        end
    end

    # ---- angular boundary coupling
    an = fx["api_angular"]
    mu, w = fl(FT, an["mu"]), fl(FT, an["weights"])
    n = length(mu)
    canopy = AngularEmissionBoundary(fmat(FT, an["canopy_reflection"], (n, n)),
        fmat(FT, an["emission"], an["emission_shape"]), mu, w, an["fourier_order"])
    atm = AngularAtmosphericTransfer(fmat(FT, an["return_down"], (n, n)), fmat(FT, an["transmit_up"], (n, n)),
        copy(mu), copy(w), an["fourier_order"])
    add!("couple_emission/vacuum", "couple_emission", couple_emission(canopy, VacuumAtmosphere()))
    add!("couple_emission/angular", "couple_emission", couple_emission(canopy, atm))
    add!("one_way_emission/vacuum", "one_way_emission", (; top_up = one_way_emission(canopy, VacuumAtmosphere())))
    add!("one_way_emission/angular", "one_way_emission", (; top_up = one_way_emission(canopy, atm)))

    # ---- leaf spectral emission (manufactured matrices)
    models = leaf_emission_models(FT, fx)
    add!("leaf_emission_matrix/three_argument", "LeafEmissionMatrix(front, back, reabsorbed)",
        (; out_of_window = models.three_arg.out_of_window))
    add!("validate_leaf_emission/outcomes", "validate_leaf_emission",
        (; psii = accepted(() -> validate_leaf_emission(models.psii)),
            psi = accepted(() -> validate_leaf_emission(models.psi)),
            bad = accepted(() -> validate_leaf_emission(models.bad)),
            no_leaf = accepted(() -> validate_leaf_emission(NoLeafEmission())),
            (Symbol("dual_", name) => accepted(() -> validate_leaf_emission(d)) for (name, d) in dual_combinations(models))...,
            dual_bad = accepted(() -> validate_leaf_emission(DualLeafEmissionMatrix(models.psii, models.bad)))))
    n_emit = size(models.psii.front, 1)
    n_pop = size(models.internal_psii, 2)
    f, b, r, o = escape_outputs(FT, n_emit, n_pop)
    add!("escape_leaf_emission/psii/blas", "escape_leaf_emission!",
        escape_leaf_emission!(f, b, r, models.psii, models.internal_psii; out_of_window = o))
    fo = OffsetArray(zeros(FT, n_emit, n_pop), 0, -3)
    bo, ro = similar(fo), similar(fo)
    oo = OffsetArray(zeros(FT, n_pop), -3)
    escape_leaf_emission!(fo, bo, ro, models.psii, OffsetArray(models.internal_psii, 0, -3); out_of_window = oo)
    add!("escape_leaf_emission/psii/offset_loop", "escape_leaf_emission!(OffsetArray)",
        (; front = parent(fo), back = parent(bo), reabsorbed = parent(ro), out_of_window = parent(oo)))
    f, b, r, o = escape_outputs(FT, n_emit, n_pop)
    add!("escape_leaf_emission/no_leaf", "escape_leaf_emission!(NoLeafEmission)",
        escape_leaf_emission!(f, b, r, NoLeafEmission(), models.internal_psii; out_of_window = o))
    for (name, d) in dual_combinations(models)
        f, b, r, o = escape_outputs(FT, n_emit, n_pop)
        add!("escape_dual_leaf_emission/$name", "escape_dual_leaf_emission!",
            escape_dual_leaf_emission!(f, b, r, d, models.internal_psii, models.internal_psi; out_of_window = o))
    end
    le = fx["api_leaf_emission"]
    dens = fmat(FT, le["density"], le["density_shape"])
    widths, wl = fl(FT, le["widths_nm"]), fl(FT, le["wavelengths_nm"])
    bins = photon_bins!(similar(dens), dens, widths)
    add!("photon_bins", "photon_bins!", (; output = bins))
    add!("photon_density", "photon_density!", (; output = photon_density!(similar(dens), bins, widths)))
    add!("photon_energy", "photon_energy!",
        (; output = photon_energy!(similar(dens), bins, wl, fl(FT, le["energy_coefficient"]))))

    # ---- spectral basis
    sp = fx["api_spectral"]
    idm = IdentitySpectralMap(fl(FT, sp["identity_grid"]))
    lin = LinearSpectralMap(fl(FT, sp["anchors"]), fl(FT, sp["targets"]))
    src = fmat(FT, sp["src"], sp["src_shape"])
    add!("spectral_map/identity/map_spectrum", "map_spectrum!(IdentitySpectralMap)",
        (; dest = map_spectrum!(similar(src), idm, src)))
    add!("spectral_map/identity/source_basis_weights", "source_basis_weights(IdentitySpectralMap)",
        (; weights = source_basis_weights(idm)))
    add!("spectral_map/linear/constructor", "LinearSpectralMap", lin)
    add!("spectral_map/linear/map_spectrum", "map_spectrum!(LinearSpectralMap)",
        (; dest = map_spectrum!(zeros(FT, length(target_grid(lin)), size(src, 2)), lin, src)))
    add!("spectral_map/linear/source_basis_weights", "source_basis_weights(LinearSpectralMap)",
        (; weights = source_basis_weights(lin)))
    peb = prepare_emission_basis(models.psii, fmat(FT, sp["excitation_spectra"], sp["excitation_shape"]))
    add!("prepare_emission_basis/partly_active", "prepare_emission_basis", peb)
    pez = prepare_emission_basis(models.psii, fmat(FT, sp["zero_excitation_spectra"], sp["zero_excitation_shape"]))
    add!("prepare_emission_basis/all_zero", "prepare_emission_basis",
        (; totals = pez.totals, active = pez.active, model_is_nothing = pez.model === nothing,
            conditional_excitation = pez.conditional_excitation))
    ain = anchor_inputs(FT, fx)
    basis = prepare_anchor_basis(ain.map, ain.incident, ain.eta, ain.model, ain.internal_probability, ain.energy)
    add!("prepare_anchor_basis", "prepare_anchor_basis",
        (; (k => v for (k, v) in pairs(basis) if !(k in ANCHOR_BASIS_INPUT_COPIES))...,
            labels_anchor = [l.anchor for l in basis.labels],
            labels_boundary = [l.boundary for l in basis.labels],
            labels_receiving_face = [l.receiving_face for l in basis.labels]))
    amps = anchor_amplitudes(ain.response_front, ain.response_back, basis)
    add!("anchor_amplitudes/all_populations", "anchor_amplitudes", (; amplitudes = amps))
    add!("anchor_amplitudes/population_subset", "anchor_amplitudes(populations)",
        (; amplitudes = anchor_amplitudes(ain.response_front, ain.response_back, basis; populations = ain.subset)))
    add!("validate_basis_amplitudes/outcomes", "validate_basis_amplitudes",
        (; recorded = accepted(() -> validate_basis_amplitudes(amps, length(basis.labels))),
            negated = accepted(() -> validate_basis_amplitudes(-amps, length(basis.labels)))))

    # ---- hemispheric SIF
    sl = sif_layers(FT, fx)
    sx = fx["api_sif"]
    single_models = (("psii", models.psii), ("no_leaf", NoLeafEmission()), ("nothing", nothing))
    for (mname, m) in single_models, bin in Int.(sx["bins"])
        mname == "psii" || bin == first(sx["bins"]) || continue
        add!("evaluate_hemispheric_sif_bin/single/$mname/bin$bin", "evaluate_hemispheric_sif_bin(single)",
            evaluate_hemispheric_sif_bin(sl.psii, m, sl.absorption, sl.backscatter, sl.soil, bin))
    end
    dual_bin = Int(sx["dual_bin"])
    for (name, d) in (dual_combinations(models)..., ("no_leaf", NoLeafEmission()))
        add!("evaluate_hemispheric_sif_bin/dual/$name/bin$dual_bin", "evaluate_hemispheric_sif_bin(dual)",
            evaluate_hemispheric_sif_bin(sl.dual, d, sl.absorption, sl.backscatter, sl.soil, dual_bin))
    end
    spectrum_cases = (
        ("single/psii", sl.psii, models.psii),
        ("dual/psii_psi", sl.dual, DualLeafEmissionMatrix(models.psii, models.psi)),
        ("dual/none_psi", sl.dual, DualLeafEmissionMatrix(NoLeafEmission(), models.psi)),
        ("dual/no_leaf", sl.dual, NoLeafEmission()),
    )
    for (name, layers, m) in spectrum_cases
        add!("evaluate_hemispheric_sif_spectrum/$name", "evaluate_hemispheric_sif_spectrum",
            evaluate_hemispheric_sif_spectrum(layers, m, sl.absorption_matrix, sl.backscatter_matrix,
                sl.soil_spectrum))
    end
    return rec
end

# ---------------------------------------------------------------------------
# CanopyOptics-dependent tables
# ---------------------------------------------------------------------------
function emission_case_inputs(::Type{FT}, fx, c) where {FT}
    e = fx["canopyoptics_emission"]
    spectrum = CanopyOptics.TabulatedPhotonSpectrum(fl(FT, e["wavelength_nm"]), fl(FT, e["relative_photon_mass"]))
    support = c["support"] == "strict" ? CanopyOptics.StrictStokes() :
              c["support"] == "full" ? FullEmissionSupport() : error("unknown support")
    excitation = fl(FT, e["excitation_nm"])
    window = c["window"] == "all" ? collect(eachindex(spectrum.wavelength_nm)) :
             c["window"] == "subset" ? Int.(e["subset_window"]) : error("unknown window")
    depth, trans = fl(FT, e["escape_emission_depth"]), fl(FT, e["escape_interface_transmission"])
    escapes = [CanopyOptics.RegularPlateEscape(depth[i], trans[i]) for i in window]
    a, fwd, bwd = fl(FT, e["birth_excitation_depth"]), fl(FT, e["birth_forward"]), fl(FT, e["birth_backward"])
    oow = c["out_of_window"] == "none" ? nothing :
          c["out_of_window"] == "scalar" ? fl(FT, e["out_of_window_scalar"]) :
          c["out_of_window"] == "vector" ? fl(FT, e["out_of_window_vector"]) : error("unknown out_of_window")
    births = if c["births"] == "exponential"
        ([CanopyOptics.ExponentialBirthProfile(a[i], fwd[i], bwd[i]) for i in eachindex(a)],)
    elseif c["births"] == "uniform"
        ([CanopyOptics.UniformBirthProfile() for _ in eachindex(a)],)
    elseif c["births"] == "faces"
        ([CanopyOptics.ExponentialBirthProfile(a[i], fwd[i], bwd[i]) for i in eachindex(a)],
            [CanopyOptics.ExponentialBirthProfile(a[i], bwd[i], fwd[i]) for i in eachindex(a)])
    else
        error("unknown births")
    end
    return (; spectrum, support, excitation, window, escapes, births, oow)
end

"""
    compute_canopyoptics_record(FT, fx) -> Vector{RecordEntry}

Record entry points whose values are computed (in part) by CanopyOptics code.
"""
function compute_canopyoptics_record(::Type{FT}, fx) where {FT}
    P = string(FT)
    rec = RecordEntry[]
    add!(key, entry, x) = push!(rec, RecordEntry("$P/canopyoptics/$key", entry, fields_of(FT, x)))
    ld = fx["canopyoptics_leaf_distribution"]
    lad = CanopyOptics.beta_leaves(fl(FT, ld["beta_alpha"]), fl(FT, ld["beta_beta"]))
    q = leaf_inclination_quadrature(lad, ld["n_leaf"], FT)
    add!("leaf_inclination_quadrature", "leaf_inclination_quadrature", (; theta = q.θ, w_measure = q.w_measure))
    for c in fx["canopyoptics_orientation_cases"]
        add!("orientation_hemisphere_samples/$(c["name"])", "orientation_hemisphere_samples",
            orientation_hemisphere_samples(lad, fl(FT, c["mu_s"]); n_leaf = c["n_leaf"], n_azimuth = c["n_azimuth"]))
    end
    pl = fx["canopyoptics_plate"]
    add!("regular_plate_passive/diffuse_plate_optics", "regular_plate_passive",
        regular_plate_passive(CanopyOptics.diffuse_plate_optics(fl(FT, pl["k"]), fl(FT, pl["nr"]))))
    for c in fx["canopyoptics_emission_cases"]
        inp = emission_case_inputs(FT, fx, c)
        method = length(inp.births) == 1 ? "births" : "front_back_births"
        model = build_leaf_emission_matrices(inp.spectrum, inp.support, inp.excitation, inp.escapes,
            inp.births...; window_indices = inp.window, out_of_window_fraction = inp.oow)
        add!("build_leaf_emission_matrices/$method/$(c["name"])", "build_leaf_emission_matrices($method)", model)
    end
    return rec
end

"""Version and git commit of the CanopyOptics package that is loaded."""
function canopyoptics_provenance()
    root = pkgdir(CanopyOptics)
    commit = "unknown"
    dirty = "unknown"
    if root !== nothing && isdir(joinpath(root, ".git"))
        try
            commit = readchomp(`git -C $root rev-parse HEAD`)
            dirty = success(`git -C $root diff --quiet HEAD -- src Project.toml`) ? "clean" : "modified"
        catch
        end
    end
    return (; version = string(pkgversion(CanopyOptics)), commit, tracked_source = dirty)
end
