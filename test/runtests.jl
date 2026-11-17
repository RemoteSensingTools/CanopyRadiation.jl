using CanopyOptics
using CanopyRadiation
using LinearAlgebra
using OffsetArrays
using Test

include("qa_and_ad.jl")
include("precision_and_records.jl")
include("frozen_record.jl")
include("frozen_record_api.jl")
include("frozen_record_regimes.jl")

@testset "0-ULP donor parity, Native vs Analytic two-stream, and 7-field cavity conservation" begin
    for FT in (Float32, Float64)
        rtol = FT === Float32 ? 2.0f-5 : 1.0e-12
        lad = beta_leaves(FT(2.2), FT(1.8))
        geom = FourSAILGeometry(
            lad;
            sza_deg = FT(32),
            vza_deg = zero(FT),
            raa_deg = zero(FT),
            quadrature = CanopyQuadrature(n_leaf = 16, n_azimuth = 16),
        )
        rho = (FT(0.07), FT(0.09), FT(0.42))
        tau = (FT(0.05), FT(0.06), FT(0.46))
        lais = (FT(0.0), FT(1.1), FT(1.9))

        # Exact 0-ULP parity of foursail_canopy_rates with CanopyModelExamples/setup/canopy.jl
        rates = map((r, t) -> foursail_canopy_rates(geom, r, t), rho, tau)
        for i in 1:3
            r, t = rho[i], tau[i]
            donor = CanopyRates(
                one(FT) - (r + t),
                geom.ddb * r + geom.ddf * t,
                geom.sdb * r + geom.sdf * t,
                geom.sdf * r + geom.sdb * t,
                geom.ks,
            )
            @test rates[i] === donor
        end
        # Broadcast over vector
        rates_vec = foursail_canopy_rates.((geom,), collect(rho), collect(tau), FT(0.85))
        @test length(rates_vec) == 3
        @test isapprox(rates_vec[2].extinction, FT(0.85) * geom.ks; rtol)

        # Exact 0-ULP parity between solve_column(::NTuple) and zero-scratch solve_column!
        m_nat = NativeTwoStream(FT(0.01), 18)
        ops_nat = map((rt, L) -> layer_operator(m_nat, rt, L), rates, lais)
        ops_ana = map((rt, L) -> layer_operator(AnalyticTwoStream(), rt, L), rates, lais)
        for i in 1:3
            @test isapprox(ops_nat[i].reflection, ops_ana[i].reflection; rtol)
            @test isapprox(ops_nat[i].transmission, ops_ana[i].transmission; rtol)
            @test isapprox(ops_nat[i].beam_up, ops_ana[i].beam_up; rtol, atol = eps(FT) * FT(16))
            @test isapprox(ops_nat[i].beam_down, ops_ana[i].beam_down; rtol, atol = eps(FT) * FT(16))
            @test isapprox(ops_nat[i].beam_transmission, ops_ana[i].beam_transmission; rtol)
        end

        sol_tup = solve_column(ops_nat, FT(0.75), FT(0.25), FT(0.18))
        ws = ColumnFluxWorkspace(FT, 3)
        sol_vec = solve_column!(ws, collect(ops_nat), FT(0.75), FT(0.25), FT(0.18))
        @test Tuple(sol_vec.direct) === sol_tup.direct
        @test Tuple(sol_vec.down) === sol_tup.down
        @test Tuple(sol_vec.up) === sol_tup.up
        @test Tuple(sol_vec.absorption) === sol_tup.absorption
        @test sol_vec.ground === sol_tup.ground
        @test isapprox(sol_tup.up[1] + sum(sol_tup.absorption) + sol_tup.ground, one(FT); rtol)

        # Conservative (α = 0), resonant (k = m), and thick-slab (L = 50) AnalyticTwoStream checks
        r_cons = CanopyRates(zero(FT), FT(0.35), FT(0.18), FT(0.22), FT(0.40))
        op_cons = layer_operator(AnalyticTwoStream(), r_cons, FT(2.4))
        @test isapprox(op_cons.reflection + op_cons.transmission, one(FT); atol = eps(FT) * FT(16))
        @test isapprox(
            op_cons.beam_up + op_cons.beam_down + op_cons.beam_transmission,
            one(FT);
            rtol,
        )

        op_thick = layer_operator(AnalyticTwoStream(), CanopyRates(FT(0.2), FT(0.15), FT(0.06), FT(0.08), FT(0.7)), FT(50))
        @test isfinite(op_thick.reflection) && isfinite(op_thick.beam_up) && isfinite(op_thick.beam_down)

        # 7-field solve_emitting_column net + gross conservation + local Gauss-Legendre interior check
        z_gl, w_gl = CanopyOptics.gauleg(16, zero(FT), one(FT))
        em1 = NoEmission()
        em2 = UniformEmission(FT(140), FT(135))
        em3 = QuadratureEmission(
            z_gl,
            w_gl,
            FT(160) .* exp.(-FT(0.4) .* z_gl),
            FT(150) .* exp.(-FT(0.4) .* z_gl),
        )
        elayers = (
            emitting_layer(rates[1].absorption, rates[1].backscatter, lais[1], em1),
            emitting_layer(rates[2].absorption, rates[2].backscatter, lais[2], em2),
            emitting_layer(rates[3].absorption, rates[3].backscatter, lais[3], em3),
        )
        down_top = FT(310)
        soil_ref = FT(0.06)
        soil_em = FT(385)
        cav_tup = solve_emitting_column(elayers, down_top, soil_ref, soil_em)
        cav_vec = solve_emitting_column(collect(elayers), down_top, soil_ref, soil_em)
        @test Tuple(cav_vec.down) === cav_tup.down
        @test Tuple(cav_vec.up) === cav_tup.up
        @test Tuple(cav_vec.net_layer) === cav_tup.net_layer
        @test Tuple(cav_vec.absorption) === cav_tup.absorption
        @test cav_vec.net_soil === cav_tup.net_soil
        @test cav_vec.ground_absorbed === cav_tup.ground_absorbed

        # Net & gross balances
        @test isapprox(
            down_top - cav_tup.up[1],
            sum(cav_tup.net_layer) + cav_tup.net_soil;
            rtol,
        )
        @test isapprox(
            down_top + cav_tup.generated + soil_em,
            cav_tup.up[1] + sum(cav_tup.absorption) + cav_tup.ground_absorbed;
            rtol,
        )

        # Independent local quadrature integral of α * (D(x) + U(x)) inside layer 2 & layer 3.
        # For QuadratureEmission (discrete Dirac sheets at z_gl), integrate piecewise between
        # sheets so Gauss-Legendre acts on smooth C^∞ subintervals.
        for (idx, em, breaks) in (
            (2, em2, FT[zero(FT), one(FT)]),
            (3, em3, vcat(zero(FT), z_gl, one(FT))),
        )
            z_sub, w_sub = CanopyOptics.gauleg(6, zero(FT), one(FT))
            loc_abs = zero(FT)
            for p in 1:(length(breaks) - 1)
                a_p, b_p = breaks[p], breaks[p + 1]
                span_p = b_p - a_p
                span_p <= zero(FT) && continue
                for (zs, ws) in zip(z_sub, w_sub)
                    zf = a_p + span_p * zs
                    fl = emission_interior(
                        rates[idx].absorption,
                        rates[idx].backscatter,
                        lais[idx],
                        zf * lais[idx],
                        em,
                        cav_tup.down[idx],
                        cav_tup.up[idx + 1],
                    )
                    loc_abs += span_p * ws * lais[idx] * rates[idx].absorption * (fl.down + fl.up)
                end
            end
            @test isapprox(loc_abs, cav_tup.absorption[idx]; rtol = FT === Float32 ? 2.0f-5 : 1.0e-12)
        end

        # Zero-incident solve_emission bit parity between tuple and vector
        se_tup = @inferred solve_emission(elayers, soil_ref)
        se_vec = solve_emission(collect(elayers), soil_ref)
        @test Tuple(se_vec.down) === se_tup.down
        @test Tuple(se_vec.up) === se_tup.up
        @test Tuple(se_vec.absorption) === se_tup.absorption
        @test se_vec.ground === se_tup.ground

        # Zero-allocation and type-inference checks for tuple and empty-tuple column solvers
        @test (@inferred solve_column((), FT(0.75), FT(0.25), FT(0.18))).absorption === ()
        @test (@inferred solve_emitting_column((), down_top, soil_ref, soil_em)).absorption === ()
        @test (@inferred solve_emission((), soil_ref)).absorption === ()
        _alloc_col(ops, d, f, s) = @allocated solve_column(ops, d, f, s)
        _alloc_emit_col(els, dt, sr, se) = @allocated solve_emitting_column(els, dt, sr, se)
        _alloc_emit(els, sr) = @allocated solve_emission(els, sr)
        _alloc_col(ops_nat, FT(0.75), FT(0.25), FT(0.18))
        _alloc_emit_col(elayers, down_top, soil_ref, soil_em)
        _alloc_emit(elayers, soil_ref)
        @test _alloc_col(ops_nat, FT(0.75), FT(0.25), FT(0.18)) == 0
        @test _alloc_emit_col(elayers, down_top, soil_ref, soil_em) == 0
        @test _alloc_emit(elayers, soil_ref) == 0
    end
end

@testset "MixedLambertianFaces solid-angle quadrature, full-spectrum Stokes tail, and PSII+PSI superposition" begin
    # 1. MixedLambertianFaces hemisphere solid-angle quadrature == leaf_hemispheric_emission
    mu_r, w_mu = CanopyOptics.gauleg(48, 0.0, 1.0)
    phi_r, w_phi = CanopyOptics.gauleg(64, 0.0, 2.0 * pi)
    for lam in (0.0, 0.35, 0.7, 1.0)
        law = lam == 1.0 ? LambertianFaces() : MixedLambertianFaces(lam)
        analytic = leaf_hemispheric_emission(law, 0.42, 0.85, 0.35)
        @test isapprox(analytic.upward + analytic.downward, 1.20; atol = 1e-15)
        num_up = 0.0
        num_dn = 0.0
        for (mr, wm) in zip(mu_r, w_mu), (pr, wp) in zip(phi_r, w_phi)
            j_up = leaf_angular_emission(law, 0.42, 0.6, mr, pr, 0.85, 0.35; mirrored_azimuth = true)
            j_dn = leaf_angular_emission(law, 0.42, 0.6, -mr, pr, 0.85, 0.35; mirrored_azimuth = true)
            @test j_up >= 0.0 && j_dn >= 0.0
            num_up += wm * wp * j_up
            num_dn += wm * wp * j_dn
        end
        @test isapprox(num_up, analytic.upward; rtol = 1e-4)
        @test isapprox(num_dn, analytic.downward; rtol = 1e-4)
    end

    # 2. Full-spectrum (600:5:1000 nm) window_indices slicing under StrictStokes & FullEmissionSupport
    wl_full = collect(600.0:5.0:1000.0) # 81 wavelengths
    mass_full = exp.(-((wl_full .- 730.0) ./ 45.0) .^ 2) .+ 0.08 .* exp.(-((wl_full .- 820.0) ./ 70.0) .^ 2)
    spec_full = TabulatedPhotonSpectrum(wl_full, mass_full)
    win_idx = findall(w -> 640.0 <= w <= 850.0, wl_full)
    exc_nm = [440.0, 620.0, 665.0, 695.0]
    escapes = [RegularPlateEscape(0.25 + 0.001 * i, 0.92) for i in eachindex(win_idx)]
    births_f = [ExponentialBirthProfile(1.8, 1.0, 0.05) for _ in exc_nm]
    births_b = [ExponentialBirthProfile(1.8, 0.05, 1.0) for _ in exc_nm]

    mat_stokes = build_leaf_emission_matrices(
        spec_full,
        StrictStokes(),
        exc_nm,
        escapes,
        births_f,
        births_b;
        window_indices = win_idx,
    )
    @test size(mat_stokes.front) == (length(win_idx), 2 * length(exc_nm))
    @test all(>(0.0), mat_stokes.out_of_window)
    # Excitation at 695 nm cuts off 600..695 nm, so the >850 nm fraction is larger than at 440 nm!
    @test mat_stokes.out_of_window[4] != mat_stokes.out_of_window[1]

    # Reject scalar nonzero out_of_window_fraction on pre-truncated StrictStokes when exc >= first_pos_nm
    spec_trunc = TabulatedPhotonSpectrum(wl_full[win_idx], mass_full[win_idx])
    @test_throws ArgumentError build_leaf_emission_matrices(
        spec_trunc,
        StrictStokes(),
        [440.0, 640.0], # inclusive >= 640.0!
        escapes,
        births_f[1:2];
        out_of_window_fraction = 0.05,
    )

    # Basis pullback preserves out_of_window conservation cleanly!
    smap = LinearSpectralMap([440.0, 695.0], exc_nm)
    inc = hcat(fill(1.2, 4), fill(0.6, 4))
    eta = [0.48, 0.45, 0.42, 0.35]
    in_prob = mat_stokes.front .+ mat_stokes.back .+ mat_stokes.reabsorbed
    en = 0.1196 ./ (exc_nm .* 1e-3)
    basis = prepare_anchor_basis(smap, inc, eta, mat_stokes, in_prob, en; atol = 1e-12)
    @test basis.model !== nothing
    @test all(>(0.0), basis.model.out_of_window)
    @test isapprox(
        vec(sum(basis.internal; dims = 1)) .+ basis.model.out_of_window,
        ones(length(basis.active));
        atol = 1e-12,
    )

    # 3. Exact PSII + PSI canopy superposition identity
    mat_psi = build_leaf_emission_matrices(
        spec_full,
        FullEmissionSupport(),
        exc_nm[1:3],
        escapes,
        births_f[1:3];
        window_indices = win_idx,
    )
    mat_psii = build_leaf_emission_matrices(
        spec_full,
        StrictStokes(),
        exc_nm,
        escapes,
        births_f;
        window_indices = win_idx,
    )
    dual_mat = DualLeafEmissionMatrix(mat_psii, mat_psi)

    z_d, w_d = CanopyOptics.gauleg(4, 0.0, 1.0)
    mu_o = [0.3, 0.7, 0.9]
    law = LambertianFaces()
    layers_psii = LayerHemisphericSource{Float64,Vector{Float64},Matrix{Float64}}[]
    layers_psi = LayerHemisphericSource{Float64,Vector{Float64},Matrix{Float64}}[]
    layers_dual = DualLayerHemisphericSource[]
    for (k, lai_k) in enumerate((1.1, 1.7))
        w2 = fill(0.05 * k, 3, 4, 4)
        w1 = fill(0.02 * k, 3, 4, 3)
        p2, m2 = zeros(4, 4), zeros(4, 4)
        p1, m1 = zeros(4, 3), zeros(4, 3)
        hemispheric_source_moments!(p2, m2, law, mu_o, w2)
        hemispheric_source_moments!(p1, m1, law, mu_o, w1)
        phys2 = [lai_k * sum(w_d[d] * sum(w2[:, d, c]) for d in 1:4) for c in 1:4]
        phys1 = [lai_k * sum(w_d[d] * sum(w1[:, d, c]) for d in 1:4) for c in 1:3]
        s2 = LayerHemisphericSource(lai_k, z_d, w_d, p2, m2, phys2)
        s1 = LayerHemisphericSource(lai_k, z_d, w_d, p1, m1, phys1)
        push!(layers_psii, s2)
        push!(layers_psi, s1)
        push!(layers_dual, DualLayerHemisphericSource(s2, s1))
    end
    n_em = length(win_idx)
    abs_mat = fill(0.06, n_em, 2)
    bck_mat = fill(0.28, n_em, 2)
    soil_sp = fill(0.22, n_em)

    sif_2 = evaluate_hemispheric_sif_spectrum(layers_psii, mat_psii, abs_mat, bck_mat, soil_sp)
    sif_1 = evaluate_hemispheric_sif_spectrum(layers_psi, mat_psi, abs_mat, bck_mat, soil_sp)
    sif_12 = evaluate_hemispheric_sif_spectrum(layers_dual, dual_mat, abs_mat, bck_mat, soil_sp)

    @test isapprox(sif_12.top, sif_2.top .+ sif_1.top; rtol = 1e-13)
    @test isapprox(sif_12.ground, sif_2.ground .+ sif_1.ground; rtol = 1e-13)
    @test isapprox(sif_12.layer_absorption, sif_2.layer_absorption .+ sif_1.layer_absorption; rtol = 1e-13)
    @test isapprox(sif_12.within_leaf, sif_2.within_leaf .+ sif_1.within_leaf; rtol = 1e-13)
    @test isapprox(sif_12.out_of_window, sif_2.out_of_window .+ sif_1.out_of_window; rtol = 1e-13)
    # Full 5-fate photon conservation: top + canopy_reabs + soil + within_leaf + oow == total_created
    total_created =
        sum(sum(s.physical_channels) for s in layers_psii) +
        sum(sum(s.physical_channels) for s in layers_psi)
    five_fate_sum =
        sum(sif_12.top) +
        sum(sif_12.layer_absorption) +
        sum(sif_12.ground) +
        sum(sif_12.within_leaf) +
        sum(sif_12.out_of_window)
    @test isapprox(five_fate_sum, total_created; rtol = 1e-13)
end

@testset "Generic indexing (OffsetArray & @view) and CPU batch kernels" begin
    ops = [
        layer_operator(AnalyticTwoStream(), CanopyRates(0.14, 0.09, 0.04, 0.05, 0.65), 1.2),
        layer_operator(AnalyticTwoStream(), CanopyRates(0.11, 0.12, 0.05, 0.06, 0.70), 1.6),
    ]
    ops_off = OffsetArray(ops, -1:0)
    ws_off = ColumnFluxWorkspace(
        OffsetArray(zeros(3), 10:12),
        OffsetArray(zeros(3), 10:12),
        OffsetArray(zeros(3), 10:12),
        OffsetArray(zeros(2), -5:-4),
    )
    sol_ref = solve_column(ops, 0.7, 0.3, 0.15)
    sol_off = solve_column!(ws_off, ops_off, 0.7, 0.3, 0.15)
    @test collect(sol_off.down) == sol_ref.down
    @test collect(sol_off.up) == sol_ref.up
    @test collect(sol_off.absorption) == sol_ref.absorption

    # OffsetArray LeafEmissionMatrix & escape_leaf_emission! & photon_bins!/density!/energy!
    f_m = OffsetArray([0.35 0.25; 0.15 0.25], 0:1, 10:11)
    b_m = OffsetArray([0.20 0.15; 0.10 0.15], 0:1, 10:11)
    r_m = OffsetArray([0.10 0.10; 0.05 0.05], 0:1, 10:11)
    oow_v = OffsetArray([0.05, 0.05], 10:11)
    lem_off = LeafEmissionMatrix(f_m, b_m, r_m, oow_v)
    validate_leaf_emission(lem_off; atol = 1e-12)
    lem_3 = LeafEmissionMatrix(f_m, b_m, OffsetArray([0.15 0.15; 0.05 0.05], 0:1, 10:11))
    @test axes(lem_3.out_of_window) == (10:11,)

    int_off = OffsetArray([2.0 3.0; 1.0 4.0], 10:11, -2:-1)
    out_f = OffsetArray(zeros(2, 2), 0:1, -2:-1)
    out_b = OffsetArray(zeros(2, 2), 0:1, -2:-1)
    out_r = OffsetArray(zeros(2, 2), 0:1, -2:-1)
    out_o = OffsetArray(zeros(2), -2:-1)
    escape_leaf_emission!(out_f, out_b, out_r, lem_off, int_off; out_of_window = out_o)
    @test isapprox(collect(out_f), parent(f_m) * parent(int_off); rtol = 1e-14)
    @test isapprox(collect(out_o), vec(transpose(parent(int_off)) * parent(oow_v)); rtol = 1e-14)

    widths_off = OffsetArray([2.0, 4.0], 0:1)
    wl_off = OffsetArray([680.0, 740.0], 0:1)
    bins_off = similar(out_f)
    dens_off = similar(out_f)
    ener_off = similar(out_f)
    photon_bins!(bins_off, out_f, widths_off)
    photon_density!(dens_off, bins_off, widths_off)
    photon_energy!(ener_off, bins_off, wl_off, 0.1196)
    @test isapprox(collect(dens_off), collect(out_f); rtol = 1e-14)

    # OffsetArray PopulationQuadrature & assemble_hemispheric_emission! / assemble_emission!
    pq_off = PopulationQuadrature(
        1,
        1.2,
        0.0,
        OffsetArray([0.3, 0.9], 5:6),
        OffsetArray([0.5, 0.5], 5:6),
        OffsetArray([0.4, 0.8], -1:0),
        OffsetArray([0.5, 1.1], -1:0),
        PairedIllumination(),
    )
    @test population_coordinates(pq_off, 1).depth_index == 5
    w_pop = OffsetArray(fill(0.125, 8), 20:27)
    fr_pop = OffsetArray(fill(0.6, 8), 20:27)
    bk_pop = OffsetArray(fill(0.4, 8), 20:27)
    up_p = OffsetArray(zeros(2), 5:6)
    dn_p = OffsetArray(zeros(2), 5:6)
    assemble_hemispheric_emission!(up_p, dn_p, LambertianFaces(), pq_off, w_pop, fr_pop, bk_pop)
    @test all(x -> isapprox(x, 1.0; rtol = 1e-13), up_p .+ dn_p)

    # Batch column & emitting-column kernels match scalar solves
    nb = 5
    refl = hcat([fill(0.05 + 0.01 * b, 2) for b in 1:nb]...)
    tran = hcat([fill(0.70 - 0.01 * b, 2) for b in 1:nb]...)
    bup = fill(0.04, 2, nb)
    bdn = fill(0.05, 2, nb)
    btr = fill(0.60, 2, nb)
    d_top = collect(range(0.5, 0.9; length = nb))
    f_top = collect(range(0.1, 0.3; length = nb))
    s_alb = fill(0.15, nb)
    dir_o, dn_o, up_o = zeros(3, nb), zeros(3, nb), zeros(3, nb)
    abs_o, grd_o = zeros(2, nb), zeros(nb)
    batch_solve_columns!(dir_o, dn_o, up_o, abs_o, grd_o, refl, tran, bup, bdn, btr, d_top, f_top, s_alb)
    for b in 1:nb
        ops_b = [
            LayerOperator(refl[1, b], tran[1, b], bup[1, b], bdn[1, b], btr[1, b]),
            LayerOperator(refl[2, b], tran[2, b], bup[2, b], bdn[2, b], btr[2, b]),
        ]
        ref_b = solve_column(ops_b, d_top[b], f_top[b], s_alb[b])
        @test dn_o[:, b] == ref_b.down
        @test up_o[:, b] == ref_b.up
        @test abs_o[:, b] == ref_b.absorption
        @test grd_o[b] == ref_b.ground
    end

    # OffsetArray support across all four batch_*! kernels
    dir_off = OffsetArray(zeros(3, nb), 10:12, -2:(nb - 3))
    dn_off = OffsetArray(zeros(3, nb), 10:12, -2:(nb - 3))
    up_off = OffsetArray(zeros(3, nb), 10:12, -2:(nb - 3))
    abs_off = OffsetArray(zeros(2, nb), -5:-4, -2:(nb - 3))
    grd_off = OffsetArray(zeros(nb), -2:(nb - 3))
    batch_solve_columns!(
        dir_off,
        dn_off,
        up_off,
        abs_off,
        grd_off,
        OffsetArray(refl, 0:1, -2:(nb - 3)),
        OffsetArray(tran, 0:1, -2:(nb - 3)),
        OffsetArray(bup, 0:1, -2:(nb - 3)),
        OffsetArray(bdn, 0:1, -2:(nb - 3)),
        OffsetArray(btr, 0:1, -2:(nb - 3)),
        OffsetArray(d_top, -2:(nb - 3)),
        OffsetArray(f_top, -2:(nb - 3)),
        OffsetArray(s_alb, -2:(nb - 3)),
    )
    @test parent(dn_off) == dn_o
    @test parent(up_off) == up_o
    @test parent(abs_off) == abs_o
    @test parent(grd_off) == grd_o

    eup = fill(12.0, 2, nb)
    edn = fill(11.0, 2, nb)
    egen = fill(25.0, 2, nb)
    dt_lw = fill(300.0, nb)
    se_lw = fill(380.0, nb)
    dn_e1, up_e1 = zeros(3, nb), zeros(3, nb)
    net_e1, abs_e1 = zeros(2, nb), zeros(2, nb)
    ns_e1, ga_e1, gen_e1 = zeros(nb), zeros(nb), zeros(nb)
    batch_solve_emitting_columns!(
        dn_e1,
        up_e1,
        net_e1,
        abs_e1,
        ns_e1,
        ga_e1,
        gen_e1,
        refl,
        tran,
        eup,
        edn,
        egen,
        dt_lw,
        s_alb,
        se_lw,
    )
    dn_eoff = OffsetArray(zeros(3, nb), 20:22, 3:(nb + 2))
    up_eoff = OffsetArray(zeros(3, nb), 20:22, 3:(nb + 2))
    net_eoff = OffsetArray(zeros(2, nb), -1:0, 3:(nb + 2))
    abs_eoff = OffsetArray(zeros(2, nb), -1:0, 3:(nb + 2))
    ns_eoff = OffsetArray(zeros(nb), 3:(nb + 2))
    ga_eoff = OffsetArray(zeros(nb), 3:(nb + 2))
    gen_eoff = OffsetArray(zeros(nb), 3:(nb + 2))
    batch_solve_emitting_columns!(
        dn_eoff,
        up_eoff,
        net_eoff,
        abs_eoff,
        ns_eoff,
        ga_eoff,
        gen_eoff,
        OffsetArray(refl, 5:6, 3:(nb + 2)),
        OffsetArray(tran, 5:6, 3:(nb + 2)),
        OffsetArray(eup, 5:6, 3:(nb + 2)),
        OffsetArray(edn, 5:6, 3:(nb + 2)),
        OffsetArray(egen, 5:6, 3:(nb + 2)),
        OffsetArray(dt_lw, 3:(nb + 2)),
        OffsetArray(s_alb, 3:(nb + 2)),
        OffsetArray(se_lw, 3:(nb + 2)),
    )
    @test parent(dn_eoff) == dn_e1
    @test parent(up_eoff) == up_e1
    @test parent(abs_eoff) == abs_e1
    @test parent(gen_eoff) == gen_e1

    mu_bm = [0.3, 0.7]
    w_bm = reshape(collect(range(0.1, 0.8; length = 24)), 2, 3, 2, 2)
    p_bm, m_bm = zeros(3, 2, 2), zeros(3, 2, 2)
    batch_hemispheric_moments!(p_bm, m_bm, MixedLambertianFaces(0.75), mu_bm, w_bm)
    p_bmo = OffsetArray(zeros(3, 2, 2), 0:2, 10:11, -1:0)
    m_bmo = OffsetArray(zeros(3, 2, 2), 0:2, 10:11, -1:0)
    batch_hemispheric_moments!(
        p_bmo,
        m_bmo,
        MixedLambertianFaces(0.75),
        OffsetArray(mu_bm, 4:5),
        OffsetArray(w_bm, 4:5, 0:2, 10:11, -1:0),
    )
    @test isapprox(parent(p_bmo), p_bm; rtol = 1e-14)
    @test isapprox(parent(m_bmo), m_bm; rtol = 1e-14)

    fr_be = [0.25 0.15; 0.10 0.20; 0.05 0.10]
    bk_be = [0.20 0.10; 0.12 0.18; 0.04 0.08]
    up_be, dn_be = zeros(3, 3), zeros(3, 3)
    batch_expand_hemispheric_sources!(up_be, dn_be, p_bm[:, :, 1], m_bm[:, :, 1], fr_be, bk_be)
    up_beo = OffsetArray(zeros(3, 3), 0:2, 20:22)
    dn_beo = OffsetArray(zeros(3, 3), 0:2, 20:22)
    batch_expand_hemispheric_sources!(
        up_beo,
        dn_beo,
        view(p_bmo, :, :, -1),
        view(m_bmo, :, :, -1),
        OffsetArray(fr_be, 20:22, 10:11),
        OffsetArray(bk_be, 20:22, 10:11),
    )
    @test isapprox(parent(up_beo), up_be; rtol = 1e-14)
    @test isapprox(parent(dn_beo), dn_be; rtol = 1e-14)
end
