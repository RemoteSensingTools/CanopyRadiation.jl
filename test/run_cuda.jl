using Aqua
using CUDA
using CanopyOptics
using CanopyRadiation
using Test

CUDA.allowscalar(false)

@testset "CanopyRadiation CUDA extension (Float32 & Float64, allowscalar=false)" begin
    ext = Base.get_extension(CanopyRadiation, :CanopyRadiationCUDAExt)
    @test ext !== nothing
    Aqua.test_piracies(ext; treat_as_own = [CanopyRadiation.adapt_radiation_storage])

    for FT in (Float32, Float64)
        rtol = FT === Float32 ? 2.0f-5 : 1.0e-12
        nb = 32
        nl = 4

        # 1. @. broadcast of foursail_canopy_rates, layer_operator, emitting_layer, & face optics
        geom = FourSAILGeometry(
            spherical_leaves();
            sza_deg = FT(35),
            vza_deg = zero(FT),
            raa_deg = zero(FT),
        )
        rho_h = FT.(range(0.05, 0.40; length = nb))
        tau_h = FT.(range(0.04, 0.42; length = nb))
        rho_d = CuArray(rho_h)
        tau_d = CuArray(tau_h)
        rates_d = foursail_canopy_rates.((geom,), rho_d, tau_d, FT(0.9))
        ops_ana_d = @. layer_operator(AnalyticTwoStream(), rates_d, FT(1.4))
        ops_nat_d = @. layer_operator(NativeTwoStream(FT(0.02), 14), rates_d, FT(1.4))
        ops_ana_h = Array(ops_ana_d)
        ops_nat_h = Array(ops_nat_d)
        for i in 1:nb
            r_cpu = foursail_canopy_rates(geom, rho_h[i], tau_h[i], FT(0.9))
            @test isapprox(ops_ana_h[i].reflection, ops_nat_h[i].reflection; rtol)
            @test isapprox(
                ops_ana_h[i].beam_down,
                layer_operator(AnalyticTwoStream(), r_cpu, FT(1.4)).beam_down;
                rtol,
            )
        end

        # 2. Batch passive & emitting column kernels on CuArray
        refl_h = FT(0.08) .+ FT(0.02) .* reshape(FT.(1:(nl * nb)) ./ FT(nl * nb), nl, nb)
        tran_h = FT(0.65) .- FT(0.03) .* reshape(FT.(1:(nl * nb)) ./ FT(nl * nb), nl, nb)
        bup_h = fill(FT(0.04), nl, nb)
        bdn_h = fill(FT(0.05), nl, nb)
        btr_h = fill(FT(0.55), nl, nb)
        d_top_h = fill(FT(0.75), nb)
        f_top_h = fill(FT(0.25), nb)
        soil_h = fill(FT(0.16), nb)

        dir_d = CUDA.zeros(FT, nl + 1, nb)
        dn_d = CUDA.zeros(FT, nl + 1, nb)
        up_d = CUDA.zeros(FT, nl + 1, nb)
        abs_d = CUDA.zeros(FT, nl, nb)
        grd_d = CUDA.zeros(FT, nb)
        batch_solve_columns!(
            dir_d,
            dn_d,
            up_d,
            abs_d,
            grd_d,
            CuArray(refl_h),
            CuArray(tran_h),
            CuArray(bup_h),
            CuArray(bdn_h),
            CuArray(btr_h),
            CuArray(d_top_h),
            CuArray(f_top_h),
            CuArray(soil_h),
        )
        dir_h, dn_h, up_h, abs_h, grd_h =
            zeros(FT, nl + 1, nb),
            zeros(FT, nl + 1, nb),
            zeros(FT, nl + 1, nb),
            zeros(FT, nl, nb),
            zeros(FT, nb)
        batch_solve_columns!(
            dir_h,
            dn_h,
            up_h,
            abs_h,
            grd_h,
            refl_h,
            tran_h,
            bup_h,
            bdn_h,
            btr_h,
            d_top_h,
            f_top_h,
            soil_h,
        )
        @test isapprox(Array(dn_d), dn_h; rtol)
        @test isapprox(Array(up_d), up_h; rtol)
        @test isapprox(Array(abs_d), abs_h; rtol)
        @test isapprox(Array(grd_d), grd_h; rtol)

        # Batch emitting cavity
        eup_h = fill(FT(12.0), nl, nb)
        edn_h = fill(FT(11.0), nl, nb)
        egen_h = fill(FT(25.0), nl, nb)
        dt_lw_h = fill(FT(300.0), nb)
        se_lw_h = fill(FT(380.0), nb)
        dn_ed = CUDA.zeros(FT, nl + 1, nb)
        up_ed = CUDA.zeros(FT, nl + 1, nb)
        net_ed = CUDA.zeros(FT, nl, nb)
        abs_ed = CUDA.zeros(FT, nl, nb)
        ns_ed = CUDA.zeros(FT, nb)
        ga_ed = CUDA.zeros(FT, nb)
        gen_ed = CUDA.zeros(FT, nb)
        batch_solve_emitting_columns!(
            dn_ed,
            up_ed,
            net_ed,
            abs_ed,
            ns_ed,
            ga_ed,
            gen_ed,
            CuArray(refl_h),
            CuArray(tran_h),
            CuArray(eup_h),
            CuArray(edn_h),
            CuArray(egen_h),
            CuArray(dt_lw_h),
            CuArray(soil_h),
            CuArray(se_lw_h),
        )
        @test isapprox(
            dt_lw_h .+ Array(gen_ed) .+ se_lw_h,
            Array(up_ed)[1, :] .+ vec(sum(Array(abs_ed); dims = 1)) .+ Array(ga_ed);
            rtol,
        )

        # 3. adapt_radiation_storage, CuBLAS dual escape, moments, and photon conversions
        m2_cpu = LeafEmissionMatrix(
            fill(FT(0.20), 3, 2),
            fill(FT(0.10), 3, 2),
            fill(FT(0.02), 3, 2),
            fill(FT(0.04), 2),
        )
        m1_cpu = LeafEmissionMatrix(
            fill(FT(0.15), 3, 2),
            fill(FT(0.10), 3, 2),
            fill(FT(0.05), 3, 2),
            fill(FT(0.10), 2),
        )
        dual_cpu = DualLeafEmissionMatrix(m2_cpu, m1_cpu)
        dual_gpu = adapt_radiation_storage(CuArray, dual_cpu)
        @test dual_gpu.psii.front isa CuArray
        @test adapt_radiation_storage(Array, dual_gpu) == dual_cpu

        bnd_cpu = AngularEmissionBoundary(FT[0.1;;], FT[0.5;;], FT[0.7], FT[1.0], 0)
        atm_cpu = AngularAtmosphericTransfer(FT[0.05;;], FT[0.9;;], FT[0.7], FT[1.0], 0)
        bnd_gpu = adapt_radiation_storage(CuArray, bnd_cpu)
        atm_gpu = adapt_radiation_storage(CuArray, atm_cpu)
        @test bnd_gpu.reflection isa CuArray && atm_gpu.return_down isa CuArray
        @test adapt_radiation_storage(Array, bnd_gpu) == bnd_cpu
        @test adapt_radiation_storage(Array, atm_gpu) == atm_cpu
        @test adapt_radiation_storage(CuArray, VacuumAtmosphere()) === VacuumAtmosphere()

        i2_h = fill(FT(1.5), 2, 16)
        i1_h = fill(FT(0.5), 2, 16)
        f_g, b_g, r_g = CUDA.zeros(FT, 3, 16), CUDA.zeros(FT, 3, 16), CUDA.zeros(FT, 3, 16)
        oow_g = CUDA.zeros(FT, 16)
        escape_dual_leaf_emission!(
            f_g,
            b_g,
            r_g,
            dual_gpu,
            CuArray(i2_h),
            CuArray(i1_h);
            out_of_window = oow_g,
        )
        f_c, b_c, r_c, oow_c = zeros(FT, 3, 16), zeros(FT, 3, 16), zeros(FT, 3, 16), zeros(FT, 16)
        escape_dual_leaf_emission!(
            f_c,
            b_c,
            r_c,
            dual_cpu,
            i2_h,
            i1_h;
            out_of_window = oow_c,
        )
        @test isapprox(Array(f_g), f_c; rtol)
        @test isapprox(Array(oow_g), oow_c; rtol)

        widths_d = CuArray(FT[2, 3, 4])
        wl_d = CuArray(FT[680, 720, 760])
        bins_g = similar(f_g)
        dens_g = similar(f_g)
        ener_g = similar(f_g)
        photon_bins!(bins_g, f_g, widths_d)
        photon_density!(dens_g, bins_g, widths_d)
        photon_energy!(ener_g, bins_g, wl_d, FT(0.1196))
        @test isapprox(Array(dens_g), f_c; rtol)

        # Hemispheric moments & batch expansion on GPU
        mu_d = CuArray(FT[0.25, 0.6, 0.9])
        w_4d = CUDA.fill(FT(0.2), 3, 4, 2, 8)
        plus_3d = CUDA.zeros(FT, 4, 2, 8)
        minus_3d = CUDA.zeros(FT, 4, 2, 8)
        batch_hemispheric_moments!(plus_3d, minus_3d, MixedLambertianFaces(FT(0.8)), mu_d, w_4d)
        @test isapprox(Array(plus_3d .+ minus_3d), fill(FT(0.6), 4, 2, 8); rtol)

        up_2d = CUDA.zeros(FT, 4, 3)
        dn_2d = CUDA.zeros(FT, 4, 3)
        batch_expand_hemispheric_sources!(
            up_2d,
            dn_2d,
            view(plus_3d, :, :, 1),
            view(minus_3d, :, :, 1),
            dual_gpu.psii.front,
            dual_gpu.psii.back,
        )
        @test all(>(zero(FT)), Array(up_2d))
    end
end

# ---------------------------------------------------------------------------
# CUDA batch kernels against the frozen v0.1.0 (33f015b) CPU record.
#
# The inputs come from test/frozen_record/inputs.toml and the reference values
# from test/frozen_record/reference.toml (CPU, KernelAbstractions batch path).
# The reference must NOT be regenerated from refactored code. GPU code may
# contract a * b + c into a fused multiply-add, so the comparison uses an
# explicit tolerance. The error of a value is measured in units of
# eps(FT) * scale, where scale is the largest magnitude in the same reference
# table (column flux scale); this keeps cancellation-dominated quantities such
# as near-zero layer absorption meaningful. For information the script also
# prints the largest deviation in eps of the value itself, over values with
# magnitude >= scale / 1024 (smaller values are cancellation residues, e.g. the
# -4e-15 absorption of a conservative layer), and the number of values that are
# not bit-identical.
# ---------------------------------------------------------------------------
isdefined(@__MODULE__, :FrozenRecord) || include(joinpath(@__DIR__, "frozen_record", "compute.jl"))

# Measured on 2x NVIDIA L40S (CUDA.jl 5.11.3, Julia 1.12.6): max 0.97 eps(FT)*scale
# (Float32 emitting), 0.71 (passive); Float64 hemispheric kernels bit-identical.
const CUDA_RECORD_TOL_EPS = 4.0

function cuda_record_deviation(ref_table, got::NamedTuple, ::Type{FT}) where {FT}
    expected = Dict(f => FT[FrozenRecord.from_hex(FT, s) for s in (v isa AbstractString ? [v] : v)]
                    for (f, v) in ref_table if !(f in FrozenRecord.METADATA_FIELDS))
    keys(expected) == Set(string.(keys(got))) || error("field sets differ: $(keys(expected)) vs $(keys(got))")
    scale = maximum((maximum(abs, v; init = zero(FT)) for v in values(expected)); init = zero(FT))
    scale > zero(FT) || (scale = one(FT))
    worst_scaled, worst_rel, n_diff = 0.0, 0.0, 0
    for (f, g) in pairs(got)
        r = expected[string(f)]
        gv = g isa AbstractArray ? vec(g) : [g]
        length(gv) == length(r) || error("field $f length differs")
        for i in eachindex(r, gv)
            d = Float64(abs(gv[i] - r[i]))
            worst_scaled = max(worst_scaled, d / Float64(eps(FT) * scale))
            n_diff += gv[i] !== r[i]
            abs(r[i]) >= scale / 1024 && (worst_rel = max(worst_rel, FrozenRecord.ulps_of_value(gv[i], r[i])))
        end
    end
    return worst_scaled, worst_rel, n_diff
end

@testset "CUDA batch kernels vs frozen v0.1.0 (33f015b) CPU record, tolerance $(CUDA_RECORD_TOL_EPS) eps" begin
    fx = FrozenRecord.load_fixture()
    ref = FrozenRecord.load_reference()
    to_gpu(x) = CuArray(x)
    for pname in fx["precisions"]
        FT = FrozenRecord.precision_type(pname)
        groups = Dict{String,NTuple{4,Float64}}()
        function track!(g, dev)
            old = get(groups, g, (0.0, 0.0, 0.0, 0.0))
            groups[g] = (max(old[1], dev[1]), max(old[2], dev[2]), old[3] + dev[3], old[4] + 1)
        end
        for col in fx["passive_columns"]
            inp = FrozenRecord.passive_batch_inputs(FT, fx, col)
            dir, dn, up, ab, gr = FrozenRecord.run_passive_batch(FT, inp, to_gpu, CUDA.zeros)
            for (b, sname) in enumerate(col["scenarios"])
                key = "$pname/passive/batch/$(col["name"])/$sname"
                got = (; direct = dir[:, b], down = dn[:, b], up = up[:, b],
                    absorption = ab[:, b], ground = gr[b])
                dev = cuda_record_deviation(ref[key], got, FT)
                track!("passive", dev)
                @test dev[1] <= CUDA_RECORD_TOL_EPS
            end
        end
        for col in fx["emitting_columns"]
            inp = FrozenRecord.emitting_batch_inputs(FT, fx, col)
            dn, up, net, ab, ns, ga, gen = FrozenRecord.run_emitting_batch(FT, inp, to_gpu, CUDA.zeros)
            for (b, sname) in enumerate(col["scenarios"])
                key = "$pname/emitting/batch/$(col["name"])/$sname"
                got = (; down = dn[:, b], up = up[:, b], net_layer = net[:, b], absorption = ab[:, b],
                    net_soil = ns[b], ground_absorbed = ga[b], generated = gen[b])
                dev = cuda_record_deviation(ref[key], got, FT)
                track!("emitting", dev)
                @test dev[1] <= CUDA_RECORD_TOL_EPS
            end
        end
        hin = FrozenRecord.hemispheric_inputs(FT, fx)
        for lname in sort!(collect(keys(fx["hemispheric"]["laws"])))
            law = FrozenRecord.law_of(FT, fx, lname)
            pb, mb = FrozenRecord.run_batch_moments(FT, law, hin, to_gpu, CUDA.zeros)
            dev = cuda_record_deviation(ref["$pname/hemispheric_moment/batch/$lname"], (; plus = pb, minus = mb), FT)
            track!("hemispheric_moments", dev)
            @test dev[1] <= CUDA_RECORD_TOL_EPS
            p, m = FrozenRecord.scalar_moments(FT, law, hin.mu, hin.weighted[:, :, :, 1])
            ub, db = FrozenRecord.run_batch_expand(FT, p, m, hin, to_gpu, CUDA.zeros)
            dev = cuda_record_deviation(ref["$pname/hemispheric_bin/batch/$lname"], (; up = ub, down = db), FT)
            track!("hemispheric_expand", dev)
            @test dev[1] <= CUDA_RECORD_TOL_EPS
        end
        for g in sort!(collect(keys(groups)))
            s, r, nd, nt = groups[g]
            println("CUDA vs frozen CPU record, $pname $g ($(Int(nt)) tables): max deviation ",
                "$(round(s; digits = 3)) eps(FT)*table scale; max $(r) eps of value (|value| >= scale/1024); ",
                "$(Int(nd)) value(s) not bit-identical")
        end
    end
end

# ---------------------------------------------------------------------------
# CUDA paths of the remaining public API against the frozen v0.1.0 CPU record
# (test/frozen_record/reference_api.toml; must NOT be regenerated from
# refactored code): photon_* KernelAbstractions kernels, CuBLAS
# escape_leaf_emission! / escape_dual_leaf_emission!, and map_spectrum!
# broadcasts. Same deviation measure and tolerance as above. The hemispheric
# SIF evaluators have no device path (they index layer sources on the host),
# so they are not run here.
# ---------------------------------------------------------------------------
@testset "CUDA API paths vs frozen v0.1.0 (33f015b) CPU record, tolerance $(CUDA_RECORD_TOL_EPS) eps" begin
    fx = FrozenRecord.load_fixture()
    ref = FrozenRecord.load_reference(FrozenRecord.REFERENCE_API_PATH)
    for pname in fx["precisions"]
        FT = FrozenRecord.precision_type(pname)
        groups = Dict{String,NTuple{4,Float64}}()
        function check!(group, key, got)
            dev = cuda_record_deviation(ref["$pname/$key"], got, FT)
            old = get(groups, group, (0.0, 0.0, 0.0, 0.0))
            groups[group] = (max(old[1], dev[1]), max(old[2], dev[2]), old[3] + dev[3], old[4] + 1)
            @test dev[1] <= CUDA_RECORD_TOL_EPS
        end

        le = fx["api_leaf_emission"]
        dens = CuArray(FrozenRecord.fmat(FT, le["density"], le["density_shape"]))
        widths = CuArray(FrozenRecord.fl(FT, le["widths_nm"]))
        wl = CuArray(FrozenRecord.fl(FT, le["wavelengths_nm"]))
        bins = photon_bins!(similar(dens), dens, widths)
        check!("photon_kernels", "photon_bins", (; output = Array(bins)))
        check!("photon_kernels", "photon_density", (; output = Array(photon_density!(similar(dens), bins, widths))))
        check!("photon_kernels", "photon_energy",
            (; output = Array(photon_energy!(similar(dens), bins, wl, FrozenRecord.fl(FT, le["energy_coefficient"])))))

        models = FrozenRecord.leaf_emission_models(FT, fx)
        n_emit, n_pop = size(models.psii.front, 1), size(models.internal_psii, 2)
        gpu_out() = (CUDA.zeros(FT, n_emit, n_pop), CUDA.zeros(FT, n_emit, n_pop), CUDA.zeros(FT, n_emit, n_pop),
            CUDA.zeros(FT, n_pop))
        host(r) = map(Array, r)
        f, b, r, o = gpu_out()
        check!("escape_cublas", "escape_leaf_emission/psii/blas",
            host(escape_leaf_emission!(f, b, r, adapt_radiation_storage(CuArray, models.psii),
                CuArray(models.internal_psii); out_of_window = o)))
        for (name, d) in FrozenRecord.dual_combinations(models)
            f, b, r, o = gpu_out()
            check!("escape_cublas", "escape_dual_leaf_emission/$name",
                host(escape_dual_leaf_emission!(f, b, r, adapt_radiation_storage(CuArray, d),
                    CuArray(models.internal_psii), CuArray(models.internal_psi); out_of_window = o)))
        end

        sp = fx["api_spectral"]
        src = FrozenRecord.fmat(FT, sp["src"], sp["src_shape"])
        lin = LinearSpectralMap(FrozenRecord.fl(FT, sp["anchors"]), FrozenRecord.fl(FT, sp["targets"]))
        idm = IdentitySpectralMap(FrozenRecord.fl(FT, sp["identity_grid"]))
        check!("map_spectrum", "spectral_map/linear/map_spectrum",
            (; dest = Array(map_spectrum!(CUDA.zeros(FT, length(target_grid(lin)), size(src, 2)), lin, CuArray(src)))))
        check!("map_spectrum", "spectral_map/identity/map_spectrum",
            (; dest = Array(map_spectrum!(CUDA.zeros(FT, size(src)...), idm, CuArray(src)))))

        for g in sort!(collect(keys(groups)))
            s, r, nd, nt = groups[g]
            println("CUDA vs frozen CPU record, $pname $g ($(Int(nt)) tables): max deviation ",
                "$(round(s; digits = 3)) eps(FT)*table scale; max $(r) eps of value (|value| >= scale/1024); ",
                "$(Int(nd)) value(s) not bit-identical")
        end
    end
end
