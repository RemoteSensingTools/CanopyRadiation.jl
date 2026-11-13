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
