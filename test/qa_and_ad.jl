using Aqua
using CanopyOptics
using CanopyRadiation
using Documenter
using ExplicitImports
using ForwardDiff
using LinearAlgebra
using OffsetArrays
using Test

include("forwarddiff_hooks.jl")

@testset "Package QA (Aqua, ExplicitImports, Docs, Doctests)" begin
    Aqua.test_all(CanopyRadiation; ambiguities = true)

    @test check_no_implicit_imports(CanopyRadiation) === nothing
    @test check_no_stale_explicit_imports(CanopyRadiation) === nothing
    @test check_all_explicit_imports_via_owners(CanopyRadiation) === nothing
    @test check_all_qualified_accesses_via_owners(CanopyRadiation) === nothing
    @test check_no_self_qualified_accesses(CanopyRadiation) === nothing
    @static if VERSION >= v"1.11"
        # The ForwardDiff extension imports `Dual`, `value` and `partials`, which
        # ForwardDiff documents but does not declare `public`.
        @test check_all_explicit_imports_are_public(
            CanopyRadiation;
            ignore = (:Dual, :value, :partials),
        ) === nothing
        @test check_all_qualified_accesses_are_public(CanopyRadiation) === nothing
    end

    @static if isdefined(Docs, :undocumented_names)
        @test isempty(Docs.undocumented_names(CanopyRadiation))
    end

    DocMeta.setdocmeta!(
        CanopyRadiation,
        :DocTestSetup,
        :(using CanopyRadiation, CanopyOptics);
        recursive = true,
    )
    doctest(CanopyRadiation; manual = false)
end

@testset "Continuous scalar guards, signed-zero hashing, and Base interfaces" begin
    mkdual(v, p) = ForwardDiff.Dual{Nothing}(v, p)
    discrete_dual = ForwardDiff.Dual{Nothing}(1, 1)

    # Functions under the precision policy (two_stream.jl, emission_transport.jl)
    # reject non-floating-point scalars with an ArgumentError; unit_sin_from_cos
    # (scalars.jl, not yet converted) still with a MethodError.
    for bad in (true, 1, 1 // 2, pi, discrete_dual)
        @test_throws ArgumentError CanopyRates(bad, bad, bad, bad, bad)
        @test_throws ArgumentError LayerOperator(bad, bad, bad, bad, bad)
        @test_throws ArgumentError UniformEmission(bad, bad)
        @test_throws ArgumentError diffuse_response(bad, bad, bad)
        @test_throws MethodError CanopyRadiation.unit_sin_from_cos(bad)
    end

    # Mixed Float32 + Float64 rejected via ArgumentError (precision policy)
    @test_throws ArgumentError CanopyRates(0.1f0, 0.1, 0.05, 0.05, 0.5)
    @test_throws ArgumentError diffuse_response(0.1f0, 0.1, 1.0)
    @test_throws ArgumentError solve_column(
        (LayerOperator(0.1f0, 0.5f0, 0.05f0, 0.05f0, 0.6f0),),
        1.0,
        0.2,
        0.1,
    )

    # Nonfinite Dual partials rejected at validated boundaries
    nan_dual = mkdual(0.5, NaN)
    inf_dual = mkdual(0.5, Inf)
    @test_throws ArgumentError NativeTwoStream(nan_dual, 12)
    @test_throws ArgumentError MixedLambertianFaces(inf_dual)
    @test_throws ArgumentError QuadratureEmission(
        [nan_dual],
        [mkdual(1.0, 0.0)],
        [mkdual(0.2, 0.0)],
        [mkdual(0.2, 0.0)],
    )
    @test_throws ArgumentError LayerHemisphericSource(
        nan_dual,
        [mkdual(0.5, 0.0)],
        [mkdual(1.0, 0.0)],
        reshape([mkdual(0.1, 0.0)], 1, 1),
        reshape([mkdual(0.1, 0.0)], 1, 1),
        [mkdual(0.4, 0.0)],
    )

    # Signed-zero equality & hash laws on scalar AND array structs
    r_pos = CanopyRates(+0.0, 0.1, +0.0, 0.1, 0.5)
    r_neg = CanopyRates(-0.0, 0.1, -0.0, 0.1, 0.5)
    @test r_pos == r_neg
    @test hash(r_pos) == hash(r_neg)

    rd_pos = CanopyRates(mkdual(+0.0, 1.0), mkdual(0.1, 0.0), mkdual(+0.0, 0.0), mkdual(0.1, 0.0), mkdual(0.5, 0.0))
    rd_neg = CanopyRates(mkdual(-0.0, 1.0), mkdual(0.1, 0.0), mkdual(-0.0, 0.0), mkdual(0.1, 0.0), mkdual(0.5, 0.0))
    @test rd_pos == rd_neg
    @test hash(rd_pos) == hash(rd_neg)

    q_pos = QuadratureEmission([+0.0, 0.5], [0.5, 0.5], [+0.0, 0.2], [0.1, -0.0])
    q_neg = QuadratureEmission([-0.0, 0.5], [0.5, 0.5], [-0.0, 0.2], [0.1, +0.0])
    @test q_pos == q_neg
    @test hash(q_pos) == hash(q_neg)

    m_pos = LeafEmissionMatrix([+0.0 0.4; 0.5 0.1], [0.5 0.3; +0.0 0.2], [0.0 0.0; 0.0 0.0])
    m_neg = LeafEmissionMatrix([-0.0 0.4; 0.5 0.1], [0.5 0.3; -0.0 0.2], [-0.0 -0.0; 0.0 0.0])
    @test m_pos == m_neg
    @test hash(m_pos) == hash(m_neg)

    qd_pos = QuadratureEmission(
        [mkdual(+0.0, 1.0)],
        [mkdual(1.0, 0.0)],
        [mkdual(+0.0, 2.0)],
        [mkdual(0.2, -0.0)],
    )
    qd_neg = QuadratureEmission(
        [mkdual(-0.0, 1.0)],
        [mkdual(1.0, -0.0)],
        [mkdual(-0.0, 2.0)],
        [mkdual(0.2, +0.0)],
    )
    @test qd_pos == qd_neg
    @test hash(qd_pos) == hash(qd_neg)

    # Primal vs zero-partial Dual hash & equality law: a == b => hash(a) == hash(b)
    r_f64 = CanopyRates(0.1, 0.2, 0.05, 0.05, 0.5)
    r_d0 = CanopyRates(
        mkdual(0.1, 0.0),
        mkdual(0.2, 0.0),
        mkdual(0.05, 0.0),
        mkdual(0.05, 0.0),
        mkdual(0.5, 0.0),
    )
    @test r_d0 == r_f64
    @test hash(r_d0) == hash(r_f64)

    op_f64 = LayerOperator(0.1, 0.5, 0.04, 0.05, 0.6)
    op_d0 = LayerOperator(
        mkdual(0.1, 0.0),
        mkdual(0.5, 0.0),
        mkdual(0.04, 0.0),
        mkdual(0.05, 0.0),
        mkdual(0.6, 0.0),
    )
    @test op_d0 == op_f64
    @test hash(op_d0) == hash(op_f64)

    bnd_f64 = AngularEmissionBoundary([0.1;;], [0.5;;], [0.7], [1.0], 0)
    bnd_d0 = AngularEmissionBoundary(
        [mkdual(0.1, 0.0);;],
        [mkdual(0.5, 0.0);;],
        [mkdual(0.7, 0.0)],
        [mkdual(1.0, 0.0)],
        0,
    )
    @test bnd_d0 == bnd_f64
    @test hash(bnd_d0) == hash(bnd_f64)

    atm_f64 = AngularAtmosphericTransfer([0.05;;], [0.9;;], [0.7], [1.0], 0)
    atm_d0 = AngularAtmosphericTransfer(
        [mkdual(0.05, 0.0);;],
        [mkdual(0.9, 0.0);;],
        [mkdual(0.7, 0.0)],
        [mkdual(1.0, 0.0)],
        0,
    )
    @test atm_d0 == atm_f64
    @test hash(atm_d0) == hash(atm_f64)

    # isequal reflexivity on NaN fields and signed-zero distinction across custom-== structs
    @test isequal(r_pos, r_neg) == false
    @test isequal(q_pos, q_neg) == false
    @test isequal(m_pos, m_neg) == false

    r_nan = CanopyRates(NaN, 0.2, 0.05, 0.05, 0.5)
    op_nan = LayerOperator(NaN, 0.5, 0.04, 0.05, 0.6)
    ue_nan = UniformEmission(NaN, 0.2)
    qe_nan = QuadratureEmission([0.5], [1.0], [NaN], [0.2]; validate = false)
    el_nan = EmittingLayer(op_nan, NaN, 0.2, 0.4)
    bnd_nan = AngularEmissionBoundary([NaN;;], [0.5;;], [0.7], [1.0], 0)
    atm_nan = AngularAtmosphericTransfer([NaN;;], [0.9;;], [0.7], [1.0], 0)
    lhs_nan = LayerHemisphericSource(
        1.0,
        [0.5],
        [1.0],
        [NaN;;],
        [0.1;;],
        [0.4];
        validate = false,
    )
    dlhs_nan = DualLayerHemisphericSource(lhs_nan, lhs_nan)
    # Mutate validated array-backed containers post-construction to inject NaN for reflexivity check
    pq_nan = PopulationQuadrature(1, 1.0, 0.0, [0.5], [1.0], [0.5], [0.2], PairedIllumination())
    pq_nan.cos_inclinations[1] = NaN
    lem_nan = LeafEmissionMatrix([0.2;;], [0.2;;], [0.1;;], [0.1])
    lem_nan.front[1, 1] = NaN
    dlem_nan = DualLeafEmissionMatrix(lem_nan, lem_nan)
    ism_nan = IdentitySpectralMap([400.0, 500.0])
    ism_nan.wavelengths[1] = NaN
    lsm_nan = LinearSpectralMap([400.0, 500.0], [450.0])
    lsm_nan.targets[1] = NaN

    for x_nan in (
        r_nan,
        op_nan,
        ue_nan,
        qe_nan,
        el_nan,
        bnd_nan,
        atm_nan,
        lhs_nan,
        dlhs_nan,
        pq_nan,
        lem_nan,
        dlem_nan,
        ism_nan,
        lsm_nan,
    )
        @test isequal(x_nan, x_nan)
    end
    @test NativeTwoStream(mkdual(0.1, +0.0), 12) == NativeTwoStream(mkdual(0.1, -0.0), 12)
    @test !isequal(NativeTwoStream(mkdual(0.1, +0.0), 12), NativeTwoStream(mkdual(0.1, -0.0), 12))
    @test !isequal(MixedLambertianFaces(+0.0), MixedLambertianFaces(-0.0))

    # Bounds, channel-axis, n_emit == 0, and scratch-buffer aliasing guards
    src_4ch = LayerHemisphericSource(
        1.2,
        [0.25, 0.75],
        [0.5, 0.5],
        fill(0.05, 2, 4),
        fill(0.05, 2, 4),
        fill(0.24, 4),
    )
    mat_1ch = LeafEmissionMatrix(fill(0.2, 2, 1), fill(0.2, 2, 1), fill(0.1, 2, 1), [0.1])
    dual_mat_1ch = DualLeafEmissionMatrix(mat_1ch, mat_1ch)
    @test_throws DimensionMismatch evaluate_hemispheric_sif_bin(
        [src_4ch],
        mat_1ch,
        [0.1],
        [0.2],
        0.15,
        1,
    )
    @test_throws DimensionMismatch evaluate_hemispheric_sif_spectrum(
        [src_4ch],
        mat_1ch,
        zeros(0, 1),
        zeros(0, 1),
        Float64[],
    )
    @test_throws ArgumentError evaluate_hemispheric_sif_spectrum(
        [src_4ch],
        dual_mat_1ch,
        zeros(0, 1),
        zeros(0, 1),
        Float64[],
    )
    @test_throws ArgumentError evaluate_hemispheric_sif_spectrum(
        [DualLayerHemisphericSource(src_4ch, src_4ch)],
        mat_1ch,
        zeros(0, 1),
        zeros(0, 1),
        Float64[],
    )

    @test_throws DimensionMismatch hemispheric_moment(LambertianFaces(), [0.4, 0.8], ones(3, 2, 1), 1, 1)
    @test_throws BoundsError hemispheric_moment(LambertianFaces(), [0.4, 0.8], ones(2, 2, 1), 3, 1)
    @test_throws DimensionMismatch hemispheric_bin(ones(2, 3), ones(2, 3), ones(2), ones(2), 1)
    @test_throws BoundsError hemispheric_bin(ones(2, 2), ones(2, 2), ones(2), ones(2), 5)
    @test_throws BoundsError depth_hemispheric_source(
        LambertianFaces(),
        PairedIllumination(),
        [0.5],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.0, 1.0],
        2,
        1.0,
    )
    @test_throws BoundsError depth_angular_source(
        LambertianFaces(),
        PairedIllumination(),
        [0.5],
        [0.2],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.0, 1.0],
        2,
        1.0,
        0.7,
        0.1,
    )

    shared_vec = zeros(4)
    @test_throws ArgumentError ColumnFluxWorkspace(zeros(4), shared_vec, shared_vec, zeros(3))
    @test_throws ArgumentError EmittingColumnWorkspace(shared_vec, shared_vec, zeros(3), zeros(3))
    shared_m = zeros(3, 2)
    @test_throws ArgumentError batch_solve_columns!(
        zeros(3, 2),
        shared_m,
        shared_m,
        zeros(2, 2),
        zeros(2),
        fill(0.1, 2, 2),
        fill(0.6, 2, 2),
        fill(0.04, 2, 2),
        fill(0.05, 2, 2),
        fill(0.5, 2, 2),
        ones(2),
        ones(2),
        fill(0.1, 2),
    )
    @test_throws ArgumentError batch_solve_emitting_columns!(
        shared_m,
        shared_m,
        zeros(2, 2),
        zeros(2, 2),
        zeros(2),
        zeros(2),
        zeros(2),
        fill(0.1, 2, 2),
        fill(0.6, 2, 2),
        ones(2, 2),
        ones(2, 2),
        ones(2, 2),
        ones(2),
        fill(0.1, 2),
        ones(2),
    )
end

@testset "ForwardDiff 1st/2nd-order AD: conservative α -> 0, resonant k -> m, lai = 0, and zenith |mu| = 1" begin
    # 1. Conservative α = 0 and tiny α = 1e-12 right-derivatives vs BigFloat
    for alpha0 in (0.0, 1.0e-12, 0.12)
        f_refl(a) = diffuse_response(a, 0.28, 2.1).reflection
        f_s1(a) = diffuse_response(a, 0.28, 2.1).source_first
        d1 = ForwardDiff.derivative(f_refl, alpha0)
        d2 = ForwardDiff.derivative(a -> ForwardDiff.derivative(f_refl, a), alpha0)
        s1_d1 = ForwardDiff.derivative(f_s1, alpha0)
        s1_d2 = ForwardDiff.derivative(a -> ForwardDiff.derivative(f_s1, a), alpha0)
        @test isfinite(d1) && isfinite(d2) && isfinite(s1_d1) && isfinite(s1_d2)

        ab = BigFloat(alpha0)
        hb = BigFloat(10)^(-15)
        # Central or 4th-order forward difference in BigFloat at α = 0
        d1_bf = Float64(ForwardDiff.derivative(a -> diffuse_response(a, BigFloat("0.28"), BigFloat("2.1")).reflection, ab))
        d2_bf = Float64(
            ForwardDiff.derivative(
                b -> ForwardDiff.derivative(a -> diffuse_response(a, BigFloat("0.28"), BigFloat("2.1")).reflection, b),
                ab,
            ),
        )
        @test isapprox(d1, d1_bf; rtol = 1e-11, atol = 1e-13)
        @test isapprox(d2, d2_bf; rtol = 1e-10, atol = 1e-12)
        # Independent finite-difference check at interior point
        if alpha0 > 0
            fd1 = Float64((diffuse_response(ab + hb, BigFloat("0.28"), BigFloat("2.1")).reflection -
                           diffuse_response(ab - hb, BigFloat("0.28"), BigFloat("2.1")).reflection) / (2 * hb))
            @test isapprox(d1, fd1; rtol = 1e-11)
        else
            # Exact analytical right-derivative at α = 0 for R = bL / (1 + bL):
            # D(q) = cosh(sqrt(q)) + (α + b) L sinh(sqrt(q))/sqrt(q), q = 2 b α L^2 + α^2 L^2
            # At α = 0: v_tilde = L * (1 + q/6), D_tilde = (1 + q/2) + (α + b) * L * (1 + q/6)
            b = BigFloat("0.28")
            L = BigFloat("2.1")
            fd_fwd = Float64(
                (-diffuse_response(2 * hb, b, L).reflection +
                 4 * diffuse_response(hb, b, L).reflection -
                 3 * diffuse_response(zero(BigFloat), b, L).reflection) / (2 * hb),
            )
            @test isapprox(d1, fd_fwd; rtol = 1e-11)
        end
    end

    # 2. AnalyticTwoStream & NativeTwoStream 1st and 2nd derivatives at α = 0 AND resonant k = m
    for method in (AnalyticTwoStream(), NativeTwoStream(0.02, 16))
        # Resonant k = m: α = 0.16, b = 0.09 => m = sqrt(0.16 * 0.34)
        alpha_r = 0.16
        b_r = 0.09
        m_r = sqrt(alpha_r * (alpha_r + 2 * b_r))
        for (a0, k0) in ((0.0, 0.65), (1e-11, 0.65), (alpha_r, m_r), (alpha_r, m_r + 1e-7))
            vec_fn(p) = begin
                al, bk, su, sd, ex, lai = p
                op = layer_operator(method, CanopyRates(al, bk, su, sd, ex), lai)
                return op.reflection + 1.3 * op.transmission + 2.1 * op.beam_up + 1.7 * op.beam_down
            end
            p0 = [a0, b_r, 0.04, 0.05, k0, 1.8]
            g = ForwardDiff.gradient(vec_fn, p0)
            H = ForwardDiff.hessian(vec_fn, p0)
            @test all(isfinite, g)
            @test all(isfinite, H)

            pb = BigFloat.(p0)
            hb_hess = BigFloat(10)^(-20)
            m_bf = method isa NativeTwoStream ? NativeTwoStream(BigFloat("0.02"), 16) : AnalyticTwoStream()
            vec_bf(p) = begin
                al, bk, su, sd, ex, lai = p
                op = layer_operator(m_bf, CanopyRates(al, bk, su, sd, ex), lai)
                return op.reflection + BigFloat("1.3") * op.transmission +
                       BigFloat("2.1") * op.beam_up + BigFloat("1.7") * op.beam_down
            end
            g_bf = Float64.(ForwardDiff.gradient(vec_bf, pb))
            H_bf = Matrix{Float64}(undef, 6, 6)
            for j in 1:6
                p_plus = copy(pb)
                p_plus[j] += hb_hess
                if a0 == 0.0 && j == 1
                    p_plus2 = copy(pb)
                    p_plus2[j] += 2 * hb_hess
                    g0 = ForwardDiff.gradient(vec_bf, pb)
                    g1 = ForwardDiff.gradient(vec_bf, p_plus)
                    g2 = ForwardDiff.gradient(vec_bf, p_plus2)
                    H_bf[:, j] .= Float64.((-g2 .+ 4 .* g1 .- 3 .* g0) ./ (2 * hb_hess))
                else
                    p_minus = copy(pb)
                    p_minus[j] -= hb_hess
                    H_bf[:, j] .= Float64.((ForwardDiff.gradient(vec_bf, p_plus) .- ForwardDiff.gradient(vec_bf, p_minus)) ./ (2 * hb_hess))
                end
            end
            @test isapprox(g, g_bf; rtol = 1e-10, atol = 1e-12)
            @test isapprox(H, H_bf; rtol = 1e-8, atol = 1e-10)
        end

        # Empty-layer right-tangent sensitivity wrt lai at lai = 0
        r_test = CanopyRates(0.15, 0.08, 0.04, 0.06, 0.7)
        @test ForwardDiff.derivative(L -> layer_operator(method, r_test, L).beam_up, 0.0) ≈ r_test.beam_up rtol = 1e-12
        @test ForwardDiff.derivative(L -> layer_operator(method, r_test, L).beam_down, 0.0) ≈ r_test.beam_down rtol = 1e-12
        @test ForwardDiff.derivative(L -> layer_operator(method, r_test, L).reflection, 0.0) ≈ r_test.backscatter rtol = 1e-12
    end

    # 3. Zenith sun (mu_s = 1.0) & nadir view (ray_mu = 1.0) tangent safety (zero NaNs!)
    g_zenith = ForwardDiff.gradient(
        v -> begin
            mu_l, phi_l, fr, bk = v
            proj = source_projection(mu_l, phi_l, 1.0, 0.4)
            bf = beam_face_absorption(proj, 900.0, 0.85)
            ang = leaf_angular_emission(MixedLambertianFaces(0.75), mu_l, phi_l, 1.0, 0.0, fr, bk; mirrored_azimuth = true)
            return bf.front + bf.back + ang
        end,
        [0.6, 0.8, 1.2, 0.7],
    )
    @test all(isfinite, g_zenith)

    # Interior cosine derivatives match 256-bit BigFloat central finite differences of 1st-order AD
    f_ang(v) = leaf_angular_emission(LambertianFaces(), v[1], v[2], v[3], v[4], v[5], v[6]; mirrored_azimuth = true)
    v_ang = [0.55, 0.7, 0.82, 0.2, 1.1, 0.6]
    vb_ang = BigFloat.(v_ang)
    hb_ang = BigFloat(10)^(-20)
    H_ang_bf = Matrix{Float64}(undef, 6, 6)
    for j in 1:6
        vp, vm = copy(vb_ang), copy(vb_ang)
        vp[j] += hb_ang
        vm[j] -= hb_ang
        H_ang_bf[:, j] .= Float64.((ForwardDiff.gradient(f_ang, vp) .- ForwardDiff.gradient(f_ang, vm)) ./ (2 * hb_ang))
    end
    @test isapprox(ForwardDiff.hessian(f_ang, v_ang), H_ang_bf; rtol = 1e-11)

    # 4. End-to-end column + emitting cavity + plate + atmospheric coupling AD
    function column_cavity_pipeline(x)
        al, bk, lai1, lai2, soil, qu, qd, r_atm = x
        P = CanopyRadiation.primal_float_type(eltype(x))
        rates = CanopyRates(al, bk, P(0.04), P(0.05), P(0.65))
        op1 = layer_operator(AnalyticTwoStream(), rates, lai1)
        op2 = layer_operator(AnalyticTwoStream(), rates, lai2)
        sw = solve_column([op1, op2], P(0.75), P(0.25), soil)
        el1 = emitting_layer(al, bk, lai1, UniformEmission(qu, qd))
        el2 = emitting_layer(
            al,
            bk,
            lai2,
            QuadratureEmission(
                [P(0.25) + zero(al), P(0.75) + zero(al)],
                [P(0.5) + zero(al), P(0.5) + zero(al)],
                [qu, P(0.8) * qu],
                [qd, P(0.8) * qd],
            ),
        )
        lw = solve_emitting_column([el1, el2], P(300.0), soil, P(380.0))
        bnd = AngularEmissionBoundary(
            reshape([sw.up[1]], 1, 1),
            reshape([lw.up[1]], 1, 1),
            [P(0.7) + zero(al)],
            [P(1.0) + zero(al)],
            0,
        )
        atm = AngularAtmosphericTransfer(
            reshape([r_atm], 1, 1),
            reshape([one(al) - r_atm], 1, 1),
            [P(0.7) + zero(al)],
            [P(1.0) + zero(al)],
            0,
        )
        c = couple_emission(bnd, atm)
        return c.top_up[1, 1] + sum(sw.absorption)
    end
    x_col = [0.12, 0.09, 1.1, 1.6, 0.18, 12.0, 10.0, 0.06]
    xb_col = BigFloat.(x_col)
    hb_col = BigFloat(10)^(-20)
    H_col_bf = Matrix{Float64}(undef, 8, 8)
    for j in 1:8
        xp, xm = copy(xb_col), copy(xb_col)
        xp[j] += hb_col
        xm[j] -= hb_col
        H_col_bf[:, j] .= Float64.((ForwardDiff.gradient(column_cavity_pipeline, xp) .- ForwardDiff.gradient(column_cavity_pipeline, xm)) ./ (2 * hb_col))
    end
    @test isapprox(ForwardDiff.hessian(column_cavity_pipeline, x_col), H_col_bf; rtol = 1e-10)

    # 5. MixedLambertianFaces(Dual), hemispheric_bin(Dual), and mixed primal LeafEmissionMatrix{Float64}
    #    with LayerHemisphericSource{Dual} & DualLayerHemisphericSource{Dual}
    mu_vec = [0.35, 0.75]
    phi_vec = [0.3, 1.1]
    w_pop_ad = [0.25, 0.25, 0.25, 0.25]
    fr_pop_ad = [0.9, 0.6, 0.8, 0.5]
    bk_pop_ad = [0.4, 0.3, 0.5, 0.2]
    w3_ad = reshape([0.2, 0.4, 0.3, 0.5], 2, 1, 2)
    w4_ad = reshape([0.2, 0.4, 0.3, 0.5], 2, 1, 2, 1)

    d_dhs = ForwardDiff.derivative(
        lam -> depth_hemispheric_source(
            MixedLambertianFaces(lam),
            PairedIllumination(),
            mu_vec,
            w_pop_ad,
            fr_pop_ad,
            bk_pop_ad,
            1,
            1.0,
        ).upward,
        0.65,
    )
    hb_lam = BigFloat(10)^(-20)
    lam_bf = BigFloat("0.65")
    d_dhs_bf = Float64(
        (
            depth_hemispheric_source(
                MixedLambertianFaces(lam_bf + hb_lam),
                PairedIllumination(),
                BigFloat.(mu_vec),
                BigFloat.(w_pop_ad),
                BigFloat.(fr_pop_ad),
                BigFloat.(bk_pop_ad),
                1,
                BigFloat(1),
            ).upward -
            depth_hemispheric_source(
                MixedLambertianFaces(lam_bf - hb_lam),
                PairedIllumination(),
                BigFloat.(mu_vec),
                BigFloat.(w_pop_ad),
                BigFloat.(fr_pop_ad),
                BigFloat.(bk_pop_ad),
                1,
                BigFloat(1),
            ).upward
        ) / (2 * hb_lam),
    )
    @test isapprox(d_dhs, d_dhs_bf; rtol = 1e-12)

    d_das = ForwardDiff.derivative(
        lam -> depth_angular_source(
            MixedLambertianFaces(lam),
            PairedIllumination(),
            mu_vec,
            phi_vec,
            w_pop_ad,
            fr_pop_ad,
            bk_pop_ad,
            1,
            1.0,
            0.8,
            0.2,
        ),
        0.65,
    )
    @test isfinite(d_das)

    d_hm = ForwardDiff.derivative(
        lam -> hemispheric_moment(MixedLambertianFaces(lam), mu_vec, w3_ad, 1, 1).plus,
        0.65,
    )
    d_hsm = ForwardDiff.derivative(
        lam -> begin
            T = typeof(lam)
            p_m, m_m = zeros(T, 1, 2), zeros(T, 1, 2)
            hemispheric_source_moments!(p_m, m_m, MixedLambertianFaces(lam), mu_vec, w3_ad)
            return p_m[1, 1]
        end,
        0.65,
    )
    d_bhm = ForwardDiff.derivative(
        lam -> begin
            T = typeof(lam)
            p_b, m_b = zeros(T, 1, 2, 1), zeros(T, 1, 2, 1)
            batch_hemispheric_moments!(p_b, m_b, MixedLambertianFaces(lam), mu_vec, w4_ad)
            return p_b[1, 1, 1]
        end,
        0.65,
    )
    @test isapprox(d_hm, d_hsm; rtol = 1e-14)
    @test isapprox(d_hm, d_bhm; rtol = 1e-14)

    # hemispheric_bin with Dual front/back over Float64 plus/minus
    plus_f64 = [0.3 0.5; 0.2 0.4]
    minus_f64 = [0.2 0.4; 0.3 0.1]
    g_hbin = ForwardDiff.gradient(
        fb -> hemispheric_bin(plus_f64, minus_f64, fb[1:2], fb[3:4], 1).up,
        [0.4, 0.2, 0.1, 0.3],
    )
    @test isapprox(g_hbin, [0.3, 0.5, 0.2, 0.4]; rtol = 1e-14)
    @test_throws MethodError hemispheric_bin(plus_f64, minus_f64, Float32[0.4, 0.2], Float32[0.1, 0.3], 1)

    # Mixed primal LeafEmissionMatrix{Float64} & DualLeafEmissionMatrix + AD LayerHemisphericSource{Dual}
    primal_m2 = LeafEmissionMatrix(
        [0.22 0.18; 0.14 0.16],
        [0.18 0.14; 0.12 0.13],
        [0.10 0.08; 0.04 0.06],
        [0.05, 0.05],
    )
    primal_m1 = LeafEmissionMatrix(
        [0.15; 0.20;;],
        [0.14; 0.18;;],
        [0.08; 0.07;;],
        [0.08],
    )
    primal_dual_m = DualLeafEmissionMatrix(primal_m2, primal_m1)
    abs_m_f64 = [0.06 0.07; 0.05 0.06]
    bck_m_f64 = [0.28 0.30; 0.32 0.34]
    soil_f64 = [0.20, 0.24]

    sif_ad_fn(q) = begin
        q2_a, q2_b, q1_a = q
        T = eltype(q)
        z_v = T[0.25, 0.75]
        w_v = T[0.5, 0.5]
        s2_1 = LayerHemisphericSource(
            T(1.1),
            z_v,
            w_v,
            [0.5 * q2_a 0.5 * q2_b; 0.4 * q2_a 0.4 * q2_b],
            [0.5 * q2_a 0.5 * q2_b; 0.6 * q2_a 0.6 * q2_b],
            T[1.1 * q2_a, 1.1 * q2_b],
        )
        s2_2 = LayerHemisphericSource(
            T(1.4),
            z_v,
            w_v,
            [0.45 * q2_a 0.45 * q2_b; 0.35 * q2_a 0.35 * q2_b],
            [0.55 * q2_a 0.55 * q2_b; 0.65 * q2_a 0.65 * q2_b],
            T[1.4 * q2_a, 1.4 * q2_b],
        )
        s1_1 = LayerHemisphericSource(
            T(1.1),
            z_v,
            w_v,
            reshape(T[0.5 * q1_a, 0.5 * q1_a], 2, 1),
            reshape(T[0.5 * q1_a, 0.5 * q1_a], 2, 1),
            T[1.1 * q1_a],
        )
        s1_2 = LayerHemisphericSource(
            T(1.4),
            z_v,
            w_v,
            reshape(T[0.5 * q1_a, 0.5 * q1_a], 2, 1),
            reshape(T[0.5 * q1_a, 0.5 * q1_a], 2, 1),
            T[1.4 * q1_a],
        )
        bin1 = evaluate_hemispheric_sif_bin(
            [s2_1, s2_2],
            primal_m2,
            view(abs_m_f64, 1, :),
            view(bck_m_f64, 1, :),
            soil_f64[1],
            1,
        )
        spec_d = evaluate_hemispheric_sif_spectrum(
            [DualLayerHemisphericSource(s2_1, s1_1), DualLayerHemisphericSource(s2_2, s1_2)],
            primal_dual_m,
            abs_m_f64,
            bck_m_f64,
            soil_f64,
        )
        return bin1.up[1] + sum(spec_d.top) + sum(spec_d.out_of_window)
    end
    q0 = [0.8, 0.5, 0.3]
    g_sif = ForwardDiff.gradient(sif_ad_fn, q0)
    H_sif = ForwardDiff.hessian(sif_ad_fn, q0)
    @test all(isfinite, g_sif)
    @test all(isfinite, H_sif)
    # Linear wrt source channel amplitudes => Hessian is identically zero!
    @test maximum(abs, H_sif) < 1e-14
end
