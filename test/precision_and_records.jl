# Precision policy, scalar hooks and shared record behaviour of the converted core
# (two_stream.jl, emission_transport.jl): negative precision tests, promotion of
# mixed plain and dual inputs, inference and allocations, equality and hashing.

using CanopyOptics
using CanopyRadiation
using ForwardDiff
using OffsetArrays
using Test

include("forwarddiff_hooks.jl")

const CR = CanopyRadiation
dual(v, p) = ForwardDiff.Dual{Nothing}(v, p)

# One call per converted public function, every scalar of type FT (records and arrays
# included). Used with FT = Float32 or Float64 and with one argument replaced.
function converted_calls(FT)
    geom = FourSAILGeometry(spherical_leaves(); sza_deg = FT(30), vza_deg = zero(FT), raa_deg = zero(FT))
    r = CanopyRates(FT(0.15), FT(0.08), FT(0.04), FT(0.05), FT(0.65))
    op = LayerOperator(FT(0.05), FT(0.72), FT(0.04), FT(0.06), FT(0.5))
    q = QuadratureEmission(FT[0.25, 0.75], FT[0.5, 0.5], FT[0.3, 0.2], FT[0.3, 0.2])
    u = UniformEmission(FT(0.4), FT(0.35))
    el = emitting_layer(FT(0.1), FT(0.08), FT(1.5), u)
    return (; geom, r, op, q, u, el)
end

@testset "Precision policy: mixed Float32/Float64 and non-float inputs throw ArgumentError" begin
    c64 = converted_calls(Float64)
    c32 = converted_calls(Float32)
    nat64 = NativeTwoStream(0.05, 12)
    nat32 = NativeTwoStream(0.05f0, 12)
    ana = AnalyticTwoStream()
    for bad in (1.0f0, 1, true, 1 // 2, pi)
        @test_throws ArgumentError CanopyRates(0.1, 0.2, 0.05, bad, 0.6)
        @test_throws ArgumentError LayerOperator(bad, 0.7, 0.04, 0.06, 0.5)
        @test_throws ArgumentError UniformEmission(0.3, bad)
        @test_throws ArgumentError EmittingLayer(c64.op, 0.1, bad, 0.3)
        @test_throws ArgumentError foursail_canopy_rates(c64.geom, 0.1, bad)
        @test_throws ArgumentError foursail_canopy_rates(c64.geom, 0.1, 0.1, bad)
        @test_throws ArgumentError diffuse_response(0.1, 0.2, bad)
        @test_throws ArgumentError layer_operator(nat64, c64.r, bad)
        @test_throws ArgumentError layer_operator(ana, c64.r, bad)
        @test_throws ArgumentError interior_flux(ana, c64.r, 1.2, 0.6, 0.8, 0.1, bad)
        @test_throws ArgumentError point_emission_response(0.1, 0.2, 1.0, 0.4, bad, 0.5)
        @test_throws ArgumentError point_emission_interior(0.1, 0.2, 1.0, 0.4, bad, 0.5, 0.5)
        @test_throws ArgumentError emitting_layer(0.1, bad, 1.5, NoEmission())
        @test_throws ArgumentError emitting_layer(0.1, 0.2, bad, c64.u)
        @test_throws ArgumentError emitting_layer(0.1, 0.2, bad, c64.q)
        @test_throws ArgumentError quadrature_emitting_layer(bad, 0.2, 1.0, [0.5], [1.0], [0.3], [0.2])
        @test_throws ArgumentError emission_interior(0.1, 0.2, 1.0, 0.5, NoEmission(), bad, 0.1)
        @test_throws ArgumentError emission_interior(0.1, 0.2, 1.0, 0.5, c64.u, 0.3, bad)
        @test_throws ArgumentError emission_interior(0.1, 0.2, 1.0, bad, c64.q, 0.3, 0.1)
        @test_throws ArgumentError solve_column((), 0.7, bad, 0.15)
        @test_throws ArgumentError solve_column((c64.op,), 0.7, 0.3, bad)
        @test_throws ArgumentError solve_column([c64.op], bad, 0.3, 0.15)
        @test_throws ArgumentError solve_column!(ColumnFluxWorkspace(Float64, 1), [c64.op], 0.7, bad, 0.15)
        @test_throws ArgumentError solve_emitting_column((), bad, 0.05, 390.0)
        @test_throws ArgumentError solve_emitting_column((c64.el,), 300.0, bad, 390.0)
        @test_throws ArgumentError solve_emitting_column([c64.el], 300.0, 0.05, bad)
        @test_throws ArgumentError solve_emitting_column!(EmittingColumnWorkspace(Float64, 1), [c64.el], bad, 0.05, 390.0)
        bad isa AbstractFloat || @test_throws ArgumentError solve_emission((), bad)
        @test_throws ArgumentError solve_emission((c64.el,), bad)
        @test_throws ArgumentError solve_emission([c64.el], bad)
    end
    # Float32 records with Float64 scalars and vice versa
    @test_throws ArgumentError layer_operator(nat32, c64.r, 1.0)
    @test_throws ArgumentError layer_operator(nat64, c32.r, 1.0)
    @test_throws ArgumentError layer_operator(ana, c32.r, 1.0)
    @test_throws ArgumentError interior_flux(nat32, c64.r, 1.2, 0.6, 0.8, 0.1, 0.7)
    @test_throws ArgumentError EmittingLayer(c32.op, 0.1, 0.2, 0.3)
    @test_throws ArgumentError emitting_layer(0.1, 0.2, 1.5, c32.u)
    @test_throws ArgumentError emitting_layer(0.1, 0.2, 1.5, c32.q)
    @test_throws ArgumentError emission_interior(0.1, 0.2, 1.0, 0.5, c32.u, 0.3, 0.1)
    @test_throws ArgumentError quadrature_emitting_layer(0.1, 0.2, 1.0, Float32[0.5], [1.0], [0.3], [0.2])
    @test_throws ArgumentError solve_column((c64.op, c32.op), 0.7, 0.3, 0.15)
    @test_throws ArgumentError solve_column([c32.op], 0.7, 0.3, 0.15)
    @test_throws ArgumentError solve_column!(ColumnFluxWorkspace(Float64, 1), [c32.op], 0.7, 0.3, 0.15)
    @test_throws ArgumentError solve_column!(ColumnFluxWorkspace(Float32, 1), [c32.op], 0.7, 0.3, 0.15)
    @test_throws ArgumentError solve_emitting_column((c64.el, c32.el), 300.0, 0.05, 390.0)
    @test_throws ArgumentError solve_emitting_column!(EmittingColumnWorkspace(Float64, 1), [c32.el], 300.0, 0.05, 390.0)
    @test_throws ArgumentError solve_emission([c32.el], 0.1)
    @test_throws ArgumentError convert(LayerOperator{Float32}, c64.op)
    @test_throws ArgumentError convert(EmittingLayer{Float64}, c32.el)
    @test_throws ArgumentError convert(Vector{CanopyRates{Float32}}, [c64.r])
    @test_throws ArgumentError NativeTwoStream(Float32; thin_norm = 0.05)
    @test_throws ArgumentError NativeTwoStream(1, 12)
    # Integer element types of records, workspaces, arrays and batch inputs
    @test_throws ArgumentError CanopyRates{Int}(1, 1, 1, 1, 1)
    @test_throws ArgumentError LayerOperator{Int}(1, 1, 1, 1, 1)
    @test_throws ArgumentError UniformEmission{Int}(1, 1)
    @test_throws ArgumentError QuadratureEmission([0], [1], [1], [1])
    @test_throws ArgumentError ColumnFluxWorkspace(Int, 2)
    @test_throws ArgumentError EmittingColumnWorkspace(Int, 2)
    @test_throws ArgumentError ColumnFluxWorkspace(zeros(Int, 2), zeros(Int, 2), zeros(Int, 2), zeros(Int, 1))
    @test_throws ArgumentError EmittingColumnWorkspace(zeros(Int, 2), zeros(Int, 2), zeros(Int, 1), zeros(Int, 1))
    @test_throws ArgumentError batch_solve_columns!(
        zeros(Int, 2, 1), zeros(Int, 2, 1), zeros(Int, 2, 1), zeros(Int, 1, 1), zeros(Int, 1),
        zeros(Int, 1, 1), ones(Int, 1, 1), zeros(Int, 1, 1), zeros(Int, 1, 1), ones(Int, 1, 1),
        ones(Int, 1), zeros(Int, 1), zeros(Int, 1),
    )
    @test_throws ArgumentError batch_solve_emitting_columns!(
        zeros(Int, 2, 1), zeros(Int, 2, 1), zeros(Int, 1, 1), zeros(Int, 1, 1), zeros(Int, 1), zeros(Int, 1),
        zeros(Int, 1), zeros(Int, 1, 1), ones(Int, 1, 1), zeros(Int, 1, 1), zeros(Int, 1, 1), zeros(Int, 1, 1),
        ones(Int, 1), zeros(Int, 1), zeros(Int, 1),
    )
    # Validation messages are static and unchanged
    @test_throws ArgumentError("NativeTwoStream thin_norm must be finite and positive") NativeTwoStream(0.0, 12)
    @test_throws ArgumentError("NativeTwoStream source_terms must be positive") NativeTwoStream(0.05, 0)
    @test_throws ArgumentError("Clumping index must be finite and nonnegative") foursail_canopy_rates(c64.geom, 0.1, 0.1, -1.0)
    @test_throws ArgumentError foursail_canopy_rates(c64.geom, 0.6, 0.6)
    @test_throws ArgumentError QuadratureEmission([1.5], [1.0], [0.1], [0.1])
    @test_throws ArgumentError QuadratureEmission([0.5], [-1.0], [0.1], [0.1])
end

@testset "Precision policy: mixed plain and dual inputs promote to the dual" begin
    D = typeof(dual(1.0, 1.0))
    r = CanopyRates(0.15, 0.08, 0.04, 0.05, 0.65)
    rd = CanopyRates(dual(0.15, 1.0), 0.08, 0.04, 0.05, 0.65)
    @test rd isa CanopyRates{D}
    @test ForwardDiff.value(rd.absorption) === 0.15 && rd.backscatter == 0.08
    for m in (AnalyticTwoStream(), NativeTwoStream(0.02, 14), NativeTwoStream(dual(0.02, 0.0), 14))
        op = layer_operator(m, r, 1.2)
        opd = layer_operator(m, r, dual(1.2, 1.0))
        @test opd isa LayerOperator{D}
        @test all(map((a, b) -> ForwardDiff.value(a) ≈ b, CR._fields(opd), CR._fields(op)))
        @test layer_operator(m, rd, 1.2) isa LayerOperator{D}
        f = interior_flux(m, r, 1.2, dual(0.5, 1.0), 0.8, 0.1, 0.7)
        @test f.down isa D && ForwardDiff.value(f.down) ≈ interior_flux(m, r, 1.2, 0.5, 0.8, 0.1, 0.7).down
    end
    @test NativeTwoStream(D).thin_norm isa D
    op = layer_operator(AnalyticTwoStream(), r, 1.2)
    opd = layer_operator(AnalyticTwoStream(), r, dual(0.8, 1.0))
    @test EmittingLayer(op, dual(0.1, 1.0), 0.2, 0.3) isa EmittingLayer{D}
    @test diffuse_response(0.1, 0.2, dual(1.0, 1.0)).reflection isa D
    @test UniformEmission(0.3, dual(0.2, 1.0)) isa UniformEmission{D}
    @test emitting_layer(0.1, 0.2, dual(1.5, 1.0), UniformEmission(0.3, 0.2)) isa EmittingLayer{D}
    @test emitting_layer(dual(0.1, 1.0), 0.2, 1.5, QuadratureEmission([0.5], [1.0], [0.3], [0.2])) isa EmittingLayer{D}
    @test emitting_layer(0.1, 0.2, 1.5, NoEmission()) isa EmittingLayer{Float64}
    @test emission_interior(0.1, 0.2, 1.5, 0.7, UniformEmission(0.3, 0.2), dual(0.4, 1.0), 0.1).down isa D
    @test point_emission_response(0.1, 0.2, 1.0, dual(0.4, 1.0), 0.5, 0.5).upward isa D
    # Columns: one dual layer or one dual boundary value promotes the whole solution
    sol = solve_column((op, opd), 0.7, 0.3, 0.15)
    @test eltype(sol.down) === D
    # AnalyticTwoStream selects its branches on primal values: the dual carries the
    # plain result bit for bit
    @test map(ForwardDiff.value, sol.down) === solve_column((op, layer_operator(AnalyticTwoStream(), r, 0.8)), 0.7, 0.3, 0.15).down
    @test eltype(solve_column([op, op], dual(0.7, 1.0), 0.3, 0.15).down) === D
    el = emitting_layer(0.1, 0.2, 1.5, UniformEmission(0.3, 0.2))
    eld = emitting_layer(0.1, 0.2, dual(1.5, 1.0), UniformEmission(0.3, 0.2))
    @test eltype(solve_emitting_column((el, eld), 300.0, 0.05, 390.0).up) === D
    @test eltype(solve_emitting_column([el], 300.0, dual(0.05, 1.0), 390.0).up) === D
    @test eltype(solve_emission((eld, el), 0.1).up) === D
    # Containers of dual records accept plain records (conversion keeps the value)
    v = [opd]
    push!(v, op)
    @test v[2] isa LayerOperator{D} && map(ForwardDiff.value, CR._fields(v[2])) === CR._fields(op)
    @test convert(EmittingLayer{D}, el).passive isa LayerOperator{D}
end

@testset "Scalar hooks" begin
    @test Base.get_extension(CanopyRadiation, :CanopyRadiationForwardDiffExt) isa Module
    @test CR.primal_type(Float32) === Float32
    @test CR.primal_type(typeof(dual(1.0f0, 1.0f0))) === Float32
    @test CR.primal_type(typeof(dual(dual(1.0, 1.0), dual(0.0, 1.0)))) === Float64
    @test CR.primal(dual(2.5, 1.0)) === 2.5
    @test CR.all_finite(dual(1.0, 2.0)) && !CR.all_finite(dual(1.0, NaN)) && !CR.all_finite(Inf)
    @test CR.exact_zero(dual(-0.0, 0.0)) && !CR.exact_zero(dual(0.0, 1.0))
    @test CR.scalar_equal(==, dual(1.0, 0.0), 1.0) && !CR.scalar_equal(==, dual(1.0, 1.0), 1.0)
    @test CR.scalar_equal(==, dual(1.0, -0.0), dual(1.0, 0.0))
    @test !CR.scalar_equal(isequal, dual(1.0, -0.0), dual(1.0, 0.0))
    # Empty layer: exact zero gives the identity operator; a zero value with a nonzero
    # derivative evaluates the slab (right derivative, see the AD tests)
    r = CanopyRates(0.15, 0.08, 0.04, 0.05, 0.65)
    for m in (AnalyticTwoStream(), NativeTwoStream(0.02, 14))
        @test layer_operator(m, r, 0.0) === LayerOperator(0.0, 1.0, 0.0, 0.0, 1.0)
        @test ForwardDiff.partials(layer_operator(m, r, dual(0.0, 1.0)).beam_up, 1) ≈ r.beam_up
    end
    @test_throws ArgumentError NativeTwoStream(dual(0.05, Inf), 12)
end

@testset "Inference and zero allocations ($FT)" for FT in (Float32, Float64)
    c = converted_calls(FT)
    nat = NativeTwoStream(FT(0.02), 14)
    ana = AnalyticTwoStream()
    x = FT(0.6)
    @test (@inferred CanopyRates(FT(0.1), FT(0.2), FT(0.03), FT(0.04), FT(0.6))) isa CanopyRates{FT}
    @test (@inferred LayerOperator(FT(0.1), FT(0.7), FT(0.03), FT(0.04), FT(0.6))) isa LayerOperator{FT}
    @test (@inferred NativeTwoStream(FT)) isa NativeTwoStream{FT}
    @test (@inferred foursail_canopy_rates(c.geom, FT(0.08), FT(0.06))) isa CanopyRates{FT}
    @test (@inferred foursail_canopy_rates(c.geom, FT(0.08), FT(0.06), FT(0.9))) isa CanopyRates{FT}
    @test (@inferred diffuse_response(FT(0.1), FT(0.2), FT(1.3))).reflection isa FT
    @test (@inferred layer_operator(nat, c.r, FT(1.2))) isa LayerOperator{FT}
    @test (@inferred layer_operator(ana, c.r, FT(1.2))) isa LayerOperator{FT}
    @test (@inferred interior_flux(nat, c.r, FT(1.2), x, FT(0.8), FT(0.1), FT(0.7))).down isa FT
    @test (@inferred interior_flux(ana, c.r, FT(1.2), x, FT(0.8), FT(0.1), FT(0.7))).down isa FT
    @test (@inferred point_emission_response(FT(0.1), FT(0.2), FT(1), x, FT(0.5), FT(0.5))).upward isa FT
    @test (@inferred point_emission_interior(FT(0.1), FT(0.2), FT(1), FT(0.4), x, FT(0.5), FT(0.5))).up isa FT
    @test (@inferred emitting_layer(FT(0.1), FT(0.2), FT(1.5), NoEmission())) isa EmittingLayer{FT}
    @test (@inferred emitting_layer(FT(0.1), FT(0.2), FT(1.5), c.u)) isa EmittingLayer{FT}
    @test (@inferred emitting_layer(FT(0.1), FT(0.2), FT(1.5), c.q)) isa EmittingLayer{FT}
    @test (@inferred emission_interior(FT(0.1), FT(0.2), FT(1.5), x, c.u, FT(0.3), FT(0.1))).up isa FT
    @test (@inferred emission_interior(FT(0.1), FT(0.2), FT(1.5), x, c.q, FT(0.3), FT(0.1))).up isa FT
    ops = [layer_operator(ana, c.r, FT(l)) for l in (0.5, 1.0, 1.5)]
    els = [emitting_layer(FT(0.1), FT(0.2), FT(l), c.u) for l in (0.5, 1.0, 1.5)]
    ws = ColumnFluxWorkspace(FT, 3)
    we = EmittingColumnWorkspace(FT, 3)
    @test (@inferred solve_column!(ws, ops, FT(0.7), FT(0.3), FT(0.15))).ground isa FT
    @test (@inferred solve_emitting_column!(we, els, FT(300), FT(0.05), FT(390))).generated isa FT
    @test (@inferred solve_column((ops...,), FT(0.7), FT(0.3), FT(0.15))).ground isa FT
    @test (@inferred solve_emitting_column((els...,), FT(300), FT(0.05), FT(390))).generated isa FT

    alloc_layer(m, r, l) = @allocated layer_operator(m, r, l)
    alloc_rates(g, a, b) = @allocated foursail_canopy_rates(g, a, b)
    alloc_interior(m, r, l, x) = @allocated interior_flux(m, r, l, x, l, l, l)
    alloc_emit(a, b, l, e) = @allocated emitting_layer(a, b, l, e)
    alloc_eint(a, b, l, x, e) = @allocated emission_interior(a, b, l, x, e, l, l)
    alloc_col!(ws, ops, d) = @allocated solve_column!(ws, ops, d, d, d)
    alloc_ecol!(ws, els, d) = @allocated solve_emitting_column!(ws, els, d, d, d)
    measure() = (
        alloc_layer(nat, c.r, FT(1.2)),
        alloc_layer(ana, c.r, FT(1.2)),
        alloc_rates(c.geom, FT(0.08), FT(0.06)),
        alloc_interior(nat, c.r, FT(1.2), x),
        alloc_interior(ana, c.r, FT(1.2), x),
        alloc_emit(FT(0.1), FT(0.2), FT(1.5), c.u),
        alloc_emit(FT(0.1), FT(0.2), FT(1.5), c.q),
        alloc_eint(FT(0.1), FT(0.2), FT(1.5), x, c.u),
        alloc_eint(FT(0.1), FT(0.2), FT(1.5), x, c.q),
        alloc_col!(ws, ops, FT(0.2)),
        alloc_ecol!(we, els, FT(0.2)),
    )
    measure()
    @test all(iszero, measure())
end

@testset "Record equality, hashing and broadcasting" begin
    r = CanopyRates(0.1, 0.2, 0.05, 0.05, 0.5)
    op = LayerOperator(0.1, 0.2, 0.05, 0.05, 0.5)
    u = UniformEmission(0.0, 0.2)
    el = EmittingLayer(op, 0.0, 0.2, 0.4)
    q = QuadratureEmission([0.0, 0.5], [0.5, 0.5], [0.0, 0.2], [0.1, 0.0])
    records = (
        (NativeTwoStream(0.05, 12), NativeTwoStream(0.05, 13)),
        (AnalyticTwoStream(), NoEmission()),
        (r, CanopyRates(0.1, 0.2, 0.05, 0.05, 0.6)),
        (op, LayerOperator(0.1, 0.2, 0.05, 0.06, 0.5)),
        (u, UniformEmission(0.0, 0.3)),
        (el, EmittingLayer(op, 0.0, 0.2, 0.5)),
        (q, QuadratureEmission([0.0, 0.5], [0.5, 0.5], [0.0, 0.2], [0.1, 0.1])),
    )
    for (a, b) in records
        @test a == a && isequal(a, a) && hash(a) == hash(a)
        @test a != b && !isequal(a, b)
        @test Base.broadcastable(a) isa Ref
    end
    # Same field values, different record kinds
    @test r != op && !isequal(r, op)
    @test AnalyticTwoStream() == AnalyticTwoStream() && NoEmission() == NoEmission()
    # -0.0 == 0.0 with equal hashes; isequal distinguishes them
    for (pos, neg) in (
        (u, UniformEmission(-0.0, 0.2)),
        (el, EmittingLayer(op, -0.0, 0.2, 0.4)),
        (q, QuadratureEmission([-0.0, 0.5], [0.5, 0.5], [-0.0, 0.2], [0.1, -0.0])),
    )
        @test pos == neg && hash(pos) == hash(neg)
        @test !isequal(pos, neg)
    end
    # NaN != NaN, isequal(NaN, NaN), consistent hash
    un = UniformEmission(NaN, 0.2)
    @test un != un && isequal(un, un) && hash(un) == hash(UniformEmission(NaN, 0.2))
    # Duals: zero derivative equals the plain record (same hash); a nonzero derivative
    # does not, on every ForwardDiff version
    @test CanopyRates(dual(0.1, 0.0), 0.2, 0.05, 0.05, 0.5) == r
    @test hash(CanopyRates(dual(0.1, 0.0), 0.2, 0.05, 0.05, 0.5)) == hash(r)
    @test CanopyRates(dual(0.1, 1.0), 0.2, 0.05, 0.05, 0.5) != r
    @test !isequal(CanopyRates(dual(0.1, 1.0), 0.2, 0.05, 0.05, 0.5), r)
    @test QuadratureEmission(dual.([0.0, 0.5], 1.0), dual.([0.5, 0.5], 0.0), dual.([0.0, 0.2], 0.0), dual.([0.1, 0.0], 0.0)) != q
    # Array fields: axes are part of the value
    qo = QuadratureEmission(OffsetArray([0.0, 0.5], 0:1), OffsetArray([0.5, 0.5], 0:1),
        OffsetArray([0.0, 0.2], 0:1), OffsetArray([0.1, 0.0], 0:1))
    @test qo != q && hash(qo) != hash(q)
    @test qo == QuadratureEmission(OffsetArray([0.0, 0.5], 0:1), OffsetArray([0.5, 0.5], 0:1),
        OffsetArray([0.0, 0.2], 0:1), OffsetArray([0.1, 0.0], 0:1))
    # Scalar records: inferred, allocation free
    eq(a, b) = @allocated a == b
    ie(a, b) = @allocated isequal(a, b)
    hs(a) = @allocated hash(a)
    for (a, b) in records[1:6]
        @test (@inferred a == b) isa Bool
        @test (@inferred isequal(a, b)) isa Bool
        @test (@inferred hash(a)) isa UInt
        eq(a, b), ie(a, b), hs(a)
        @test eq(a, b) == 0 && ie(a, b) == 0 && hs(a) == 0
    end
    # Records broadcast as scalars, QuadratureEmission included
    @test layer_operator.(AnalyticTwoStream(), r, [0.5, 1.0]) ==
          [layer_operator(AnalyticTwoStream(), r, l) for l in (0.5, 1.0)]
    @test emitting_layer.(0.1, 0.2, [0.5, 1.0], q) == [emitting_layer(0.1, 0.2, l, q) for l in (0.5, 1.0)]
    # eltype of records and workspaces
    @test eltype(r) === Float64 && eltype(NativeTwoStream(0.05f0, 3)) === Float32
    @test eltype(q) === Float64 && eltype(el) === Float64 && eltype(ColumnFluxWorkspace(Float32, 2)) === Float32
end

# ============================================================================
# Leaf faces, populations, emission laws, atmosphere coupling, leaf spectral
# emission, hemispheric moments and SIF assembly
# ============================================================================

function api_objects(FT)
    lem = LeafEmissionMatrix(FT[0.3 0.2; 0.2 0.3], FT[0.2 0.2; 0.1 0.2], FT[0.1 0.1; 0.2 0.1], FT[0.1, 0.2])
    lhs = LayerHemisphericSource(FT(1.2), FT[0.25, 0.75], FT[0.5, 0.5], fill(FT(0.1), 2, 2), fill(FT(0.2), 2, 2), FT[0.4, 0.5])
    return (;
        pq = PopulationQuadrature(1, FT(1.0), FT(0.0), FT[0.5], FT[1.0], FT[0.4], FT[0.8], PairedIllumination()),
        law = MixedLambertianFaces(FT(0.7)),
        bnd = AngularEmissionBoundary(FT[0.1;;], FT[0.5;;], FT[0.7], FT[1.0], 0),
        atm = AngularAtmosphericTransfer(FT[0.05;;], FT[0.9;;], FT[0.7], FT[1.0], 0),
        lem,
        dlem = DualLeafEmissionMatrix(lem, lem),
        lhs,
        dlhs = DualLayerHemisphericSource(lhs, lhs),
        ism = IdentitySpectralMap(FT[400, 500]),
        lsm = LinearSpectralMap(FT[400, 600], FT[400, 500, 600]),
    )
end

@testset "Precision policy: leaf faces, populations, emission laws, coupling, SIF" begin
    o64 = api_objects(Float64)
    o32 = api_objects(Float32)
    mu = [0.3, 0.8]
    w3 = reshape([0.2, 0.4, 0.3, 0.5], 2, 1, 2)
    for bad in (1.0f0, 1, true, 1 // 2, pi)
        bad isa AbstractFloat || @test_throws ArgumentError CanopyRadiation.unit_sin_from_cos(bad)
        @test_throws ArgumentError source_projection(0.5, 0.1, bad, 0.2)
        @test_throws ArgumentError beam_face_absorption(0.5, bad, 0.8)
        @test_throws ArgumentError isotropic_face_absorption(0.4, 200.0, bad, 0.8)
        @test_throws ArgumentError leaf_hemispheric_emission(LambertianFaces(), 0.5, bad, 0.2)
        @test_throws ArgumentError leaf_hemispheric_emission(o64.law, 0.5, 0.6, bad)
        @test_throws ArgumentError leaf_angular_emission(o64.law, 0.5, 0.1, bad, 0.0, 0.6, 0.2)
        @test_throws ArgumentError leaf_angular_emission(LambertianFaces(), 0.5, 0.1, 0.7, bad, 0.6, 0.2; mirrored_azimuth = true)
        @test_throws ArgumentError depth_hemispheric_source(o64.law, PairedIllumination(), mu, fill(0.25, 4), ones(4), ones(4), 1, bad)
        @test_throws ArgumentError depth_angular_source(LambertianFaces(), PairedIllumination(), mu, [0.1, 0.2], fill(0.25, 4), ones(4), ones(4), 1, 1.0, bad, 0.0)
        @test_throws ArgumentError hemispheric_moment(o64.law, mu, Float32.(w3), 1, 1)
        @test_throws ArgumentError evaluate_hemispheric_sif_bin([o64.lhs], o64.lem, [0.1], [0.2], bad, 1)
        bad isa AbstractFloat || @test_throws ArgumentError MixedLambertianFaces(bad)
        bad isa AbstractFloat || @test_throws ArgumentError orientation_hemisphere_samples(spherical_leaves(), bad)
    end
    # Float32 records with Float64 inputs
    @test_throws ArgumentError leaf_hemispheric_emission(o32.law, 0.5, 0.6, 0.2)
    @test_throws ArgumentError couple_emission(o64.bnd, o32.atm)
    @test_throws ArgumentError one_way_emission(o32.bnd, o64.atm)
    @test_throws ArgumentError hemispheric_bin(ones(1, 2), ones(1, 2), Float32[1, 1], Float32[1, 1], 1)
    @test_throws ArgumentError hemispheric_source_moments!(zeros(1, 2), zeros(1, 2), o32.law, mu, w3)
    @test_throws ArgumentError expand_hemispheric_source!(zeros(2), zeros(2), ones(2, 2), ones(2, 2), Float32[1, 1], Float32[1, 1])
    @test_throws ArgumentError batch_hemispheric_moments!(zeros(1, 2, 1), zeros(1, 2, 1), o32.law, mu, reshape(w3, 2, 1, 2, 1))
    @test_throws ArgumentError batch_expand_hemispheric_sources!(zeros(2, 1), zeros(2, 1), ones(2, 2), ones(2, 2), Float32[1 1], Float32[1 1])
    @test_throws ArgumentError evaluate_hemispheric_sif_bin([o32.lhs], o64.lem, [0.1], [0.2], 0.1, 1)
    @test_throws ArgumentError evaluate_hemispheric_sif_spectrum([o64.lhs], o32.lem, fill(0.1, 2, 1), fill(0.2, 2, 1), [0.1, 0.1])
    @test_throws ArgumentError assemble_hemispheric_emission!(zeros(Int, 1), zeros(Int, 1), LambertianFaces(), o64.pq, fill(0.5, 2), ones(2), ones(2))
    # Output element type must equal the promoted input type
    @test_throws ArgumentError hemispheric_source_moments!(zeros(1, 2), zeros(1, 2), MixedLambertianFaces(dual(0.7, 1.0)), mu, w3)
    # Integer storage and quadrature types
    @test_throws ArgumentError LeafEmissionMatrix([1;;], [0;;], [0;;])
    @test_throws ArgumentError AngularEmissionBoundary([0;;], [1;;], [1], [1], 0)
    @test_throws ArgumentError LayerHemisphericSource(1, [0], [1], [0;;], [0;;], [1])
    @test_throws ArgumentError PopulationQuadrature(1, 1, 0, [0], [1], [0], [1], PairedIllumination())
    @test_throws ArgumentError leaf_inclination_quadrature(spherical_leaves(), 4, Int)
    @test_throws ArgumentError photon_energy!(zeros(Int, 1, 1), ones(Int, 1, 1), ones(Int, 1), 1)
    @test_throws ArgumentError escape_leaf_emission!(zeros(Int, 1, 1), zeros(Int, 1, 1), zeros(Int, 1, 1), NoLeafEmission())
    # Static validation messages unchanged
    @test_throws ArgumentError("lambertian_fraction must be finite and lie in [0, 1]") MixedLambertianFaces(1.5)
    @test_throws ArgumentError("internal_psi matrix required for active PSI LeafEmissionMatrix") escape_dual_leaf_emission!(
        zeros(2, 1), zeros(2, 1), zeros(2, 1), o64.dlem, ones(2, 1))
    @test_throws ArgumentError("Unsupported out_of_window_fraction specification") CanopyRadiation._tail_fractions(
        Float32[0.1], Float64, 1, StrictStokes(), [440.0], 600.0)
end

@testset "Precision policy: plain and dual inputs promote (faces, laws, coupling)" begin
    D = typeof(dual(1.0, 1.0))
    @test source_projection(dual(0.5, 1.0), 0.1, 0.7, 0.2) isa D
    @test beam_face_absorption(0.5, 1000.0, dual(0.85, 1.0)).front isa D
    @test leaf_hemispheric_emission(MixedLambertianFaces(0.7), dual(0.5, 1.0), 0.6, 0.2).upward isa D
    @test leaf_hemispheric_emission(MixedLambertianFaces(dual(0.7, 1.0)), 0.5, 0.6, 0.2).upward isa D
    b = AngularEmissionBoundary([0.2;;], [1.0;;], [0.7], [1.0], 0)
    a = AngularAtmosphericTransfer([dual(0.1, 1.0);;], [dual(0.8, 0.0);;], [dual(0.7, 0.0)], [dual(1.0, 0.0)], 0)
    c = couple_emission(b, a)
    @test eltype(c.top_up) === D
    @test ForwardDiff.value(c.canopy_up[1, 1]) == couple_emission(b, AngularAtmosphericTransfer([0.1;;], [0.8;;], [0.7], [1.0], 0)).canopy_up[1, 1]
    @test eltype(one_way_emission(b, a)) === D
    @test hemispheric_bin([0.3 0.5], [0.2 0.4], [dual(0.4, 1.0), 0.2], [0.1, 0.3], 1).up isa D
end

@testset "Inference and allocations, leaf faces to SIF ($FT)" for FT in (Float32, Float64)
    o = api_objects(FT)
    mu = FT[0.3, 0.8]
    w3 = reshape(FT[0.2, 0.4, 0.3, 0.5], 2, 1, 2)
    wpop, fr, bk = fill(FT(0.25), 4), FT[0.9, 0.6, 0.8, 0.5], FT[0.4, 0.3, 0.5, 0.2]
    plate = diffuse_plate_optics(FT(0.15), FT(1.4))
    @test (@inferred CanopyRadiation.unit_sin_from_cos(FT(0.6))) isa FT
    @test (@inferred source_projection(FT(0.5), FT(0.1), FT(0.7), FT(0.2))) isa FT
    @test (@inferred beam_face_absorption(FT(0.5), FT(900), FT(0.85))).front isa FT
    @test (@inferred isotropic_face_absorption(FT(0.4), FT(200), FT(50), FT(0.8))).back isa FT
    @test (@inferred regular_plate_passive(plate)).absorptance isa FT
    @test (@inferred population_coordinates(o.pq, 2)).depth_lai isa FT
    @test (@inferred leaf_hemispheric_emission(LambertianFaces(), FT(0.5), FT(0.6), FT(0.2))).upward isa FT
    @test (@inferred leaf_hemispheric_emission(o.law, FT(0.5), FT(0.6), FT(0.2))).upward isa FT
    @test (@inferred leaf_angular_emission(o.law, FT(0.5), FT(0.1), FT(0.7), FT(0), FT(0.6), FT(0.2))) isa FT
    @test (@inferred depth_hemispheric_source(o.law, PairedIllumination(), mu, wpop, fr, bk, 1, FT(1))).upward isa FT
    @test (@inferred depth_angular_source(o.law, BlockedIllumination(), mu, FT[0.1, 0.2], wpop, fr, bk, 1, FT(1), FT(0.8), FT(0.2))) isa FT
    @test (@inferred hemispheric_moment(o.law, mu, w3, 1, 1)).plus isa FT
    @test (@inferred hemispheric_bin(fill(FT(0.2), 2, 2), fill(FT(0.1), 2, 2), FT[0.4, 0.2], FT[0.1, 0.3], 1)).up isa FT
    @test (@inferred couple_emission(o.bnd, o.atm)).top_up isa Matrix{FT}
    @test eltype((@inferred evaluate_hemispheric_sif_bin([o.lhs], o.lem, FT[0.1], FT[0.2], FT(0.1), 1)).up) === FT
    @test eltype((@inferred evaluate_hemispheric_sif_bin([o.dlhs], o.dlem, FT[0.1], FT[0.2], FT(0.1), 1)).up) === FT

    alloc_proj(a, b, c, d) = @allocated source_projection(a, b, c, d)
    alloc_beam(a, b, c) = @allocated beam_face_absorption(a, b, c)
    alloc_hemi(law, a, b, c) = @allocated leaf_hemispheric_emission(law, a, b, c)
    alloc_ang(law, a, b, c, d) = @allocated leaf_angular_emission(law, a, b, c, d, a, b; mirrored_azimuth = true)
    alloc_depth(law, mu, w, f, b) = @allocated depth_hemispheric_source(law, PairedIllumination(), mu, w, f, b, 1, one(eltype(mu)))
    alloc_moment(law, mu, w3) = @allocated hemispheric_moment(law, mu, w3, 1, 1)
    alloc_energy(o, p, w, c) = @allocated photon_energy!(o, p, w, c)
    alloc_bins(o, p, w) = @allocated photon_bins!(o, p, w)
    out, ph, wl = zeros(FT, 2, 3), ones(FT, 2, 3), FT[680, 740]
    measure() = (
        alloc_proj(FT(0.5), FT(0.1), FT(0.7), FT(0.2)),
        alloc_beam(FT(0.5), FT(900), FT(0.85)),
        alloc_hemi(o.law, FT(0.5), FT(0.6), FT(0.2)),
        alloc_ang(o.law, FT(0.5), FT(0.1), FT(0.7), FT(0.3)),
        alloc_depth(o.law, mu, wpop, fr, bk),
        alloc_moment(o.law, mu, w3),
        alloc_energy(out, ph, wl, FT(0.1196)),
        alloc_bins(out, ph, wl),
    )
    measure()
    @test all(iszero, measure())
end

# An equal object with freshly allocated arrays.
fresh_copy(m::IdentitySpectralMap) = IdentitySpectralMap(copy(source_grid(m)))
fresh_copy(m::LinearSpectralMap) = LinearSpectralMap(copy(source_grid(m)), copy(target_grid(m)))
fresh_copy(x) = CanopyRadiation._rebuild(x, map(f -> f isa AbstractArray ? copy(f) : f, CanopyRadiation._fields(x)))

@testset "Record equality, hashing and broadcasting: populations to SIF" begin
    o = api_objects(Float64)
    other = (
        pq = PopulationQuadrature(1, 1.0, 0.0, [0.5], [1.0], [0.4], [0.9], PairedIllumination()),
        law = MixedLambertianFaces(0.6),
        bnd = AngularEmissionBoundary([0.1;;], [0.6;;], [0.7], [1.0], 0),
        atm = AngularAtmosphericTransfer([0.05;;], [0.9;;], [0.7], [1.0], 1),
        lem = LeafEmissionMatrix([0.3 0.2; 0.2 0.3], [0.2 0.2; 0.1 0.2], [0.1 0.1; 0.2 0.1], [0.1, 0.3]),
        dlem = DualLeafEmissionMatrix(o.lem, NoLeafEmission()),
        lhs = LayerHemisphericSource(1.3, [0.25, 0.75], [0.5, 0.5], fill(0.1, 2, 2), fill(0.2, 2, 2), [0.4, 0.5]),
        dlhs = DualLayerHemisphericSource(o.lhs, LayerHemisphericSource(1.2, [0.25, 0.75], [0.5, 0.5], fill(0.1, 2, 2), fill(0.2, 2, 2), [0.4, 0.6])),
        ism = IdentitySpectralMap([400.0, 501.0]),
        lsm = LinearSpectralMap([400.0, 600.0], [400.0, 550.0, 600.0]),
    )
    for k in keys(o)
        a, b = o[k], other[k]
        @test a == a && isequal(a, a) && hash(a) == hash(a)
        @test a != b && !isequal(a, b)
        @test Base.broadcastable(a) isa Ref
        # an equal copy (fresh arrays) is equal and hashes equal
        c = fresh_copy(a)
        @test c == a && isequal(c, a) && hash(c) == hash(a)
    end
    for (x, y) in ((PairedIllumination(), BlockedIllumination()), (LambertianFaces(), VacuumAtmosphere()), (NoLeafEmission(), LambertianFaces()))
        @test x == x && x != y && hash(x) != hash(y)
    end
    # -0.0 == 0.0 with equal hashes; isequal distinguishes them; NaN
    z_pos = AngularEmissionBoundary([0.0;;], [0.5;;], [0.7], [1.0], 0)
    z_neg = AngularEmissionBoundary([-0.0;;], [0.5;;], [0.7], [1.0], 0)
    @test z_pos == z_neg && hash(z_pos) == hash(z_neg) && !isequal(z_pos, z_neg)
    l_pos, l_neg = MixedLambertianFaces(0.0), MixedLambertianFaces(-0.0)
    @test l_pos == l_neg && hash(l_pos) == hash(l_neg) && !isequal(l_pos, l_neg)
    nan_lem = LeafEmissionMatrix([0.2;;], [0.2;;], [0.1;;], [0.1])
    nan_lem.front[1, 1] = NaN
    @test nan_lem != nan_lem && isequal(nan_lem, nan_lem)
    # Axes are part of an array-holding record's value
    q1 = PopulationQuadrature(1, 1.0, 0.0, OffsetArray([0.5], 2:2), OffsetArray([1.0], 2:2), [0.4], [0.8], PairedIllumination())
    q2 = PopulationQuadrature(1, 1.0, 0.0, [0.5], [1.0], [0.4], [0.8], PairedIllumination())
    @test q1 != q2 && hash(q1) != hash(q2)
    # Duals: zero derivative equals the plain record; a nonzero derivative does not
    @test MixedLambertianFaces(dual(0.7, 0.0)) == o.law && hash(MixedLambertianFaces(dual(0.7, 0.0))) == hash(o.law)
    @test MixedLambertianFaces(dual(0.7, 1.0)) != o.law
    # Records broadcast as scalars
    @test leaf_hemispheric_emission.(o.law, [0.2, 0.4], 0.6, 0.2) ==
          [leaf_hemispheric_emission(o.law, m, 0.6, 0.2) for m in (0.2, 0.4)]
    # adapt_radiation_storage to Array returns an equal object of the same kind
    # (all eleven kinds; field order must match the positional constructors)
    ws_col = ColumnFluxWorkspace(Float64, 2)
    ws_emit = EmittingColumnWorkspace(Float64, 2)
    ws_col.down .= 1.0:3.0
    ws_emit.net_layer .= [0.5, -0.5]
    kinds = (o.bnd, o.atm, o.lem, o.dlem, o.lhs, o.dlhs, NoLeafEmission(), VacuumAtmosphere(),
        QuadratureEmission([0.5], [1.0], [0.3], [0.2]), ws_col, ws_emit)
    @test length(unique(map(x -> nameof(typeof(x)), kinds))) == 11
    for x in kinds
        y = adapt_radiation_storage(Array, x)
        @test typeof(y) === typeof(x)
        @test y == x
    end
    # Objects constructible only without value checks keep their values
    q_raw = QuadratureEmission([1.5], [-1.0], [0.3], [0.2]; validate = false)
    l_raw = LayerHemisphericSource(1.0, [1.5], [1.0], [-0.1;;], [0.1;;], [0.4]; validate = false)
    @test adapt_radiation_storage(Array, q_raw) == q_raw
    @test adapt_radiation_storage(Array, l_raw) == l_raw
    # Ranges become arrays of the target type with the same element type
    rm = reshape(0.05:0.05:0.2, 2, 2)
    r = adapt_radiation_storage(Array, LeafEmissionMatrix(rm, rm, rm, 0.05:0.05:0.1))
    @test r.front isa Matrix{Float64} && r.out_of_window isa Vector{Float64}
    @test r == LeafEmissionMatrix(collect(rm), collect(rm), collect(rm), [0.05, 0.1])
end

@testset "Abstract element types, resized workspaces, nested photosystems, dual LAI" begin
    # Abstract floating-point element types could mix Float32 and Float64
    af(x...) = AbstractFloat[x...]
    @test_throws ArgumentError QuadratureEmission(af(0.5), af(1.0), af(0.3), af(0.2))
    @test_throws ArgumentError LayerHemisphericSource(1.0, af(0.5), af(1.0), AbstractFloat[0.1;;], AbstractFloat[0.1;;], af(0.4))
    @test_throws ArgumentError ColumnFluxWorkspace(AbstractFloat, 2)
    @test_throws ArgumentError EmittingColumnWorkspace(AbstractFloat, 2)
    m(x) = Matrix{AbstractFloat}(fill(x, 1, 2))
    v(x) = Vector{AbstractFloat}(fill(x, 2))
    @test_throws ArgumentError batch_solve_columns!(m(0.0), m(0.0), m(0.0), Matrix{AbstractFloat}(undef, 0, 2), v(0.0),
        Matrix{AbstractFloat}(undef, 0, 2), Matrix{AbstractFloat}(undef, 0, 2), Matrix{AbstractFloat}(undef, 0, 2),
        Matrix{AbstractFloat}(undef, 0, 2), Matrix{AbstractFloat}(undef, 0, 2), Vector{AbstractFloat}([0.7, 0.7f0]), v(0.3), v(0.1))
    @test_throws ArgumentError CanopyRates{AbstractFloat}(0.1, 0.2f0, 0.04, 0.05, 0.6)
    @test_throws ArgumentError LayerOperator{AbstractFloat}(0.1, 0.6f0, 0.04, 0.05, 0.5)
    @test_throws ArgumentError UniformEmission{AbstractFloat}(0.1, 0.2f0)
    @test_throws ArgumentError PopulationQuadrature(1, 1.0, 0.0, af(0.5), af(1.0), af(0.4), af(0.8), PairedIllumination())
    # Interface vectors resized after construction are caught before the unchecked sweep
    ops = [layer_operator(AnalyticTwoStream(), CanopyRates(0.1, 0.2, 0.04, 0.05, 0.6), 1.0) for _ in 1:2]
    ws = ColumnFluxWorkspace(Float64, 2)
    resize!(ws.down, 2)
    @test_throws DimensionMismatch solve_column!(ws, ops, 0.7, 0.3, 0.1)
    els = [emitting_layer(0.1, 0.2, 1.0, UniformEmission(0.3, 0.2)) for _ in 1:2]
    we = EmittingColumnWorkspace(Float64, 2)
    resize!(we.up, 1)
    @test_throws DimensionMismatch solve_emitting_column!(we, els, 300.0, 0.05, 390.0)
    we2 = EmittingColumnWorkspace(Float64, 2)
    resize!(we2.net_layer, 1)
    @test_throws DimensionMismatch solve_emitting_column!(we2, els, 300.0, 0.05, 390.0)
    # A DualLeafEmissionMatrix holds two single photosystems, not another pair
    lem = LeafEmissionMatrix([0.3;;], [0.3;;], [0.3;;], [0.1])
    inner = DualLeafEmissionMatrix(lem, NoLeafEmission())
    @test_throws ArgumentError DualLeafEmissionMatrix(inner, lem)
    @test_throws ArgumentError DualLeafEmissionMatrix(NoLeafEmission(), inner)
    # PSII and PSI layer sources must share LAI in value and derivative components,
    # identically on every ForwardDiff version; plain floats as before
    mk(lai) = LayerHemisphericSource(lai, [0.5], [1.0], [0.1;;], [0.1;;], [0.4])
    mkd(lai) = LayerHemisphericSource(lai, [dual(0.5, 0.0)], [dual(1.0, 0.0)], [dual(0.1, 0.0);;], [dual(0.1, 0.0);;], [dual(0.4, 0.0)])
    @test DualLayerHemisphericSource(mk(1.0), mk(1.0)).psii.lai == 1.0
    @test DualLayerHemisphericSource(mk(-0.0 + 1.0), mk(1.0)) isa DualLayerHemisphericSource
    @test_throws ArgumentError DualLayerHemisphericSource(mk(1.0), mk(1.1))
    @test DualLayerHemisphericSource(mkd(dual(1.0, 1.0)), mkd(dual(1.0, 1.0))) isa DualLayerHemisphericSource
    @test_throws ArgumentError DualLayerHemisphericSource(mkd(dual(1.0, 1.0)), mkd(dual(1.0, 0.0)))
end

@testset "Allocation-free equality of host-array records; scalar hooks on duals" begin
    mkq() = QuadratureEmission(OffsetArray([0.25, 0.75], 0:1), OffsetArray([0.5, 0.5], 0:1),
        OffsetArray([0.3, 0.2], 0:1), OffsetArray([0.3, 0.2], 0:1))
    mkl() = LeafEmissionMatrix(view([0.3 0.2 9.0; 0.2 0.3 9.0], :, 1:2), view([0.2 0.2 9.0; 0.1 0.2 9.0], :, 1:2),
        view([0.1 0.1 9.0; 0.2 0.1 9.0], :, 1:2), [0.1, 0.2])
    eq(a, b) = @allocated a == b
    ie(a, b) = @allocated isequal(a, b)
    hs(a) = @allocated hash(a)
    for (a, b) in ((mkq(), mkq()), (mkl(), mkl()))
        @test a == b && isequal(a, b) && hash(a) == hash(b)
        eq(a, b), ie(a, b), hs(a)
        @test eq(a, b) == 0 && ie(a, b) == 0 && hs(a) == 0
    end
    a, b, nd = dual(1.0, 2.0), dual(1.0, -0.0), dual(dual(1.0, 1.0), dual(0.0, 1.0))
    af(x) = @allocated CR.all_finite(x)
    ez(x) = @allocated CR.exact_zero(x)
    se(c, x, y) = @allocated CR.scalar_equal(c, x, y)
    for _ in 1:2
        global hook_allocs = (af(a), ez(b), se(==, a, b), se(isequal, a, 1.0), af(nd), se(==, nd, nd))
    end
    @test all(iszero, hook_allocs)
    @test (@inferred CR.all_finite(nd)) && (@inferred CR.exact_zero(dual(-0.0, 0.0)))
    @test !(@inferred CR.scalar_equal(==, a, b)) && (@inferred CR.scalar_equal(==, b, 1.0))
    @test !(@inferred CR.scalar_equal(isequal, b, 1.0))
    @test (@inferred CR.scalar_equal(==, nd, nd))
end
