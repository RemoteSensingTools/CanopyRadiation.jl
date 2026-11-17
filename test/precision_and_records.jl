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
