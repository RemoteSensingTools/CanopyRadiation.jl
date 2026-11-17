# Frozen numerical record test.
#
# Recomputes every case in test/frozen_record/inputs.toml and compares it bit
# for bit with test/frozen_record/reference.toml, which was produced from the
# unmodified CanopyRadiation v0.1.0 source (commit 33f015b) by
# test/frozen_record/generate.jl. The reference must NOT be regenerated from
# refactored code; regeneration is a deliberate, reviewed act.
#
# Each code path (tuple recursion, vector workspace, KernelAbstractions batch)
# has its own reference table, so a change confined to one path is caught even
# if the paths still agree with each other.

using CanopyRadiation
using Test
using TOML

isdefined(@__MODULE__, :FrozenRecord) || include(joinpath(@__DIR__, "frozen_record", "compute.jl"))

@testset "Frozen v0.1.0 (33f015b) numerical record, bit for bit" begin
    fx = FrozenRecord.load_fixture()
    ref = FrozenRecord.load_reference()
    # Empty on hosts with native FMA (this is where the record was made, and
    # CI runners); see test/frozen_record/host_tolerances.toml.
    tolerances = FrozenRecord.host_tolerances()
    @test ref["meta"]["frozen_commit"] == "33f015b"
    @test ref["meta"]["source_matches_frozen_commit"] === true
    total = 0
    for pname in fx["precisions"]
        FT = FrozenRecord.precision_type(pname)
        @testset "$pname" begin
            rec = FrozenRecord.compute_record(FT, fx)
            mismatches, missing_keys, unmatched, checked, tolerated =
                FrozenRecord.compare_record(rec, ref; tolerances)
            isempty(tolerated) ||
                @warn "Host without native FMA: $(length(tolerated)) $pname value(s) differ from the " *
                      "frozen record within the documented host tolerance\n" * join(tolerated, "\n")
            total += FrozenRecord.record_value_count(rec)
            if !isempty(mismatches)
                shown = first(mismatches, 50)
                @error "Frozen record mismatch in $pname: $(length(mismatches)) value(s) differ" *
                       (length(mismatches) > 50 ? " (first 50 shown)" : "") * "\n" * join(shown, "\n")
            end
            isempty(missing_keys) ||
                @error "Computed tables missing from reference.toml ($pname):\n" * join(missing_keys, "\n")
            isempty(unmatched) ||
                @error "Reference tables never recomputed ($pname):\n" * join(unmatched, "\n")
            @test isempty(mismatches)
            @test isempty(missing_keys)
            @test isempty(unmatched)
            @test checked > 0

            # Multi-workgroup (threaded) CPU batch launches reproduce the
            # single-copy batch result bit for bit in every replica.
            R = fx["batch_replicas"]
            same_bits(a, b) = size(a) == size(b) && all(
                FrozenRecord.bitstring_hex(x) == FrozenRecord.bitstring_hex(y) for (x, y) in zip(a, b))
            tiled(x::AbstractMatrix) = repeat(x, 1, R)
            tiled(x::AbstractVector) = repeat(x, R)
            for col in fx["passive_columns"]
                inp = FrozenRecord.passive_batch_inputs(FT, fx, col)
                base = FrozenRecord.run_passive_batch(FT, inp)
                rep = FrozenRecord.run_passive_batch(FT, FrozenRecord.replicate_batch(inp, R))
                @test all(same_bits(tiled(b), r) for (b, r) in zip(base, rep))
            end
            for col in fx["emitting_columns"]
                inp = FrozenRecord.emitting_batch_inputs(FT, fx, col)
                base = FrozenRecord.run_emitting_batch(FT, inp)
                rep = FrozenRecord.run_emitting_batch(FT, FrozenRecord.replicate_batch(inp, R))
                @test all(same_bits(tiled(b), r) for (b, r) in zip(base, rep))
            end
        end
    end
    @test total == ref["meta"]["value_count"]
end
