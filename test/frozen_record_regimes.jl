# Frozen numerical record test, extension 2: numerical regimes.
#
# Recomputes the regime_* cases of test/frozen_record/inputs.toml and compares
# them bit for bit with test/frozen_record/reference_regimes.toml, which was
# produced from the unmodified CanopyRadiation v0.1.0 source (commit 33f015b)
# by test/frozen_record/generate.jl --part=regimes. The reference must NOT be
# regenerated from refactored code; regeneration is a deliberate, reviewed act.

using CanopyOptics
using CanopyRadiation
using OffsetArrays
using Test
using TOML

isdefined(@__MODULE__, :FrozenRecord) || include(joinpath(@__DIR__, "frozen_record", "compute.jl"))

@testset "Frozen v0.1.0 (33f015b) record, numerical regimes, bit for bit" begin
    fx = FrozenRecord.load_fixture()
    ref = FrozenRecord.load_reference(FrozenRecord.REFERENCE_REGIMES_PATH)
    tolerances = FrozenRecord.host_tolerances()
    @test ref["meta"]["frozen_commit"] == "33f015b"
    @test ref["meta"]["source_matches_frozen_commit"] === true
    total = 0
    for pname in fx["precisions"]
        FT = FrozenRecord.precision_type(pname)
        @testset "$pname" begin
            rec = FrozenRecord.compute_regime_record(FT, fx)
            mismatches, missing_keys, unmatched, checked, tolerated =
                FrozenRecord.compare_record(rec, ref; tolerances)
            total += FrozenRecord.record_value_count(rec)
            isempty(tolerated) ||
                @warn "Host without native FMA: $(length(tolerated)) $pname regime value(s) differ from the " *
                      "frozen record within the documented host tolerance\n" * join(tolerated, "\n")
            if !isempty(mismatches)
                @error "Frozen regime record mismatch in $pname: $(length(mismatches)) value(s) differ" *
                       (length(mismatches) > 50 ? " (first 50 shown)" : "") * "\n" *
                       join(first(mismatches, 50), "\n")
            end
            isempty(missing_keys) ||
                @error "Computed regime tables missing from the reference ($pname):\n" * join(missing_keys, "\n")
            isempty(unmatched) ||
                @error "Reference regime tables never recomputed ($pname):\n" * join(unmatched, "\n")
            @test length(mismatches) == 0  # details are in the @error above
            @test isempty(missing_keys)
            @test isempty(unmatched)
            @test checked > 0

            # The empty-column cases were chosen so that v0.1.0's tuple and
            # vector paths differ; the record pins each path's own value.
            for c in fx["regime_empty_cases"]
                t = ref["$pname/regimes/empty_column/tuple/$(c["name"])"]
                v = ref["$pname/regimes/empty_column/vector/$(c["name"])"]
                @test t["up"] != v["up"] || t["ground"] != v["ground"]
            end
        end
    end
    @test total == ref["meta"]["value_count"]
end
