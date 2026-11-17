# Frozen numerical record test, extension: remaining public API.
#
# Recomputes the api_* and canopyoptics_* cases of
# test/frozen_record/inputs.toml and compares them bit for bit with
# test/frozen_record/reference_api.toml and
# test/frozen_record/reference_canopyoptics.toml, which were produced from the
# unmodified CanopyRadiation v0.1.0 source (commit 33f015b) by
# test/frozen_record/generate.jl. The references must NOT be regenerated from
# refactored code; regeneration is a deliberate, reviewed act.
#
# reference_canopyoptics.toml holds every value that CanopyOptics code helps
# compute. It is compared only when the loaded CanopyOptics has the version
# recorded in its [meta]; otherwise the test says so once and skips it.

using CanopyOptics
using CanopyRadiation
using OffsetArrays
using Test
using TOML

isdefined(@__MODULE__, :FrozenRecord) || include(joinpath(@__DIR__, "frozen_record", "compute.jl"))

function frozen_record_report(label, pname, mismatches, missing_keys, unmatched, tolerated)
    isempty(tolerated) ||
        @warn "Host without native FMA: $(length(tolerated)) $pname $label value(s) differ from the " *
              "frozen record within the documented host tolerance\n" * join(tolerated, "\n")
    if !isempty(mismatches)
        shown = first(mismatches, 50)
        @error "Frozen $label record mismatch in $pname: $(length(mismatches)) value(s) differ" *
               (length(mismatches) > 50 ? " (first 50 shown)" : "") * "\n" * join(shown, "\n")
    end
    isempty(missing_keys) ||
        @error "Computed $label tables missing from the reference ($pname):\n" * join(missing_keys, "\n")
    isempty(unmatched) ||
        @error "Reference $label tables never recomputed ($pname):\n" * join(unmatched, "\n")
    return nothing
end

@testset "Frozen v0.1.0 (33f015b) record, remaining public API, bit for bit" begin
    fx = FrozenRecord.load_fixture()
    ref = FrozenRecord.load_reference(FrozenRecord.REFERENCE_API_PATH)
    tolerances = FrozenRecord.host_tolerances()
    @test ref["meta"]["frozen_commit"] == "33f015b"
    @test ref["meta"]["source_matches_frozen_commit"] === true
    total = 0
    for pname in fx["precisions"]
        FT = FrozenRecord.precision_type(pname)
        @testset "$pname" begin
            rec = FrozenRecord.compute_api_record(FT, fx)
            mismatches, missing_keys, unmatched, checked, tolerated =
                FrozenRecord.compare_record(rec, ref; tolerances)
            total += FrozenRecord.record_value_count(rec)
            frozen_record_report("API", pname, mismatches, missing_keys, unmatched, tolerated)
            @test length(mismatches) == 0  # details are in the @error above
            @test isempty(missing_keys)
            @test isempty(unmatched)
            @test checked > 0
        end
    end
    @test total == ref["meta"]["value_count"]
end

@testset "Frozen v0.1.0 (33f015b) record, CanopyOptics-dependent tables" begin
    fx = FrozenRecord.load_fixture()
    ref = FrozenRecord.load_reference(FrozenRecord.REFERENCE_CANOPYOPTICS_PATH)
    meta = ref["meta"]
    prov = FrozenRecord.canopyoptics_provenance()
    @test meta["frozen_commit"] == "33f015b"
    @test meta["source_matches_frozen_commit"] === true
    if prov.version != meta["canopyoptics_version"]
        @warn "CanopyOptics-dependent frozen tables NOT compared: loaded CanopyOptics " *
              "$(prov.version) (commit $(prov.commit)) differs from the recorded " *
              "$(meta["canopyoptics_version"]) (commit $(meta["canopyoptics_git_commit"]))."
        @test_skip prov.version == meta["canopyoptics_version"]
    else
        commit_note = prov.commit == meta["canopyoptics_git_commit"] ? "" :
            " Note: the loaded CanopyOptics commit $(prov.commit) differs from the recorded " *
            "commit $(meta["canopyoptics_git_commit"]) (same version $(prov.version)); a " *
            "difference here may come from CanopyOptics rather than CanopyRadiation."
        total = 0
        for pname in fx["precisions"]
            FT = FrozenRecord.precision_type(pname)
            @testset "$pname" begin
                rec = FrozenRecord.compute_canopyoptics_record(FT, fx)
                mismatches, missing_keys, unmatched, checked, tolerated =
                    FrozenRecord.compare_record(rec, ref; tolerances = FrozenRecord.host_tolerances())
                total += FrozenRecord.record_value_count(rec)
                if !isempty(mismatches)
                    tables = unique(first(split(m, " [")) for m in mismatches)
                    @error "CanopyOptics-dependent frozen tables differ ($pname): " *
                           "$(length(mismatches)) value(s) in $(length(tables)) table(s):\n  " *
                           join(tables, "\n  ") * "\n" * commit_note
                    @debug "Value-level differences" mismatches
                end
                frozen_record_report("CanopyOptics", pname, String[], missing_keys, unmatched, tolerated)
                @test length(mismatches) == 0  # details are in the @error above
                @test isempty(missing_keys)
                @test isempty(unmatched)
                @test checked > 0
            end
        end
        @test total == meta["value_count"]
    end
end
