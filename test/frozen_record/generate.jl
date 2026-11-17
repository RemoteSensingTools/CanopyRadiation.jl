# Generator for the CanopyRadiation frozen numerical record.
#
# reference.toml was produced by this script from the UNMODIFIED CanopyRadiation
# v0.1.0 source (commit 33f015b). It must NOT be regenerated from refactored
# code: the record exists to detect changes in numbers that move all code paths
# together. Regeneration is a deliberate, reviewed act; the script refuses to
# run when src/ or ext/ differ from 33f015b unless
# --allow-modified-source is passed, and that flag should only be used with an
# explicit review decision recorded in the commit message.
#
# Usage (from a temporary environment that develops CanopyRadiation and
# CanopyOptics and adds TOML and OffsetArrays):
#
#     julia --startup-file=no --project=<env> test/frozen_record/generate.jl
#
#     julia ... test/frozen_record/generate.jl --part=api,canopyoptics
#
# Output (each overwritten when its part is selected; default: all parts):
#   core         -> test/frozen_record/reference.toml
#   api          -> test/frozen_record/reference_api.toml
#   canopyoptics -> test/frozen_record/reference_canopyoptics.toml
#   regimes      -> test/frozen_record/reference_regimes.toml
# The canopyoptics part also records the CanopyOptics version and git commit,
# and refuses to run if tracked CanopyOptics source is modified.

using CanopyOptics
using CanopyRadiation
using OffsetArrays
using TOML

include(joinpath(@__DIR__, "compute.jl"))
using .FrozenRecord

const FROZEN_COMMIT = "33f015b"

function check_source_is_frozen(allow_modified::Bool)
    root = pkgdir(CanopyRadiation)
    pkgversion(CanopyRadiation) == v"0.1.0" ||
        allow_modified || error("CanopyRadiation version is $(pkgversion(CanopyRadiation)), expected 0.1.0")
    clean = success(`git -C $root diff --quiet $FROZEN_COMMIT -- src ext`)
    if !clean
        allow_modified ||
            error("src/ or ext/ differ from $FROZEN_COMMIT; refusing to regenerate the frozen record " *
                  "(pass --allow-modified-source only after a reviewed decision)")
        @warn "Regenerating the frozen record from source that differs from $FROZEN_COMMIT"
    end
    return clean
end

function write_value(io, x)
    print(io, '"', FrozenRecord.bitstring_hex(x), '"')
end

# Reference-table representation of one field value (used for the
# entry-point agreement check below).
table_value(v::AbstractFloat) = FrozenRecord.bitstring_hex(v)
table_value(v::Union{Bool,Integer}) = v
table_value(v::AbstractVector) =
    FrozenRecord.is_exact_field(v) ? collect(v) : [FrozenRecord.bitstring_hex(y) for y in v]

const CORE_HEADER = [
    "# CanopyRadiation frozen numerical record (reference values).",
    "#",
    "# Produced by test/frozen_record/generate.jl from CanopyRadiation v0.1.0,",
    "# commit $FROZEN_COMMIT, with the inputs in test/frozen_record/inputs.toml.",
    "# DO NOT regenerate from refactored code. Regeneration is a deliberate,",
    "# reviewed act; see the header of generate.jl.",
    "#",
    "# Each table is one code path on one case: <precision>/<group>/<path>/<case>.",
    "# Values are IEEE-754 bit patterns (hex); the trailing comment gives the",
    "# decimal value for reading only. Arrays are column-major. Every entry point",
    "# listed in `entry_points` is compared against the same table.",
]

const API_HEADER = [
    "# CanopyRadiation frozen numerical record, extension: remaining public API",
    "# on manufactured inputs (no CanopyOptics computation involved).",
    "#",
    "# Produced by test/frozen_record/generate.jl --part=api from CanopyRadiation",
    "# v0.1.0, commit $FROZEN_COMMIT, with the api_* inputs in",
    "# test/frozen_record/inputs.toml. DO NOT regenerate from refactored code.",
    "# Regeneration is a deliberate, reviewed act; see the header of generate.jl.",
    "#",
    "# Table names: <precision>/<entry point>/<case>. Float values are IEEE-754 bit",
    "# patterns (hex) with the decimal value in a trailing comment; integer and",
    "# Bool fields (indices, flags, validation outcomes) are plain TOML values and",
    "# compared exactly. Arrays are column-major.",
]

const CANOPYOPTICS_HEADER = [
    "# CanopyRadiation frozen numerical record, extension: CanopyOptics-DEPENDENT",
    "# tables. Every value here is computed in part by CanopyOptics code (leaf-angle",
    "# distribution and Gauss-Legendre quadrature, emission-spectrum conditioning,",
    "# regular-plate escape fates, diffuse plate optics). [meta] records the",
    "# CanopyOptics version and git commit used; the test does not compare these",
    "# tables against a different CanopyOptics version.",
    "#",
    "# Produced by test/frozen_record/generate.jl --part=canopyoptics from",
    "# CanopyRadiation v0.1.0, commit $FROZEN_COMMIT, with the canopyoptics_*",
    "# inputs in test/frozen_record/inputs.toml. DO NOT regenerate from refactored",
    "# code. Regeneration is a deliberate, reviewed act; see generate.jl.",
    "#",
    "# Table names: <precision>/canopyoptics/<entry point>/<case>. Encoding as in",
    "# reference_api.toml.",
]

const REGIMES_HEADER = [
    "# CanopyRadiation frozen numerical record, extension 2: numerical regimes",
    "# named by the refactor reviews (low sun in 20/40-layer columns with a thick",
    "# interior layer, darkness, soil reflectance exactly 0 and 1, 4SAIL chain,",
    "# layer-edge interior reconstruction, empty columns where the tuple and",
    "# vector paths differ, asymmetric photosystems, photon conversion rounding).",
    "# Manufactured inputs only (no CanopyOptics computation).",
    "#",
    "# Produced by test/frozen_record/generate.jl --part=regimes from",
    "# CanopyRadiation v0.1.0, commit $FROZEN_COMMIT, with the regime_* inputs in",
    "# test/frozen_record/inputs.toml. DO NOT regenerate from refactored code.",
    "# Regeneration is a deliberate, reviewed act; see generate.jl.",
    "#",
    "# Table names: <precision>/regimes/<group>/<case...>. Encoding as in",
    "# reference_api.toml. Deep columns record selected interfaces (see inputs.toml).",
]

function write_reference(path, entries, fx, clean; header = CORE_HEADER, extra_meta = String[])
    # Collapse entry points that share a key; they must agree bit for bit.
    order = String[]
    by_key = Dict{String,Vector{FrozenRecord.RecordEntry}}()
    for e in entries
        haskey(by_key, e.key) || (push!(order, e.key); by_key[e.key] = FrozenRecord.RecordEntry[])
        push!(by_key[e.key], e)
    end
    for k in order
        first_e = by_key[k][1]
        for other in by_key[k][2:end]
            table = Dict{String,Any}(f => table_value(v) for (f, v) in first_e.fields)
            table["entry_points"] = [other.entry_point]
            mm, _, _, _, _ = FrozenRecord.compare_record([other], Dict(k => table))
            isempty(mm) || error("entry points sharing key $k disagree:\n" * join(mm, "\n"))
        end
    end
    n_values = FrozenRecord.record_value_count(entries)
    open(path, "w") do io
        foreach(l -> println(io, l), header)
        println(io)
        println(io, "[meta]")
        println(io, "frozen_commit = \"$FROZEN_COMMIT\"")
        println(io, "package_version = \"0.1.0\"")
        println(io, "source_matches_frozen_commit = $clean")
        println(io, "generated_with_julia = \"$(VERSION)\"")
        println(io, "value_count = $n_values")
        println(io, "precisions = [", join(("\"$p\"" for p in fx["precisions"]), ", "), "]")
        foreach(l -> println(io, l), extra_meta)
        for k in order
            es = by_key[k]
            println(io)
            println(io, "[", repr(k), "]")
            println(io, "entry_points = [", join((repr(e.entry_point) for e in es), ", "), "]")
            for (f, v) in es[1].fields
                print(io, repr(f), " = ")
                if v isa AbstractFloat
                    write_value(io, v)
                    println(io, "  # ", repr(v))
                elseif v isa Union{Bool,Integer}
                    println(io, v)
                elseif FrozenRecord.is_exact_field(v)
                    println(io, "[", join(string.(v), ", "), "]")
                else
                    print(io, "[")
                    for (i, y) in enumerate(v)
                        i > 1 && print(io, ", ")
                        write_value(io, y)
                    end
                    println(io, "]  # ", isempty(v) ? "empty" : join(repr.(v), ", "))
                end
            end
        end
    end
    return n_values
end

function main(args)
    allow = "--allow-modified-source" in args
    part_arg = findfirst(a -> startswith(a, "--part="), args)
    parts = part_arg === nothing ? ["core", "api", "canopyoptics", "regimes"] :
            split(args[part_arg][length("--part=") + 1:end], ',')
    clean = check_source_is_frozen(allow)
    fx = load_fixture()
    precisions = [FrozenRecord.precision_type(p) for p in fx["precisions"]]
    collect_entries(f) = reduce(vcat, [f(FT, fx) for FT in precisions])
    for part in parts
        if part == "core"
            path = FrozenRecord.REFERENCE_PATH
            entries = collect_entries(compute_record)
            n = write_reference(path, entries, fx, clean)
        elseif part == "api"
            path = FrozenRecord.REFERENCE_API_PATH
            entries = collect_entries(FrozenRecord.compute_api_record)
            n = write_reference(path, entries, fx, clean; header = API_HEADER)
        elseif part == "canopyoptics"
            path = FrozenRecord.REFERENCE_CANOPYOPTICS_PATH
            entries = collect_entries(FrozenRecord.compute_canopyoptics_record)
            prov = FrozenRecord.canopyoptics_provenance()
            prov.tracked_source == "clean" || allow ||
                error("CanopyOptics checkout has modified tracked source; refusing to record")
            meta = [
                "canopyoptics_version = \"$(prov.version)\"",
                "canopyoptics_git_commit = \"$(prov.commit)\"",
                "canopyoptics_tracked_source = \"$(prov.tracked_source)\"",
            ]
            n = write_reference(path, entries, fx, clean; header = CANOPYOPTICS_HEADER, extra_meta = meta)
        elseif part == "regimes"
            path = FrozenRecord.REFERENCE_REGIMES_PATH
            entries = collect_entries(FrozenRecord.compute_regime_record)
            n = write_reference(path, entries, fx, clean; header = REGIMES_HEADER)
        else
            error("unknown part $(repr(part)); use core, api, canopyoptics or regimes")
        end
        println("wrote $path: $n values, $(length(unique(e.key for e in entries))) tables")
    end
    return nothing
end

main(ARGS)
