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
# CanopyOptics and adds TOML):
#
#     julia --startup-file=no --project=<env> test/frozen_record/generate.jl
#
# Output: test/frozen_record/reference.toml (overwritten).

using CanopyRadiation
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

function write_reference(path, entries, fx, clean)
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
            table = Dict{String,Any}(
                f => (v isa AbstractFloat ? FrozenRecord.bitstring_hex(v) :
                      [FrozenRecord.bitstring_hex(y) for y in v]) for (f, v) in first_e.fields)
            table["entry_points"] = [other.entry_point]
            mm, _, _, _, _ = FrozenRecord.compare_record([other], Dict(k => table))
            isempty(mm) || error("entry points sharing key $k disagree:\n" * join(mm, "\n"))
        end
    end
    n_values = FrozenRecord.record_value_count(entries)
    open(path, "w") do io
        println(io, "# CanopyRadiation frozen numerical record (reference values).")
        println(io, "#")
        println(io, "# Produced by test/frozen_record/generate.jl from CanopyRadiation v0.1.0,")
        println(io, "# commit $FROZEN_COMMIT, with the inputs in test/frozen_record/inputs.toml.")
        println(io, "# DO NOT regenerate from refactored code. Regeneration is a deliberate,")
        println(io, "# reviewed act; see the header of generate.jl.")
        println(io, "#")
        println(io, "# Each table is one code path on one case: <precision>/<group>/<path>/<case>.")
        println(io, "# Values are IEEE-754 bit patterns (hex); the trailing comment gives the")
        println(io, "# decimal value for reading only. Arrays are column-major. Every entry point")
        println(io, "# listed in `entry_points` is compared against the same table.")
        println(io)
        println(io, "[meta]")
        println(io, "frozen_commit = \"$FROZEN_COMMIT\"")
        println(io, "package_version = \"0.1.0\"")
        println(io, "source_matches_frozen_commit = $clean")
        println(io, "generated_with_julia = \"$(VERSION)\"")
        println(io, "value_count = $n_values")
        println(io, "precisions = [", join(("\"$p\"" for p in fx["precisions"]), ", "), "]")
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
    clean = check_source_is_frozen(allow)
    fx = load_fixture()
    entries = FrozenRecord.RecordEntry[]
    for p in fx["precisions"]
        append!(entries, compute_record(FrozenRecord.precision_type(p), fx))
    end
    n = write_reference(FrozenRecord.REFERENCE_PATH, entries, fx, clean)
    println("wrote $(FrozenRecord.REFERENCE_PATH): $n values, $(length(unique(e.key for e in entries))) tables")
    return nothing
end

main(ARGS)
