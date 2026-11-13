using Documenter
using CanopyOptics
using CanopyRadiation

@info "Building docs with CanopyRadiation source" path = pathof(CanopyRadiation)

function build()
    pages = Any[
        "Home" => "index.md",
        "API Reference" => [
            "Overview & Index" => "api/index.md",
            "Passive & Emitting Column Solvers" => "api/column_and_cavity.md",
            "Leaf-Face Illumination & Spectral Escape" => "api/faces_and_emission.md",
            "Hemispheric SIF, Spectral Bases, & Kernels" => "api/hemispheric_and_kernels.md",
        ],
    ]

    mathengine = MathJax(
        Dict(
            :TeX => Dict(
                :equationNumbers => Dict(:autoNumber => "AMS"),
                :Macros => Dict(),
            ),
        ),
    )

    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "main",
        repolink = "https://github.com/RemoteSensingTools/CanopyRadiation.jl",
        mathengine = mathengine,
        collapselevel = 1,
        size_threshold = nothing,
        size_threshold_warn = nothing,
        example_size_threshold = nothing,
    )

    DocMeta.setdocmeta!(
        CanopyRadiation,
        :DocTestSetup,
        :(using CanopyRadiation, CanopyOptics);
        recursive = true,
    )

    makedocs(
        sitename = "CanopyRadiation.jl",
        format = format,
        remotes = nothing,
        clean = true,
        doctest = true,
        checkdocs = :all,
        modules = [CanopyRadiation],
        pages = pages,
    )
end

build()

if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/RemoteSensingTools/CanopyRadiation.jl.git",
        target = "build",
        devbranch = "main",
        push_preview = true,
    )
end
