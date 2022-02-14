using ParametricMachines
using Documenter

makedocs(;
    modules=[ParametricMachines],
    authors="Pietro Vertechi <pietro.vertechi@veos.digital> & Mattia G. Bergomi <mattia.bergomi@veos.digital>",
    repo="https://github.com/Veos-Digital/ParametricMachines.jl/blob/{commit}{path}#{line}",
    sitename="Parametric Machines",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages=Any[
        "Home" => "index.md",
        "Machines" => "machines.md",
        "Internals" => "internals.md",
    ],
    strict=true,
)

deploydocs(;
    repo="github.com/Veos-Digital/ParametricMachines.jl",
    push_preview=true,
)
