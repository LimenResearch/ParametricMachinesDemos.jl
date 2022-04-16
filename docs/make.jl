using ParametricMachines
using Documenter

makedocs(;
    modules=[ParametricMachines],
    authors="Pietro Vertechi <pietro.vertechi@protonmail.com> & Mattia G. Bergomi <mattiagbergomi@gmail.com>",
    repo="https://github.com/BeaverResearch/ParametricMachines.jl/blob/{commit}{path}#{line}",
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
    repo="github.com/BeaverResearch/ParametricMachines.jl",
    push_preview=true,
)
