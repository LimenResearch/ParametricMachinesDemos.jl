using ParametricMachinesDemos
using Documenter

makedocs(;
    modules=[ParametricMachinesDemos],
    authors="xxxxxx xxxxxxxx <xxxxxx.xxxxxxxx@xxxxxxxxx.xxx> & xxxxxx x. xxxxxxx <xxxxxxxxxxxxxx@xxxxx.xxx>",
    repo="https://github.com/xxxxxxxxx/xxxxxxxxxxxxxx.jl/blob/{commit}{path}#{line}",
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
    repo="github.com/xxxxxxxxx/xxxxxxxxxxxxxx.jl.jl",
    push_preview=true,
)
