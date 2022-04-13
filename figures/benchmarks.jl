using CSV

df = CSV.File(joinpath(@__DIR__, "benchmarks.csv"))

# Display formatted table

function pretty_print(io, x::Number)
    d, r = fldmod(x, 1)
    r ≈ 0 ? print(io, Int(d)) : print(io, x)
end

function print_benchmarks(io, df)
    print(io, raw"""
    \begin{tabular}{llllll}
        \hline
        \textbf{machine} & \textbf{size} & \textbf{device} & \textbf{forward (\si{\milli\second})} & \textbf{backward (\si{\milli\second})} & \textbf{ratio} \\ \hline
    """)
    for row in df
        print(io, "    ")
        print(io, row.machine)
        print(io, " & ")
        print(io, row.problemsize)
        print(io, " & ")
        print(io, row.device)
        print(io, " & ")
        pretty_print(io, round(row.forward / 1000, sigdigits=5))
        print(io, " & ")
        pretty_print(io, round(row.backward / 1000, sigdigits=5))
        print(io, " & ")
        pretty_print(io, round(row.backward / row.forward, digits=3))
        print(io, raw" \\ ")
        print(io, '\n')
    end
    print(io, raw"""
        \hline
    \end{tabular}
    """)
end

print_benchmarks(stdout, df)

# Display figure

using AlgebraOfGraphics, CairoMakie
set_aog_theme!()

passes = [:forward, :backward]

plt = data(df) * mapping(
    :machine => sorter("dense", "convolution", "recurrent"),
    passes .=> (t -> (t / 1000)) .=> "runtime (μs)",
    color = dims(1) => renamer(passes) => "pass",
    dodge = dims(1) => renamer(passes) => "pass",
    col = :device,
    row = :problemsize => renamer("small" => "small size", "medium" => "medium size")
)

fig = Figure(resolution=(600, 800))

ag = draw!(
    fig, plt * visual(BarPlot, fillto = 1, dodge_gap = 0.),
    axis=(xticklabelrotation=π/6, yscale=log10, yticks= LogTicks(LinearTicks(6)))
)
legend!(fig[0, 1:2], ag, orientation=:horizontal)

display(fig)