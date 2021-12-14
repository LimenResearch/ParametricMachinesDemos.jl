using CSV

df = CSV.File(joinpath(@__DIR__, "benchmarks.csv"))

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