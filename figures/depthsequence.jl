using Random, LinearAlgebra

function strictlyuppertriangular(n)
    M = (rand(n, n) .+ 0.5)
    for c in CartesianIndices(M)
        if c[1] ≤ c[2]
            M[c] = 0
        end
    end
    return M
end

Random.seed!(1234)

n = 5
ratio = (1, 1)
σ = identity
x₀ = rand(n) .+ 0.5
W = strictlyuppertriangular(n)

expensive_sequence = [x₀]
let xcurrent = x₀
    for i in 1:(n-1)
        xnew = W * σ.(xcurrent) + x₀
        push!(expensive_sequence, (xnew - xcurrent))
        xcurrent = xnew
    end
end

optimized_sequence = [x₀]
let xcurrent = x₀
    for i in 1:(n-1)
        xnew = copy(xcurrent)
        xnew[i+1] += W[i+1, :] ⋅ σ.(xcurrent)
        push!(optimized_sequence, (xnew - xcurrent))
        xcurrent = xnew
    end
end

using AlgebraOfGraphics, CairoMakie
set_aog_theme!()

y = hcat(optimized_sequence, expensive_sequence)
x = map(_ -> 0:n-1, y)

plt = mapping(
    x => "node",
    y => "activation value",
    stack=0:n-1 => nonnumeric => "step",
    color=0:n-1 => nonnumeric => "step",
    row=[1 2] => renamer(1 => "optimized sequence", 2 => "expensive sequence")
)


##

using GraphMakie, CairoMakie, Graphs
using AlgebraOfGraphics
set_aog_theme!()

font = AlgebraOfGraphics.firasans("Light")

fig = Figure(resolution=(600, 600))

ax1 = Axis(fig[1, 1:ratio[1]], aspect=1)
ax2 = Axis(fig[2, 1:ratio[1]], aspect=1)

colormap = collect(cgrad(:sienna, n, categorical=true, rev=true))

g = complete_graph(n)
colors = map(edges(g)) do edge
    colormap[dst(edge)]
end

arrow_size = 25
edge_width_base = 3
node_size=45

function layout(g::AbstractGraph)
    N = nv(g)
    return map(1:N) do i
        θ = 2π*(i - 1)/N
        return Point(cos(π - θ), sin(π - θ))
    end
end

graphplot!(
    ax1, g, arrow_show=true, arrow_size=arrow_size, edge_width=edge_width_base, edge_color=colors,
    nlabels=map(string, 0:n-1), nlabels_align=(:center, :center), nlabels_attr=(; font, textsize=16f0),
    node_color=:white, node_size=node_size, layout=layout
)

edge_width = map(edges(g)) do edge
    edge_width_base * src(edge)
end

graphplot!(
    ax2, g, edge_width=edge_width, edge_color=colors,
    nlabels=map(string, 0:n-1), nlabels_align=(:center, :center), nlabels_attr=(; font, textsize=16f0),
    node_color=:white, node_size=node_size, layout=layout
)

foreach(hidedecorations!, (ax1, ax2))
foreach(hidespines!, (ax1, ax2))

ag = draw!(
    fig[:, ratio[1] .+ (1:ratio[2])],
    plt * visual(BarPlot),
    palettes=(color=colormap,),
    axis = (xticks=0:n-1,),
)

legend!(fig[0, 1:sum(ratio)], ag, orientation=:horizontal)

display(fig)
