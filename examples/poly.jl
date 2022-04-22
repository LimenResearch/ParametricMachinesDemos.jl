using ParametricMachinesDemos, Flux
  
poly(x, y) = (2x-1)^2 + 2y + x * y - 3
poly((x, y)) = poly(x, y)
rg = 0:0.01:1
N = length(rg)
flat = hcat(repeat(rg, inner=N), repeat(rg, outer=N))
truth = map(poly, eachrow(flat))

## Let us generate a `6 x 6` trainig grid.

N_train = 6
rg_train = range(0, 1, length=N_train)
X_train = hcat(repeat(rg_train, inner=N_train), repeat(rg_train, outer=N_train)) |> permutedims
Y_train = map(poly, eachcol(X_train)) |> permutedims

dimensions = [2, 4, 4, 4]

machine = DenseMachine(dimensions, sigmoid)

model = Flux.Chain(machine, Dense(sum(dimensions), 1)) |> f64

function loss(X_train, Y_train)
    Flux.Losses.mse(model(X_train), Y_train)
end

opt = ADAM(0.1)
ps = Flux.params(model)

# check that learning happens correctly

N = 14
rg = range(0, 1, length=N)
X = hcat(repeat(rg, inner=N), repeat(rg, outer=N)) |> permutedims
Y = map(poly, eachcol(X)) |> permutedims

for i in 1:10000
    gs = gradient(ps) do
        loss(X_train, Y_train)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 500 == 0
        @show loss(X_train, Y_train)
        @show loss(X, Y)
    end
end
