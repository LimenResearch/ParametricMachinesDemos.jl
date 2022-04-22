using ParametricMachinesDemos, Flux

t = -pi:0.1:pi

minibatch = 32
x = zeros(length(t), 1, minibatch)
y = zeros(length(t), 1, minibatch)
shift = 10
for i in 1:minibatch
    v = @. sin(t) + 0.1 * (rand() - 0.5)
    x[:, 1, i] .= v
    y[:, 1, i] .= circshift(v, 10)
end

dimensions = [1, 4, 4, 4]

## ConvMachine

machine = ConvMachine(dimensions, sigmoid; pad=(3, 0))

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

model(x)

function loss(X_train, Y_train)
    Flux.Losses.mse(model(X_train), Y_train)
end

loss(x, y)

opt = ADAM(0.1)
ps = Flux.params(model)

# check that learning happens correctly

for i in 1:1000
    gs = gradient(ps) do
        loss(x, y)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 10 == 0
        @show loss(x, y)
    end
end

## RecurMachine

machine = RecurMachine(dimensions, sigmoid; pad=3, timeblock=5)

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

model(x)

function loss(X_train, Y_train)
    Flux.Losses.mse(model(X_train), Y_train)
end

loss(x, y)

opt = ADAM(0.1)
ps = Flux.params(model)

# check that learning happens correctly

for i in 1:1000
    gs = gradient(ps) do
        loss(x, y)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 10 == 0
        @show loss(x, y)
    end
end
