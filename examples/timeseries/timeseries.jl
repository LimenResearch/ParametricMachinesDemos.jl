using ParametricMachinesDemos
using Flux.Losses: mse
using Flux: gpu


"""
smoothness(W, dims::Int...)
Compute sum of squares of derivatives of `W` along dimensions in `dims`.
"""
smoothness(W) = zero(eltype(W))
smoothness(W, d::Int, ds::Int...) = sum(abs2, diff(W; dims=d)) + smoothness(W, ds...)

time_smoothness(m::RecurMachine) = smoothness(m.W, 1)

dimensions = [1, 8, 8, 8]
timeblock=4*24*2 # two days
pad =4*24
f_reltol=1e-6
iterations=250
train_range = 1:44 * 672

# This code allows to explore different costs more easily
loss = function (machine, input, output)
    p = machine(input)
    l = mse(p, output)
    c_t =  0.01f0 * time_smoothness(machine[1])
    return l + c_t
end

user_idxs = findall(==(1358568), energy_dataset.USER)
@assert issorted(energy_dataset.DATE[user_idxs])
user_demand = energy_dataset.var"DEMANDA ACTIVA"[user_idxs]

result =
    train_forecast(
        reshape(user_demand, :, 1, 1);
        loss=loss,
        model=RecurMachine,
        dimensions=dimensions,
        pad=pad,
        timeblock=timeblock,
        f_reltol=f_reltol,
        iterations=iterations,
        center=true,
        rescale=true,
        device=cpu,
        train_range=train_range
    )

machine, optimization_result, input, output, μ, st_dev =
    result.machine, result.result, result.input, result.output, result.μ, result.σ

## Visualize
using Plots

let week = 46
    offset = week * 672
    ground_truth = vec(collect(output.*st_dev .+ μ))[offset + 1:offset + 672]
    prediction = vec(collect(machine(input).*st_dev .+ μ))[offset + 1:offset + 672]
    plot(ground_truth, label="truth")
    plot!(prediction, label="prediction", linewidth=2)
end

test_range = last(train_range)+1:length(input)
perc_err = (machine(input[test_range, :, :]) - output[test_range, :, :]) ./
    (output[test_range, :, :] .+ μ)

mean(abs, perc_err)
# mse(machine(input[test_range, :, :]), output[test_range, :, :])
## Iterated prediction

long_pred = foldl(|>, fill(machine, 4), init=gpu(user_demand[1:end-4*672, :, :]) .- μ)
long_output = gpu(user_demand[4*672+1:end, :, :] ).- μ

let week = 44
    offset = week * 672
    ground_truth = long_output[offset + 1:offset + 672].*st_dev .+ μ |> collect |> vec
    prediction = long_pred[offset + 1:offset + 672].*st_dev .+ μ |> collect |> vec
    plot(ground_truth, label="truth")
    plot!(prediction, label="prediction", linewidth=2)
end