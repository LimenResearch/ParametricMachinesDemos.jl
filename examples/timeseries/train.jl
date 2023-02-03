using CSV, DataFrames

energy_dataset = CSV.read(joinpath(@__DIR__, "data", "dataset.csv"), DataFrame)

using Optim: optimize, only_fg!, minimizer, LBFGS
using Zygote: pullback
using DelimitedFiles, Statistics

function train_forecast(data;
    loss, model,
    dimensions, timeblock, pad,
    iterations=1000, f_reltol=1e-6,
    center=false, rescale=false, device=cpu,
    train_range=Colon())

    raw = data
    # Compute mean and std on training data
    μ = mean(raw[train_range, :, :], dims=(1, 3))
    center || (μ .= 0)
    σ = std(raw[train_range, :, :], dims=(1, 3))
    rescale || (σ .= 1)
    processed_data = (raw .- μ) ./ σ |> device

    week_length = 4 * 24 * 7
    input = Float32.(processed_data[1:end-week_length, :, :])
    output = Float32.(processed_data[week_length+1:end, :, :])
    machine = model(dimensions, sigmoid; pad=pad, timeblock=timeblock)
    predictor = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> device
    train_input, train_output = input[train_range, :, :], output[train_range, :, :]

    @info "Warmup."
    loss(predictor, train_input, train_output)
    θ, reconstruct = Flux.destructure(predictor)
 
    function fg!(_, G, θ)
        l, back = pullback(θ) do θ
            predictor = reconstruct(θ)
            return loss(predictor, train_input, train_output)
        end
        if !isnothing(G)
            grad, = back(one(l))
            G .= grad
        end
        return l
    end

    @info "Starting training."
    opt_res = optimize(only_fg!(fg!), θ, method=LBFGS(), iterations=iterations,
        f_reltol=f_reltol, store_trace=true, extended_trace=true)
    opt_machine = reconstruct(minimizer(opt_res))

    return (
        machine=opt_machine,
        result=opt_res,
        input=input,
        output=output,
        train_range=train_range,
        μ=device(μ),
        σ=device(σ),
        center=center,
        rescale=rescale
    )
end