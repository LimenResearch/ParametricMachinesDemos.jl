using ParametricMachines
using BenchmarkTools, ChainRulesCore, Flux, Zygote, CUDA

# Benchmarking

timeratio(b1, b2) = minimum(b1.times) / minimum(b2.times)

function benchmark((y₀, z₀, ȳ, z̄, machine), device)
    σ, W = machine.σ, machine.W
    forward, backward = ParametricMachines.filtrations(machine, y₀)

    function forward_pass(y₀, z₀)
        y, z = copy(y₀), copy(z₀)
        W̃ = ParametricMachines.clean!(copy(W), forward)
        ParametricMachines.solve!(y, z, W̃, σ, forward)
        return y, z
    end

    # Benchmark forward step
    benchmark_forward = if device == "CPU"
        @benchmark $forward_pass($y₀, $z₀)
    elseif device == "GPU"
        @benchmark CUDA.@sync $forward_pass($y₀, $z₀)
    end

    y, z = forward_pass(y₀, z₀)

    function backward_pass(ȳ, z̄)
        Dσ = ParametricMachines.derivative.(z .- z₀, σ, y)
        W̃′ = ParametricMachines.clean!(ParametricMachines.flip(W), backward)
        ȳ₀, z̄₀ = copy(ȳ), copy(z̄)
        ParametricMachines.solve!(z̄₀, ȳ₀, W̃′, Dσ, backward)
        return ȳ₀, z̄₀
    end

    # Benchmark backward step
    benchmark_backward = if device == "CPU"
        @benchmark $backward_pass($ȳ, $z̄)
    elseif device == "GPU"
        @benchmark CUDA.@sync $backward_pass($ȳ, $z̄)
    end
    @show timeratio(benchmark_backward, benchmark_forward)
    return (forward=benchmark_forward, backward=benchmark_backward)
end

df = @NamedTuple{device::String, problemsize::String, machine::String, forward::Float64, backward::Float64}[]
devices = (CPU=f64, GPU=gpu)

for problemsize in ["small", "medium"]
    dimensions = problemsize == "small" ? [2, 2, 2, 2, 2] : [32, 32, 32, 32, 32]
    minibatch = problemsize == "small" ? 2 : 32
    y₀ = rand(Float32, sum(dimensions), minibatch)
    z₀ = rand(Float32, sum(dimensions), minibatch)
    ȳ = rand(Float32, sum(dimensions), minibatch)
    z̄ = rand(Float32, sum(dimensions), minibatch)
    machine = DenseMachine(dimensions, tanh)

    for device in ["CPU", "GPU"]
        to_device = devices[Symbol(device)]
        benchmarks = benchmark(map(to_device, (y₀, z₀, ȳ, z̄, machine)), device)
        push!(
            df,
            (; device, problemsize, machine="dense",
            forward=minimum(benchmarks.forward.times),
            backward=minimum(benchmarks.backward.times))
        )
    end
end

# Convolutional machine

for problemsize in ["small", "medium"]
    dimensions = problemsize == "small" ? [2, 2, 2, 2, 2] : [32, 32, 32, 32, 32]
    minibatch = problemsize == "small" ? 2 : 32
    duration = 16
    pad = (3, 3)
    y₀ = rand(Float32, duration, sum(dimensions), minibatch)
    z₀ = rand(Float32, duration, sum(dimensions), minibatch)
    ȳ = rand(Float32, duration, sum(dimensions), minibatch)
    z̄ = rand(Float32, duration, sum(dimensions), minibatch)
    machine = ConvMachine(dimensions, tanh; pad)

    for device in ["CPU", "GPU"]
        to_device = devices[Symbol(device)]
        benchmarks = benchmark(map(to_device, (y₀, z₀, ȳ, z̄, machine)), device)
        push!(
            df,
            (; device, problemsize, machine="convolution",
            forward=minimum(benchmarks.forward.times),
            backward=minimum(benchmarks.backward.times))
        )
    end
end

# Recurrent machine

for problemsize in ["small", "medium"]
    dimensions = problemsize == "small" ? [2, 2, 2, 2, 2] : [32, 32, 32, 32, 32]
    minibatch = problemsize == "small" ? 2 : 32
    duration = 16
    pad = 6
    timeblock = 5
    y₀ = rand(Float32, duration, sum(dimensions), minibatch)
    z₀ = rand(Float32, duration, sum(dimensions), minibatch)
    ȳ = rand(Float32, duration, sum(dimensions), minibatch)
    z̄ = rand(Float32, duration, sum(dimensions), minibatch)
    machine = RecurMachine(dimensions, tanh; pad, timeblock)

    for device in ["CPU", "GPU"]
        to_device = devices[Symbol(device)]
        benchmarks = benchmark(map(to_device, (y₀, z₀, ȳ, z̄, machine)), device)
        push!(
            df,
            (; device, problemsize, machine="recurrent",
            forward=minimum(benchmarks.forward.times),
            backward=minimum(benchmarks.backward.times))
        )
    end
end

# Log to file

using CSV

CSV.write(joinpath(@__DIR__, "benchmarks.csv"), df)
