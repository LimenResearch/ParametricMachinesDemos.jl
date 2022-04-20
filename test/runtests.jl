using ParametricMachinesDemos: dense_filtrations,
                          conv_filtrations,
                          recur_filtrations,
                          filtrations,
                          DenseMachine,
                          ConvMachine,
                          RecurMachine,
                          clean,
                          clean!,
                          solve,
                          solve!,
                          derivative,
                          flip

using NNlib, ChainRulesCore, ChainRulesTestUtils
using Test

@testset "utils" begin
    W = rand(3, 3)
    @test flip(W) == permutedims(W)
    W = rand(5, 3, 3)
    @test flip(W) == permutedims(W[end:-1:1, :, :], (1, 3, 2))
    W = rand(5, 5, 3, 3)
    @test flip(W) == permutedims(W[end:-1:1, end:-1:1, :, :], (1, 2, 4, 3))
end

@testset "dense" begin
    # non linear machine
    minibatch = 6
    dims = [2, 4, 7, 1]
    y₀, z₀ = rand(sum(dims), minibatch), rand(sum(dims), minibatch)
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(sum(dims), sum(dims)), tanh
    (forward, backward) = dense_filtrations(y₀, dims)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    # test that forward pass is a machine
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ.(y) + z₀ # z = σ(y) + z₀
    @test y ≈ W̃' * z + y₀ # y = y₀ + W̃' z
    # test that backward pass is a machine
    Dσ = derivative.(z .- z₀, σ, y)
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀ # z = J ⊙ y + z₀
    @test y ≈ W̃ * z + y₀ # y = y₀ + W̃ z
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives

    # linear machine
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(sum(dims), sum(dims)), rand(sum(dims), minibatch)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    y, z = copy(y₀), copy(z₀)
    # test that forward pass is a machine
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ .* y + z₀
    @test y ≈ W̃' * z + y₀
    # test that backward pass is a machine
    Dσ = σ
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ W̃ * z + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives

    # machine versus `solve`
    machine1 = DenseMachine(dims, tanh)
    machine2 = DenseMachine(tanh, dims)
    for m in (machine1, machine2)
        @test m == DenseMachine(m.W, m.σ, m.dims)
        @test m(y₀) ≈ solve(y₀, m.W, m.σ, dense_filtrations(y₀, m.dims))
    end
end

@testset "conv 1D" begin
    # nonlinear machine
    minibatch = 6
    dims = [2, 4, 7, 1]
    pad = (1, 3)
    kernelsize = (5,)
    datasize = (12,)
    y₀, z₀ = rand(datasize..., sum(dims), minibatch), rand(datasize..., sum(dims), minibatch)
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(kernelsize..., sum(dims), sum(dims)), tanh
    (forward, backward) = conv_filtrations(y₀, dims; pad)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ.(y) + z₀
    @test y ≈ conv(z, W̃; pad) + y₀
    # test that backward pass is a machine
    Dσ = derivative.(z .- z₀, σ, y)
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ ∇conv_data(z, W̃, DenseConvDims(y, W̃, padding=pad)) + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives

    # linear machine
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(kernelsize..., sum(dims), sum(dims)), rand(datasize..., sum(dims), minibatch)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ .* y + z₀
    @test y ≈ conv(z, W̃; pad) + y₀
    # test that backward pass is a machine
    Dσ = σ
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ ∇conv_data(z, W̃, DenseConvDims(y, W̃, padding=pad)) + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives
    
    # machine versus `solve`
    machine1 = ConvMachine(dims, tanh; pad, init=rand)
    machine2 = ConvMachine(tanh, dims; pad, init=rand)
    for m in (machine1, machine2)
        @test m == ConvMachine(m.W, m.σ, m.dims; pad)
        @test m(y₀) ≈ solve(y₀, m.W, m.σ, conv_filtrations(y₀, m.dims; m.pad))
    end
end

@testset "conv 2D" begin
    # nonlinear machine
    minibatch = 3
    dims = [2, 1, 3, 1]
    pad = (1, 3, 2, 0)
    kernelsize = (pad[1] + pad[2] + 1, pad[3] + pad[4] + 1)
    datasize = (6, 9)
    y₀, z₀ = rand(datasize..., sum(dims), minibatch), rand(datasize..., sum(dims), minibatch)
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(kernelsize..., sum(dims), sum(dims)), tanh
    (forward, backward) = conv_filtrations(y₀, dims; pad)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ.(y) + z₀
    @test y ≈ conv(z, W̃; pad) + y₀
    # test that backward pass is a machine
    Dσ = derivative.(z .- z₀, σ, y)
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ ∇conv_data(z, W̃, DenseConvDims(y, W̃, padding=pad)) + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives

    # linear machine
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(kernelsize..., sum(dims), sum(dims)), rand(datasize..., sum(dims), minibatch)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ .* y + z₀
    @test y ≈ conv(z, W̃; pad) + y₀
    # test that backward pass is a machine
    Dσ = σ
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ ∇conv_data(z, W̃, DenseConvDims(y, W̃, padding=pad)) + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives
    
    # machine versus `solve`
    machine1 = ConvMachine(dims, tanh; pad, init=rand)
    machine2 = ConvMachine(tanh, dims; pad, init=rand)
    for m in (machine1, machine2)
        @test m == ConvMachine(m.W, m.σ, m.dims; pad)
        @test m(y₀) ≈ solve(y₀, m.W, m.σ, conv_filtrations(y₀, m.dims; m.pad))
    end
end

@testset "recurrent" begin
    # nonlinear machine
    minibatch = 6
    dims = [2, 4, 7, 1]
    pad = 6
    kernelsize = (pad + 1,)
    timeblock = 5
    datalength = 32 # test `datalength % timeblock != 0`
    y₀, z₀ = rand(datalength, sum(dims), minibatch), rand(datalength, sum(dims), minibatch)
    y, z = copy(y₀), copy(z₀)
    W, σ = randn(kernelsize..., sum(dims), sum(dims)), tanh
    (forward, backward) = recur_filtrations(y₀, dims; pad, timeblock)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ.(y) + z₀
    @test y ≈ conv(z, W̃; pad=(pad, 0)) + y₀
    # test that backward pass is a machine
    Dσ = derivative.(z .- z₀, σ, y)
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ ∇conv_data(z, W̃, DenseConvDims(y, W̃, padding=(pad, 0))) + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives

    y, z = copy(y₀), copy(z₀)
    W, σ = randn(kernelsize..., sum(dims), sum(dims)), rand(datalength, sum(dims), minibatch)
    @test flip(clean(W, forward)) ≈ clean(flip(W), backward)
    W̃ = clean(W, forward)
    solve!(y, z, W̃, σ, forward)
    @test z ≈ σ .* y + z₀
    @test y ≈ conv(z, W̃; pad=(pad, 0)) + y₀
    # test that backward pass is a machine
    Dσ = σ
    y, z = copy(y₀), copy(z₀)
    solve!(y, z, flip(W̃), Dσ, backward)
    @test z ≈ Dσ .* y + z₀
    @test y ≈ ∇conv_data(z, W̃, DenseConvDims(y, W̃, padding=(pad, 0))) + y₀
    test_rrule(solve, y₀, z₀, W, σ, (forward, backward) ⊢ NoTangent()) # test derivatives
    
    # machine versus `solve`
    machine1 = RecurMachine(dims, tanh; pad, timeblock, init=rand)
    machine2 = RecurMachine(tanh, dims; pad, timeblock, init=rand)
    for m in (machine1, machine2)
        @test m == RecurMachine(m.W, m.σ, m.dims; pad, timeblock)
        @test m(y₀) ≈ solve(y₀, m.W, m.σ, recur_filtrations(y₀, m.dims; m.pad, m.timeblock))
    end
end

@testset "filtrations nograd" begin
    dimensions = [2, 2, 2]
    minibatch = 32
    y = rand(sum(dimensions), minibatch)
    machine = DenseMachine(dimensions, tanh)
    (forward, backward) = filtrations(machine, y)
    res, pb = rrule(filtrations, machine, y)
    @test res.forward.shapes == forward.shapes
    @test res.backward.shapes == backward.shapes
    @test pb(nothing, nothing) == (NoTangent(), NoTangent(), NoTangent())

    duration, minibatch = 12, 32
    pad = (3, 3)
    y = rand(duration, sum(dimensions), minibatch)
    machine = ConvMachine(dimensions, tanh; pad)
    (forward, backward) = filtrations(machine, y)
    res, pb = rrule(filtrations, machine, y)
    @test res.forward.shapes == forward.shapes
    @test res.backward.shapes == backward.shapes
    @test pb(nothing, nothing) == (NoTangent(), NoTangent(), NoTangent())

    duration, minibatch = 12, 32
    pad, timeblock = 3, 5
    y = rand(duration, sum(dimensions), minibatch)
    machine = RecurMachine(dimensions, tanh; pad, timeblock)
    (forward, backward) = filtrations(machine, y)
    res, pb = rrule(filtrations, machine, y)
    @test res.forward.shapes == forward.shapes
    @test res.backward.shapes == backward.shapes
    @test pb(nothing, nothing) == (NoTangent(), NoTangent(), NoTangent())
end