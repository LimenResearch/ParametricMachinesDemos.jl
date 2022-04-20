module ParametricMachinesDemos

using Base: front, setindex
using NNlib: conv!, ∇conv_filter!, DenseConvDims
using LinearAlgebra: mul!
using Functors: @functor

using ChainRulesCore: NoTangent, unthunk, derivatives_given_output, ChainRulesCore

import ChainRules

export DenseMachine, ConvMachine, RecurMachine

include("slice.jl")
include("solve.jl")
include("architectures.jl")

end
