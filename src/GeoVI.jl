module GeoVI

using ADTypes
using LinearAlgebra
using LogExpFunctions: log1pexp, logistic
using Optimisers
using Random
using ReactantCore

export randn_like
export AbstractLikelihood,
    GaussianLikelihood,
    PoissonLikelihood,
    BernoulliLikelihood,
    BinomialLikelihood,
    ComposedLikelihood,
    logdensity,
    energy,
    normalized_residual,
    transformation,
    leftsqrtfishermetric,
    leftsqrtmetric,
    rightsqrtfishermetric,
    rightsqrtmetric,
    fishermetric,
    metric,
    compose
export Samples, posterior_samples, recenter
export AbstractVariationalFamily,
    GeoVIFamily,
    MGVIFamily,
    AbstractVIScheme,
    MGVIScheme,
    GeoVIScheme,
    AbstractFDivergence,
    AbstractDivergence,
    ReverseKL,
    ForwardKL,
    AbstractOptimizer,
    NewtonCG,
    VIConfig,
    VIState,
    initialize_vi,
    draw_linear_residual,
    update_nonlinear_residual,
    draw_residual,
    step_vi,
    fit,
    optimize_vi,
    optimize_kl

include("tree_utils.jl")
include("likelihoods.jl")
include("sampling.jl")
include("samples.jl")
include("vi.jl")
include("nonlinear.jl")

end
