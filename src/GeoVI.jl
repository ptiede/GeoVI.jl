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
export MetricSample,
    ConjugateGradientInfo,
    ConjugateGradient,
    solve,
    LinearResidualDraw,
    NonlinearResidualUpdate,
    MirroredResidualDraw,
    OptimizationResult
export AbstractVariationalFamily,
    GeoVIFamily,
    MGVIFamily,
    AbstractFDivergence,
    ReverseKL,
    ForwardKL,
    AbstractOptimizer,
    NewtonCG,
    VIConfig,
    VariationalProblem,
    VIState,
    initialize_vi,
    draw_metric_sample,
    draw_linear_residual,
    update_nonlinear_residual,
    draw_residual,
    step_vi,
    fit

include("tree_utils.jl")
include("likelihoods.jl")
include("cg.jl")
include("optimize.jl")
include("sampling.jl")
include("samples.jl")
include("vi.jl")
include("nonlinear.jl")

end
