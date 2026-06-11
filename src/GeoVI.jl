module GeoVI

using ADTypes
using Functors: fmap
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
export AbstractVariationalDistribution, DiagonalGaussian, FisherGaussianDistribution, distribution
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
    MeanFieldGaussian,
    AbstractFDivergence,
    ReverseKL,
    AbstractOptimizer,
    NewtonCG,
    AbstractEstimator,
    MCEstimator,
    VariationalProblem,
    VIState,
    init,
    draw_metric_sample,
    draw_linear_residual,
    update_nonlinear_residual,
    draw_residual,
    draw_residuals,
    step_vi!,
    fit

include("tree_utils.jl")
include("likelihoods.jl")
include("cg.jl")
include("optimize.jl")
include("sampling.jl")
include("samples.jl")
include("families/interface.jl")
include("families/mgvi.jl")
include("families/geovi.jl")
include("families/fisher_gaussian.jl")
include("families/meanfield.jl")
include("vi.jl")
include("nonlinear.jl")

end
