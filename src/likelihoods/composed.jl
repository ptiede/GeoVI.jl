struct ComposedLikelihood{L,F,LIN,AD,AO} <: AbstractLikelihood
    likelihood::L
    forward::F
    linearize::LIN
    adtype::AD
    autodiff_options::AO
end

function ComposedLikelihood(
    likelihood::AbstractLikelihood,
    forward;
    linearize=nothing,
    pushforward=nothing,
    pullback=nothing,
    adtype=ADTypes.AutoFiniteDiff(),
    autodiff_options=(;),
)
    if linearize !== nothing && (pushforward !== nothing || pullback !== nothing)
        throw(
            ArgumentError(
                "specify either `linearize` or `pushforward`/`pullback`, not both",
            ),
        )
    end
    if linearize === nothing && (pushforward !== nothing || pullback !== nothing)
        linearize = _manual_linearize(forward, pushforward, pullback)
    end
    return ComposedLikelihood(
        likelihood,
        forward,
        linearize,
        adtype,
        autodiff_options,
    )
end

function compose(
    lh::AbstractLikelihood,
    forward;
    linearize=nothing,
    pushforward=nothing,
    pullback=nothing,
    adtype=ADTypes.AutoFiniteDiff(),
    autodiff_options=(;),
)
    return ComposedLikelihood(
        lh,
        forward;
        linearize=linearize,
        pushforward=pushforward,
        pullback=pullback,
        adtype=adtype,
        autodiff_options=autodiff_options,
    )
end

function _composed_linearization(lh::ComposedLikelihood, x)
    if lh.linearize !== nothing
        return _evaluate_linearizer(lh.linearize, x)
    end
    adtype = _infer_composed_adtype(lh.adtype, x)
    return _automatic_linearize(adtype, lh.forward, x; lh.autodiff_options...)
end

logdensity(lh::ComposedLikelihood, x) = logdensity(lh.likelihood, lh.forward(x))
normalized_residual(lh::ComposedLikelihood, x) =
    normalized_residual(lh.likelihood, lh.forward(x))
transformation(lh::ComposedLikelihood, x) = transformation(lh.likelihood, lh.forward(x))

function rightsqrtmetric(lh::ComposedLikelihood, x, v)
    linearization = _composed_linearization(lh, x)
    return rightsqrtmetric(
        lh.likelihood,
        linearization.value,
        pushforward(linearization, v),
    )
end

function leftsqrtmetric(lh::ComposedLikelihood, x, η)
    linearization = _composed_linearization(lh, x)
    lifted = leftsqrtmetric(lh.likelihood, linearization.value, η)
    return pullback(linearization, lifted)
end

function fishermetric(lh::ComposedLikelihood, x, v)
    linearization = _composed_linearization(lh, x)
    lifted = fishermetric(
        lh.likelihood,
        linearization.value,
        pushforward(linearization, v),
    )
    return pullback(linearization, lifted)
end
