# `likelihood` need NOT be an `AbstractLikelihood` — it may be any object (e.g. a plain
# `Distributions.Distribution`) that implements the observation-space likelihood interface
# (`logdensity`, `normalized_residual`, `transformation`, `leftsqrtmetric`, `rightsqrtmetric`,
# and `fishermetric`) for the prediction `y`. The `ComposedLikelihood` wrapper itself IS an
# `AbstractLikelihood`, so the rest of the VI machinery is unchanged; only the inner per-call
# methods are forwarded to `likelihood`. This lets callers reuse an existing data-likelihood
# type rather than wrapping it in a bespoke `AbstractLikelihood`.
struct ComposedLikelihood{L, F, LIN, AD, AO} <: AbstractLikelihood
    likelihood::L
    forward::F
    linearize::LIN
    adtype::AD
    autodiff_options::AO
end

function ComposedLikelihood(
        likelihood,
        forward;
        linearize = nothing,
        pushforward = nothing,
        pullback = nothing,
        adtype = ADTypes.AutoFiniteDiff(),
        autodiff_options = (;),
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
        lh,
        forward;
        linearize = nothing,
        pushforward = nothing,
        pullback = nothing,
        adtype = ADTypes.AutoFiniteDiff(),
        autodiff_options = (;),
    )
    return ComposedLikelihood(
        lh,
        forward;
        linearize = linearize,
        pushforward = pushforward,
        pullback = pullback,
        adtype = adtype,
        autodiff_options = autodiff_options,
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

# The pinned handle caches the forward-model linearization ONCE; every 2-arg metric
# application below reuses it. This is the whole point of `_at_point`: the 3-arg
# `fishermetric`/`*sqrtmetric` forms (kept below for one-shot use) rebuild the
# linearization — a forward evaluation plus AD setup — on every call, which is
# ruinous inside a CG solve that applies the metric at one fixed point many times.
struct _ComposedAtPoint{L, X, LIN}
    lh::L
    x::X
    lin::LIN
end

_at_point(lh::ComposedLikelihood, x) = _ComposedAtPoint(lh, x, _composed_linearization(lh, x))

_point(h::_ComposedAtPoint) = h.x
transformation(h::_ComposedAtPoint) = transformation(h.lh.likelihood, h.lin.value)

rightsqrtmetric(h::_ComposedAtPoint, v) =
    rightsqrtmetric(h.lh.likelihood, h.lin.value, pushforward(h.lin, v))

leftsqrtmetric(h::_ComposedAtPoint, η) =
    pullback(h.lin, leftsqrtmetric(h.lh.likelihood, h.lin.value, η))

fishermetric(h::_ComposedAtPoint, v) =
    pullback(h.lin, fishermetric(h.lh.likelihood, h.lin.value, pushforward(h.lin, v)))

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
