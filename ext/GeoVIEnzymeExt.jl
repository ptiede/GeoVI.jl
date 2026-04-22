module GeoVIEnzymeExt

using Enzyme
import ADTypes: AutoEnzyme
import GeoVI

function _forward_to!(y, x, forward)
    copyto!(y, forward(x))
    return nothing
end

struct _EnzymeLinearization{F,X,V}
    forward::F
    x::X
    value::V
end

function GeoVI.pushforward(lin::_EnzymeLinearization, v::AbstractArray)
    dres, _ = Enzyme.autodiff(
        Enzyme.ForwardWithPrimal,
        lin.forward,
        Enzyme.Duplicated,
        Enzyme.Duplicated(lin.x, v),
    )
    return dres
end

function GeoVI.pullback(lin::_EnzymeLinearization, η::AbstractArray)
    dx = zero(lin.x)
    dy = copy(η)
    y = zero(lin.value)
    Enzyme.autodiff(
        Enzyme.Reverse,
        _forward_to!,
        Enzyme.Duplicated(y, dy),
        Enzyme.Duplicated(lin.x, dx),
        Enzyme.Const(lin.forward),
    )
    return dx
end

function GeoVI._automatic_linearize(
    ::AutoEnzyme,
    forward,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    return _EnzymeLinearization(forward, x, forward(x))
end

function GeoVI._value_and_gradient(
    ::AutoEnzyme,
    objective,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, objective, x)
    return result.val, result.derivs[1]
end

end
