module GeoVIEnzymeExt

using Enzyme
import ADTypes: AutoEnzyme
import GeoVI

function _forward_to!(y, x, forward)
    copyto!(y, forward(x))
    return nothing
end

struct _EnzymeLinearization{F, X, V}
    forward::F
    x::X
    value::V
end

# Both directions differentiate the same in-place `_forward_to!`, differing only in mode
# and which shadow is seeded/read: forward seeds the input tangent `v` and reads the JVP
# out of the output shadow `dy`; reverse (below) seeds the output cotangent `η` and reads
# the VJP out of the input shadow `dx`.
function GeoVI.pushforward(lin::_EnzymeLinearization, v::AbstractArray)
    y = zero(lin.value)
    dy = zero(lin.value)
    Enzyme.autodiff(
        Enzyme.Forward,
        _forward_to!,
        Enzyme.Duplicated(y, dy),
        Enzyme.Duplicated(lin.x, v),
        Enzyme.Const(lin.forward),
    )
    return dy
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
        fd_eps::Real = 1.0e-6,
    )
    return _EnzymeLinearization(forward, x, forward(x))
end

function GeoVI._value_and_gradient(
        ::AutoEnzyme,
        objective,
        x::AbstractArray;
        fd_eps::Real = 1.0e-6,
    )
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, Enzyme.Const(objective), x)
    return result.val, result.derivs[1]
end

end
