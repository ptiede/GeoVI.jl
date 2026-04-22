randn_like(rng::AbstractRNG, x::AbstractArray) =
    randn(rng, eltype(x), size(x))
