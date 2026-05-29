struct Samples{P <: Union{Nothing, AbstractArray}, S <: Union{Nothing, AbstractArray}, K}
    position::P
    residuals::S
    keys::K
end

Samples(position, residuals; keys = nothing) = Samples(position, residuals, keys)

function posterior_samples(samples::Samples)
    samples.residuals === nothing &&
        throw(ArgumentError("`Samples` does not contain any residual draws"))
    samples.position === nothing && return samples.residuals
    return _add_position(samples.position, samples.residuals)
end

_sample_count(residuals::AbstractArray) = size(residuals, 1)

Base.length(samples::Samples) = samples.residuals === nothing ? 0 : _sample_count(samples.residuals)

function _sample_slice(x::AbstractArray, i)
    # Note: `i` is an integer-typed scalar. We omit a `::Integer`
    # annotation because under `@trace for` it arrives as a
    # `Reactant.TracedRNumber{<:Integer}`, which does not subtype
    # `Integer`. The implementation is identical in either case.
    tail = ntuple(_ -> Colon(), max(ndims(x) - 1, 0))
    return x[i, tail...]
end

function _add_position(position::AbstractArray, residuals::AbstractArray)
    return residuals .+ reshape(position, (1, size(position)...))
end

function _subtract_position(samples::AbstractArray, position::AbstractArray)
    return samples .- reshape(position, (1, size(position)...))
end

function Base.getindex(samples::Samples, i::Int)
    1 <= i <= length(samples) || throw(BoundsError(samples, i))
    draw = _sample_slice(samples.residuals, i)
    return samples.position === nothing ? draw : draw .+ samples.position
end

function Base.iterate(samples::Samples, state::Int = 1)
    state > length(samples) && return nothing
    return (samples[state], state + 1)
end

function recenter(samples::Samples, new_position)
    samples.residuals === nothing && return Samples(new_position, nothing; keys = samples.keys)
    shifted = posterior_samples(samples)
    return Samples(new_position, _subtract_position(shifted, new_position); keys = samples.keys)
end

"""
    VariationalPosterior(likelihood, position, family, samples)

The fitted variational distribution. `position` is the latent mean; `family`
records how to draw from it. The `samples` drawn during fitting are retained
and accessible, but the posterior is a *distribution*: call `rand` to draw
arbitrary new samples and `mean` to get the latent mean.
"""
struct VariationalPosterior{L, P, F, S}
    likelihood::L
    position::P
    family::F
    samples::S
end

mean(post::VariationalPosterior) = post.position

function Base.rand(rng::AbstractRNG, post::VariationalPosterior)
    return post.position .+
        _draw_one_residual(post.family, post.likelihood, post.position, rng)
end

Base.rand(post::VariationalPosterior) = rand(Random.default_rng(), post)

function Base.rand(rng::AbstractRNG, post::VariationalPosterior, n::Integer)
    n >= 0 || throw(ArgumentError("`n` must be non-negative"))
    # Each draw has the shape of the latent mean, so allocate straight from
    # `post.position` and fill all `n` rows in the loop (no special-cased first
    # draw). `@trace for` so a *compiled* `rand` builds a single MLIR while-loop
    # body instead of `n` trace-time-unrolled copies of the residual-draw graph
    # (a CG solve, and for `GeoVIFamily` the full nonlinear curve). On the host
    # path `@trace for` is a plain loop, so this is unchanged there.
    # `track_numbers = false` keeps the deep type-walk away from the plain
    # Int/Bool fields in the family/likelihood closure environment.
    out = similar(post.position, (n, size(post.position)...))
    trailing = ntuple(_ -> Colon(), ndims(post.position))
    @trace track_numbers = false for i in 1:n
        out[i, trailing...] = rand(rng, post)
    end
    return out
end

Base.rand(post::VariationalPosterior, n::Integer) = rand(Random.default_rng(), post, n)
