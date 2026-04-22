struct Samples{P<:Union{Nothing,AbstractArray},S<:Union{Nothing,AbstractArray},K}
    position::P
    residuals::S
    keys::K
end

Samples(position, residuals; keys=nothing) = Samples(position, residuals, keys)

function posterior_samples(samples::Samples)
    samples.residuals === nothing &&
        throw(ArgumentError("`Samples` does not contain any residual draws"))
    samples.position === nothing && return samples.residuals
    return _add_position(samples.position, samples.residuals)
end

_sample_count(residuals::AbstractArray) = size(residuals, 1)

Base.length(samples::Samples) = samples.residuals === nothing ? 0 : _sample_count(samples.residuals)

function _sample_slice(x::AbstractArray, i::Int)
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

function Base.iterate(samples::Samples, state::Int=1)
    state > length(samples) && return nothing
    return (samples[state], state + 1)
end

function recenter(samples::Samples, new_position)
    samples.residuals === nothing && return Samples(new_position, nothing; keys=samples.keys)
    shifted = posterior_samples(samples)
    return Samples(new_position, _subtract_position(shifted, new_position); keys=samples.keys)
end
