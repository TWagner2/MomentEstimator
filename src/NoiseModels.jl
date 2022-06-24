"""
Different error channels such as single qubit noise or correlated noise and different ways to assign random error rates to each qubit.
"""

abstract type AbstractErrorChannel end #Implements sample_errors, array_representation
function array_representation(ErrorRates::AbstractArray) #In case you are using error rates not wrapped in a channel
return ErrorRates
end
function load_channel(Source::Union{HDF5.File, HDF5.Group}, ErrorRates)
    try
        T = Symbol(attributes(Source)["Channeltype"][])
        Chan = load_channel(eval(T),Source,ErrorRates)
        return Chan
    catch e #Old format where single qubit error rates are not wrapped in a channel
        if isa(e, KeyError)
            @warn "Could not find channeltype"
            throw(e)
            return ErrorRates
        else
            throw(e)
        end
    end
end

#Sampling 
function floattocat(f,Cum)
    for i in 1:size(Cum,1)
        if f<Cum[i]
            return i
        end
    end
end
function sample_error(ErrorRates::Union{AbstractErrorChannel,AbstractMatrix};rng=Random.GLOBAL_RNG)
    return dropdims(sample_errors(1,ErrorRates;rng=rng),dims=2)
end
"""
sample_errors(m,ErrorRates)

Sample from n categorical distributions with given category probabilites.

# Arguments
- "Rates": An n_categories x n matrix giving the rates of each category for each i = 1,....,n
"""
function sample_categorical(m::Integer,Rates::AbstractMatrix; rng=Random.GLOBAL_RNG)
    n = size(Rates,2)
    Errors = rand(rng,n,m)
    Cum = accumulate(+,Rates;dims=1)
    for j = 1:m
        for i = 1:n
            Errors[i,j] = floattocat(Errors[i,j], @view(Cum[:,i]))
        end
    end
    Errors = Int.(Errors)
    return Errors
end 
"""
sample_errors(m,DataErrorRates; MeasurementErrorRates = nothing)

Sample errors according to the given data and measurement error rates.

# Arguments
- "DataErrorRates": An 4 x n matrix giving the error rates for each qubit, in order Z,X,Y as determined by indextogf4( , exchange=true)
- "MeasurementErrorRates": An 2 x m matrix giving the error rates for each measurement error in order 0, 1
"""
function sample_errors(m::Integer, DataErrorRates::AbstractMatrix, MeasurementErrorRates::Union{AbstractMatrix,Nothing} = nothing; rng=Random.GLOBAL_RNG)
    E = sample_categorical(m,DataErrorRates; rng=rng)
    E = indextogf4.(E,exchange = true)
    E = gf4tosymplectic_cat(E;exchange=true)
    if MeasurementErrorRates !== nothing
        E_m = sample_categorical(m,MeasurementErrorRates; rng=rng)
        E_m = BitMatrix((indextogf4.(E_m)))
        E = vcat(E, E_m)
    end
    return E
end
function sample_syndromes(m,H::SymplecticMatrix, DataErrorRates::AbstractMatrix, MeasurementErrorRates::Union{AbstractMatrix,Nothing} = nothing;rng=Random.GLOBAL_RNG)
    E = sample_errors(m,DataErrorRates, MeasurementErrorRates;rng=rng)
    S = symplectic_prod(H,E)
    return S
end
function sample_syndromes(m,C::QECC,DataErrorRates::AbstractMatrix, MeasurementErrorRates::Union{AbstractMatrix,Nothing} = nothing;rng=Random.GLOBAL_RNG)
    H=gf4tosymplectic_cat(C.H; n_d = C.n_d)
    @debug "H_symp" H
    return sample_syndromes(m,H,DataErrorRates, MeasurementErrorRates;rng=rng)
end
function sample_syndromes(m,H::SymplecticMatrix,Chan::AbstractErrorChannel;rng=Random.GLOBAL_RNG)
    E = sample_errors(m,Chan;rng=rng)
    S = symplectic_prod(H,E)
    return S
end
function sample_syndromes(m,C::QECC,Chan::AbstractErrorChannel;rng=Random.GLOBAL_RNG)
    H=gf4tosymplectic_cat(C.H; n_d = C.n_d)
    #@debug "H_symp" H
    return sample_syndromes(m,H,Chan;rng=rng)
end

"""
Independent single qubit and measurement errors
"""
struct Channel_SingleQubit<:AbstractErrorChannel
    ErrorRates::Matrix{Float64} #An 4 x n matrix giving the error rates for each qubit, in order X,Z,Y
    MeasurementErrorRates::Union{Matrix{Float64}, Nothing} #An 2 x m matrix giving the error rates for each measurement error
    Channel_SingleQubit(ErrorRates::Matrix{Float64}, MeasurementErrorRates::Union{Matrix{Float64}, Nothing} = nothing) = new(ErrorRates, MeasurementErrorRates)
end
function sample_errors(m::Integer,Chan::Channel_SingleQubit; rng=Random.GLOBAL_RNG)
    return sample_errors(m,Chan.ErrorRates, Chan.MeasurementErrorRates;rng=rng)
end
#Array representation of the error rates such that they can be saved if multiple channels are sampled
function array_representation(Chan::Channel_SingleQubit)
    R_d = Chan.ErrorRates
    R_m = Chan.MeasurementErrorRates
    if R_m !== nothing
        R_m = embed_measurementerrorrates(R_m) #Blow up to 4xm matrix
        R_d = hcat(R_d,R_m)
    end
    return R_d
end
#Load a channel from its array representation + params, read from a file
function load_channel(::Type{Channel_SingleQubit}, Source::Union{HDF5.File, HDF5.Group}, ErrorRates::AbstractMatrix{Float64})
    n_m = attributes(Source)["n_m"][]
    R_d = ErrorRates[:,1:end-n_m]
    R_m = (n_m > 0 ? ErrorRates[1:2,end-n_m+1:end] : nothing)
    return Channel_SingleQubit(R_d,R_m)
end
#All other parameters of the channel
function write_params(Dest::Union{HDF5.File, HDF5.Group,HDF5.Attributes},Chan::Channel_SingleQubit)
    Dest["Channeltype"] = String(Symbol(Channel_SingleQubit))
    Dest["n_m"] = ( Chan.MeasurementErrorRates !== nothing ? size(Chan.MeasurementErrorRates,2) : 0)
end

"""
Channel represented by a convolutional factor graph, i.e. independent noise processes on given supports are added.
Can be used to represent many kinds of correlated noise.
Facor Graph is on the symplectic representation.
"""
struct Channel_Factorized<:AbstractErrorChannel
    FG::FactorGraph
end
function sample_errors(m::Integer,C::Channel_Factorized; rng = Random.GLOBAL_RNG)
    Errors = falses((C.FG.n,m))
    n_factors = length(C.FG.Factors)
    f = rand(rng,m,n_factors)
    for (i,F) in enumerate(C.FG.Factors)
        len = length(F.Support)
        cum = accumulate(+,getdata(F.Values))
        inds = floattocat.(f[:,i],[cum])
        es = hcat([indextobitvector(x,len) for x in inds]...)
        @debug "" cum inds es len
        @debug "diff" [cum[i]-cum[i-1] for i in 2:2^len]
        Errors[F.Support,:] .⊻= es
    end
    return Errors
end
function array_representation(Chan::Channel_Factorized)
    return vcat([getdata(F.Values) for F in Chan.FG.Factors]...)
end
function load_channel(::Type{Channel_Factorized},Source::Union{HDF5.File, HDF5.Group}, ErrorRates::AbstractVector)
    n = attributes(Source)["n"][]
    Supports = attributes(Source)["ChannelSupports"][]
    i = 1
    Factors = []
    for s in Supports
        k = 2^(length(s))
        F = Factor(s,ErrorRates[i:i+k-1])
        push!(Factors,F)
        i += k
    end
    return Channel_Factorized(FactorGraph(n,Factors))
end
function write_params(Dest::Union{HDF5.File, HDF5.Group, HDF5.Attributes}, Chan::Channel_Factorized)
    Dest["Channeltype"] = String(Symbol(Channel_Factorized))
    Dest["ChannelSupports"] = HDF5.VLen([F.Support for F in Chan.FG.Factors])
    Dest["n"] = Chan.FG.n
end

"""
Two qubit channels on all qubits coupled by stabilizer generators
"""
function channel_nnsurfacecode(l,p1,p2)
    H = qecc_surfacecode_regular(l).H
    n = size(H,1)
    Supports = Vector{Int}[]
    for j in axes(H,2)
        inds = findall(H[:,j] .!= 0)
        @debug j inds
        for k in 1:length(inds)
            for l in 1:(k-1)
                push!(Supports, sort([inds[l],inds[k], inds[l]+n, inds[k]+n])) #Support on the symplectic representation
            end
        end
    end
    unique!(Supports)
    ErrorRates = BitlabeledVector{Float64}[]
    Rates = BitlabeledVector(Vector{Float64}(undef,16),4)
    #Single qubit error have probability p1/6, two qubit errors probability p2 / 9, need p1 + p2 <= 1
    for i in 1:16
            v = indextobitvector(i,4)
            if any(v[[1,3]]) && any(v[[2,4]]) #Two qubit error
                Rates[v] = p2/9
            elseif !any(v) #Zero error
                Rates[v] = 1-p1-p2
            else
                Rates[v] = p1 / 6
            end
    end
    for s in Supports
        push!(ErrorRates,Rates)
    end
    FG = FactorGraph(2*n,Supports,ErrorRates)
    return Channel_Factorized(FG)
end

#Random error rates according to https://www.nature.com/articles/s41534-021-00448-5
function compute_TVAD_twirled(t::AbstractFloat,T1::AbstractFloat)
    p_X = 1/4*(1 - exp(- t/T1))
    p_Y = p_X
    p_Z = 1/4*(1 + exp(- t/T1) - 2*exp(- t/(2*T1)))
    p_I = 1 - p_X - p_Y - p_Z
    return [p_I,p_Z,p_X,p_Y]
end

function sample_TVAD_twirled(t::AbstractFloat, DistrT1::Distribution)
    #sample T1 from a truncated gaussian
    T1 = rand(DistrT1)
    return compute_TVAD_twirled(t,T1)
end
"""
sample_TVAD_twirled(t::AbstractFloat, μ_T1::AbstractFloat, σ_T1::AbstractFloat)

Sample a realization of the time varying amplitude damping Pauli twirled channel defined in https://www.nature.com/articles/s41534-021-00448-5 

T1 is sampled from a truncated normal characterized by μ_T1 and σ_T1
"""
function sample_TVAD_twirled(t::AbstractFloat, μ_T1::AbstractFloat, σ_T1::AbstractFloat)
    DistrT1 = Distributions.TruncatedNormal(μ_T1,σ_T1,0,Inf)
    return sample_TVAD_twirled(t,DistrT1)
end
function sample_TVAD_twirled(t::AbstractFloat, μ_T1::AbstractFloat, σ_T1::AbstractFloat, n::Integer)
    DistrT1 = Distributions.TruncatedNormal(μ_T1,σ_T1,0,Inf)
    return sample_TVAD_twirled(t,DistrT1,n)
end
"""
sample_TVAD_twirled(t::AbstractFloat, DistrT1::Distribution, n::Integer)

Sample n realizations of the time varying amplitude damping Pauli twirled channel defined in https://www.nature.com/articles/s41534-021-00448-5 
"""
function sample_TVAD_twirled(t::AbstractFloat, DistrT1::Distribution, n::Integer)
    Rates = zeros((4,n))
    for i in axes(Rates,2)
        Rates[:,i] = sample_TVAD_twirled(t,DistrT1)
    end
    return Rates
end

function compute_TVAPD_twirled(t::AbstractFloat, T1::AbstractFloat,Tphi::AbstractFloat)
    T2inv = 1/(2*T1) + 1/(Tphi) # = 1/T2
    p_X = 1/4*(1 - exp(- t/T1))
    p_Y = p_X
    p_Z = 1/4*(1 + exp(- t/T1) - 2*exp(- t*T2inv))
    p_I = 1 - p_X - p_Y - p_Z
    return [p_I,p_Z,p_X,p_Y]
end

function sample_TVAPD_twirled(t::AbstractFloat, DistrT1::Distribution, DistrTphi::Distribution)
    T1 = rand(DistrT1)
    Tphi = rand(DistrTphi)
    return compute_TVAPD_twirled(t,T1,Tphi)
end
"""
sample_TVAPD_twirled(t::AbstractFloat, μ_T1::AbstractFloat, σ_T1::AbstractFloat, μ_Tphi::AbstractFloat, σ_Tphi::AbstractFloat)

Sample a realization of the time varying combined amplitude and phase damping Pauli twirled channel defined in https://www.nature.com/articles/s41534-021-00448-5 

T1 and Tphi are sampled from truncated normal distributions characterized by μ_T1, σ_T1 and μ_Tphi, σ_Tphi
"""
function sample_TVAPD_twirled(t::AbstractFloat, μ_T1::AbstractFloat, σ_T1::AbstractFloat, μ_Tphi::AbstractFloat, σ_Tphi::AbstractFloat)
    DistrT1 = Distributions.TruncatedNormal(μ_T1,σ_T1,0,Inf)
    DistrTphi = Distributions.TruncatedNormal(μ_Tphi,σ_Tphi,0,Inf)
    return sample_TVAPD_twirled(t,DistrT1, DistrTphi)
end
function sample_TVAPD_twirled(t::AbstractFloat, μ_T1::AbstractFloat, σ_T1::AbstractFloat, μ_Tphi::AbstractFloat, σ_Tphi::AbstractFloat, n::Integer)
    DistrT1 = Distributions.TruncatedNormal(μ_T1,σ_T1,0,Inf)
    DistrTphi = Distributions.TruncatedNormal(μ_Tphi,σ_Tphi,0,Inf)
    return sample_TVAPD_twirled(t,DistrT1,DistrTphi,n)
end
"""
sample_TVAPD_twirled(t::AbstractFloat, DistrT1::Distribution, DistrTphi::Distribution, n::Integer)

Sample n realizations of the time varying combined amplitude and phase damping Pauli twirled channel defined in https://www.nature.com/articles/s41534-021-00448-5 
"""
function sample_TVAPD_twirled(t::AbstractFloat, DistrT1::Distribution, DistrTphi::Distribution, n::Integer)
    Rates = zeros((4,n))
    for i in axes(Rates,2)
        Rates[:,i] = sample_TVAPD_twirled(t,DistrT1,DistrTphi)
    end
    return Rates
end


#Sampler definitions
abstract type AbstractChannelSampler end

"""
Sample random channel for each qubit according to TVAPD model from https://www.nature.com/articles/s41534-021-00448-5
"""
struct ChannelSampler_TVAPDTwirled <: AbstractChannelSampler
    t::Float64
    μ_T1::Float64
    μ_Tphi::Float64
    σ_T1::Float64
    σ_Tphi::Float64
    DistrT1::Distribution
    DistrTphi::Distribution
    SamplePerSimulation::Bool #If this is true, a new channel is sampled per simulation, otherwise a new channel is sampled each time a syndrome is drawn
    #ChannelSampler::Function
    function ChannelSampler_TVAPDTwirled(; t::Float64, μ_T1::Float64, μ_Tphi::Float64,σ_T1::Float64,σ_Tphi::Float64, SamplePerSimulation::Bool)
        DistrT1 = TruncatedNormal(μ_T1,σ_T1, 0, Inf)
        DistrTphi = TruncatedNormal(μ_Tphi,σ_Tphi,0,Inf)
        new(t,μ_T1,μ_Tphi,σ_T1,σ_Tphi, DistrT1, DistrTphi, SamplePerSimulation)#, ChannelSampler)
    end
end
function write_params(Dest::Union{HDF5.File,HDF5.Group},Parameters::ChannelSampler_TVAPDTwirled)
    FileParams = attributes(Dest)
    FileParams["t"] = Parameters.t
    FileParams["mu_T1"] = Parameters.μ_T1
    FileParams["sigma_T1"] = Parameters.σ_T1
    FileParams["mu_Tphi"] = Parameters.μ_Tphi
    FileParams["sigma_Tphi"] = Parameters.σ_Tphi
    FileParams["SamplePerSimulation"] = Parameters.SamplePerSimulation
    FileParams["Channelsamplertype"] = "TVAPDTwirled"
    FileParams["Channeltype"] = String(Symbol(Channel_SingleQubit))
    FileParams["n_m"] = 0
end
function sample_channel(Sampler::ChannelSampler_TVAPDTwirled, n::Integer)
    return sample_TVAPD_twirled(Sampler.t,Sampler.DistrT1,Sampler.DistrTphi,n) 
end

"""
Sample random channel for each qubit according to TVAD model from https://www.nature.com/articles/s41534-021-00448-5
"""
struct ChannelSampler_TVADTwirled <: AbstractChannelSampler
    t::Float64
    μ_T1::Float64
    σ_T1::Float64
    DistrT1::Distribution
    SamplePerSimulation::Bool #If this is true, a new channel is sampled per simulation, otherwise a new channel is sampled each time a syndrome is drawn
    function ChannelSampler_TVADTwirled(; t::Float64, μ_T1::Float64, σ_T1::Float64, SamplePerSimulation::Bool)
        DistrT1 = TruncatedNormal(μ_T1,σ_T1, 0, Inf)
        new(t,μ_T1,σ_T1,DistrT1, SamplePerSimulation)#, ChannelSampler)
    end
end
function write_params(Dest::Union{HDF5.File,HDF5.Group},Parameters::ChannelSampler_TVADTwirled)
    FileParams = attributes(Dest)
    FileParams["t"] = Parameters.t
    FileParams["mu_T1"] = Parameters.μ_T1
    FileParams["sigma_T1"] = Parameters.σ_T1
    FileParams["SamplePerSimulation"] = Parameters.SamplePerSimulation
    FileParams["Channelssamplertype"] = "TVADTwirled"
    FileParams["Channeltype"] = String(Symbol(Channel_SingleQubit))
    FileParams["n_m"] = 0
end
function sample_channel(Sampler::ChannelSampler_TVADTwirled, n::Integer)
    return sample_TVAD_twirled(Sampler.t,Sampler.DistrT1,n)
end

"""
Sample the same channel every time.
"""
struct ChannelSampler_Constant{T} <: AbstractChannelSampler where {T<:AbstractErrorChannel}
    Chan::T
    SamplePerSimulation::Bool
    function ChannelSampler_Constant(Chan::T) where {T<:AbstractErrorChannel}
        return new{T}(Chan,true)
    end
end
function ChannelSampler_Constant(;p, n)
    return ChannelSampler_Constant{Channel_SingleQubit}(Channel_SingleQubit(n, expand_errorrates(p,n)))
end
function write_params(Dest::Union{HDF5.File,HDF5.Group},Parameters::ChannelSampler_Constant)
    FileParams = attributes(Dest)
    write_params(FileParams,Parameters.Chan)
    FileParams["Channelsamplertype"] = "Const"
end
function sample_channel(Sampler::ChannelSampler_Constant, n::Integer)
    return Sampler.Chan 
end