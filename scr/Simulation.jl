"""
Some methods to simulate estimators for Pauli noise and feed the results into a decoder.
""" 

"""
function run_estimator(S::AbstractMatrix{Bool}, C::QECC, Estimator::AbstractEstimator; project=true)

Run estimator on code C and sampled syndomes S and return the results ordered by qubits and measurement errors.
Optionally projects the estimates onto the probability simplex.
"""
function run_estimator(S::AbstractMatrix{Bool}, C::QECC, Estimator::AbstractEstimator; project=true)
    Estimate = estimate(C, S, Estimator)
    Moments_d, Moments_m = momentsbyqubits(Estimate, n_d = C.n_d) #order by qubit
    @debug "Moments" Moments_d, Moments_m
    if project
        Moments_d,Rates_d = project_momentsrates(Moments_d)
        Moments_m,Rates_m = project_momentsrates(Moments_m)
    else
        Rates_d = ratesfrommoments(Moments_d)
        Rates_m = ratesfrommoments(Moments_m)
    end
    Rates_m = embed_measurementerrorrates(Rates_m) # Blow up to match data rates
    Rates = hcat(Rates_d,Rates_m)
    Moments_m = embed_measurementmoments(Moments_m)
    @debug " " Moments_d, Moments_m
    Moments = hcat(Moments_d,Moments_m)
    @debug " " Moments
    return (Rates, Moments)
end
function run_estimator(n_s::Integer, ErrorRates::Union{AbstractMatrix, AbstractErrorChannel}, C::QECC, Estimator::AbstractEstimator; project=true, rng=Random.GLOBAL_RNG)
    S = sample_syndromes(n_s,C,ErrorRates;rng=rng)
    t_estimate = @elapsed begin
        R = run_estimator(S,C,Estimator;project=project)
    end 
    @info "t_estimate: $t_estimate"
    return R
end
"""
How many hops to use in local regions for surface code.
"""
function surfacecode_hops(H::PauliMatrix)
    hops = Int[]
    for i in axes(H,1)
        if length(adjacent(H,i,true)) <= 3
            push!(hops,4)
        else
            push!(hops,2)
        end
    end
    return hops
end
function surfacecode_hops_repeated(H::PauliMatrix)
    hops = Int[]
    for i in axes(H,1)
        if length(adjacent(H,i,true)) <= 6
            push!(hops,4)
        else
            push!(hops,2)
        end
    end
    return hops
end
function checkfilepath(Filepath::String)
    if isfile(Filepath)
        error("Invalid path, data would be overwritten")
    end
    try #Check if a file can be created at the location
        f = open(Filepath,"w")
        close(f)
        rm(Filepath)
    catch
        error("Invalid path, file cannot be opened")
    end
end

"""
Parameters for a simulation of an estimator under Pauli noise.
"""
Parameters.@with_kw struct SimulationParameters_estimator
    LocalEstimator::AbstractEstimator
    ChannelSampler::AbstractChannelSampler
    Regularizer::Union{AbstractTiledRegularizer, Nothing}
    n_simulations::Int
    n_estimate::Vector{Int}
    C::Union{QECC,QECCGraph}
    project::Bool
    f_neighborhops::Function #A function returning for each qubit how many hops should be considered for the neighborhood. E.g. surfacecode_hops
end
function write_params(Dest::Union{HDF5.File,HDF5.Group},Params::SimulationParameters_estimator)
    @unpack_SimulationParameters_estimator Params
    FileParams = attributes(Dest)
    FileParams["n_simulations"] = n_simulations
    FileParams["project"] = project
    FileParams["n_estimate"] = n_estimate
    write_params(create_group(Dest, "Code"), C)
    write_params(create_group(Dest, "Estimator"),LocalEstimator)
    write_params(create_group(Dest, "Channel"),ChannelSampler)
    if Regularizer !== nothing
        write_params(create_group(Dest, "Regularizer"),Regularizer)
    end
end
function estimator_simulation(Dest::Union{HDF5.File,HDF5.Group}, Params::SimulationParameters_estimator; printintervall = 50, write = true)
    @unpack_SimulationParameters_estimator Params
    if isa(C,QECCGraph)
        C = C.C #The QECC underyling the QECC graph
    end
    H = C.H
    n = size(H,1)
    n_hop=f_neighborhops(H)
    if write
        Dest["n_hop"] = n_hop
        write_params(Dest,Params)
    end
    Estimator = Estimator_applyall_local_nhop(LocalEstimator=LocalEstimator,n_hop=n_hop,Regularizer=Regularizer)
    for n_est in n_estimate
        @info "n_estimate: $n_est" 
        if ChannelSampler.SamplePerSimulation
            repdim = ndims(array_representation(sample_channel(ChannelSampler,n)))
            EstimatedRates, ActualRates = @distributed (x,y) -> (cat(x[1],y[1],dims=3), cat(x[2],y[2],dims=repdim+1)) for i in 1:n_simulations #Returns tuple with first entry estimated data error rates, second entry estimated measurement error rates, third entry the actual channels
                if i%printintervall == 0
                    @info "i=$i" 
                end
                Chan = sample_channel(ChannelSampler,n)
                r,m = run_estimator(n_est,Chan,C, Estimator; project = project)
                (r,array_representation(Chan))
            end
        else
            EstimatedRates = @distributed (x,y) -> cat(x,y,dims=3) for i in 1:n_simulations
                if i%printintervall == 0
                    @info "i=$i" 
                end
                r,m = run_estimator(n_est,ChannelSampler,C,Estimator; project = project)
                r
            end
        end
        if n_simulations == 1
            EstimatedRates = reshape(EstimatedRates, (size(EstimatedRates)...,1)) #Consistency in number of dimensions
            if ChannelSampler.SamplePerSimulation
                ActualRates = reshape(ActualRates, (size(ActualRates)...,1))
            end
        end
        create_group(Dest,"n_est=$n_est")
        Dest["n_est=$n_est"]["EstimatedErrorRates"] = EstimatedRates
        if ChannelSampler.SamplePerSimulation
            Dest["n_est=$n_est"]["ActualErrorRates"] = ActualRates
        end
    end
    #return EstimatedRates, ActualRates
end
function estimator_simulation(Filepath::String, Params::SimulationParameters_estimator; printintervall=50)
    checkfilepath(Filepath)
    h5open(Filepath,"w") do Dest
        estimator_simulation(Dest,Params;printintervall=printintervall)
    end
end

"""
Simulation of the sweep decoder, decoder input rates and actual channel can differ.
"""
Parameters.@with_kw struct SimulationParameters_Decode
    n_simulations::Int
    n_test::Int
    χ::Int = 20
    τ::Int = 60
    ChannelSampler::AbstractChannelSampler
    C::QECCGraph
    Decoder_Input_Rates::Union{AbstractMatrix} #Some Input rates for decoder
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, Params::SimulationParameters_Decode)
    @unpack_SimulationParameters_Decode Params
    FileParams = attributes(Dest)
    write_params(create_group(Dest,"Channel"),ChannelSampler)
    write_params(create_group(Dest,"Code"),C)
    FileParams["n_simulations"] = n_simulations
    FileParams["n_test"] = n_test
    FileParams["chi"] = χ
    FileParams["tau"] = τ
    Dest["Decoder_Input_Rates"] = Decoder_Input_Rates
end
function decode_simulation(FilePath::String,Params::SimulationParameters_Decode)
    checkfilepath(FilePath)
    File = h5open(FilePath, "w")
    decode_simulation(File,Params)
    close(File)
end
function decode_simulation(Dest::Union{HDF5.File,HDF5.Group}, Params::SimulationParameters_Decode)
    @unpack_SimulationParameters_Decode Params
    write_params(Dest, Params)
    #C = qecc_surfacecode_regular(l)
    n,m = size(C.H)
    @info "Decoding"
    t_decode = @elapsed begin
    repdim = ndims(array_representation(sample_channel(ChannelSampler,n)))
    LogicalRates,ErrorRates = @distributed (a,b) -> (cat(a[1],b[1],dims=1) , cat(a[2],b[2],dims=repdim+1)) for i = 1:n_simulations
        Chan = sample_channel(ChannelSampler,n)
        (logicalerrorrate_sweepdecoder(C, Chan, n_test; ErrorRatesDecoder = Decoder_Input_Rates, χ = χ, τ = τ), array_representation(Chan))
    end
    end #elapsed
    @info "Decoding time: $t_decode"
    t_write_LogicalRates = @elapsed begin
    Dest["LogicalRates"] = LogicalRates
    Dest["ErrorRates"] = ErrorRates
    end # elapsed    
    @info "t_write_LogicalRates: $t_write_LogicalRates"
end

"""
First estimate error rates, then feed the result into sweep decoder anddetermin logical error rates.
Can optionally also decode with actual error rates for comparison.
"""
Parameters.@with_kw struct SimulationParameters_EstimateAndDecode
    EstimatorParams::SimulationParameters_estimator
    n_test::Int
    χ::Int = 20
    τ::Int = 60
    Decode_actual::Bool = true
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, Params::SimulationParameters_EstimateAndDecode)
    @unpack_SimulationParameters_EstimateAndDecode Params
    FileParams = attributes(Dest)
    write_params(Dest,EstimatorParams)
    FileParams["n_test"] = n_test
    FileParams["chi"] = χ
    FileParams["tau"] = τ
    FileParams["Decode_actual"] = Decode_actual
end
"""
estimateanddecode_surfacecode_simulation(FilePath::String, Parameters::EstimateAndDecode_Parameters)

Start an estimate and decode simulation and write the results to a hdf5 file at given path.

TODO: SamplePerSimulation = false is not supported yet, but it can simply be replaced by sampling from the averaged channel
"""
function estimateanddecode_simulation(FilePath::String,Params::SimulationParameters_EstimateAndDecode)
    checkfilepath(FilePath)
    File = h5open(FilePath, "w")
    estimateanddecode_simulation(File,Params)
    close(File)
end
#TODO: Does not support arbitrary channels for decode_actual, since our decoder can only handle single qubit noise. Furthermore it does not deal with measurement noise
function estimateanddecode_simulation(Dest::Union{HDF5.File,HDF5.Group}, Params::SimulationParameters_EstimateAndDecode; printintervall_estimate = 1000)
    @unpack_SimulationParameters_EstimateAndDecode Params
    if !ChannelSampler.SamplePerSimulation
        error("Only SamplePerSimulation=true is supported at the moment")
    end
    C = EstimatorParams.C
    if !isa(C,QECCGraph)
        throw("Only decoding of QECCGraphs is supported.")
    end
    n,m = size(C.H)
    #ErrorRates = expand_errorrates(Parameters.p,n)
    n_hop = EstimatorParams.f_neighborhops(C.H)
    write_params(Dest, Params)
    Dest["n_hop"] = n_hop
    estimator_simulation(Dest,EstimatorParams;write=false,printintervall = printintervall_estimate)
    @info "Decode" Distributed.nprocs()
    #Could be optimized by pre-allocating memory on each process and sending error rates there in one chunk?
    for n_est in EstimatorParams.n_estimate
        @info "Decoding: n_est=$n_est"
        ActualRates = Dest["n_est=$n_est"]["ActualErrorRates"][]
        EstimatedRates = Dest["n_est=$n_est"]["EstimatedErrorRates"][]
        @info size(ActualRates)
        Channels = [load_channel(Dest["Channel"],selectdim(ActualRates,ndims(ActualRates),i)) for i = 1:EstimatorParams.n_simulations] #Cannot load in separate processes because hdf5 reading does not support multiprocessing, there might still be a better solution than this
        t_decode_estimated = @elapsed begin
        LogicalRates = @distributed (a,b) -> cat(a,b,dims=1) for i = 1:EstimatorParams.n_simulations
            logicalerrorrate_sweepdecoder(C, Channels[i], n_test; ErrorRatesDecoder = EstimatedRates[:,:,i], χ = χ, τ = τ)
            #logicalerrorrate_sweepdecoder(C, ActualRates[:,:,i], n_test; ErrorRatesDecoder = EstimatedRates[:,:,i], χ = χ, τ = τ)
        end
        end #elapsed
        @info "Decoding time 1: $t_decode_estimated"
        t_write_LogicalRates = @elapsed begin
        Dest["n_est=$n_est"]["LogicalRates"] = LogicalRates
        end # elapsed    
        @info "t_write_LogicalRates: $t_write_LogicalRates"
        if Decode_actual
            t_decode_actual = @elapsed begin
                #Could be optimized by pre-allocating memory on each process and sending error rates there in one chunk?
                LogicalRatesActual = @distributed (a,b) -> cat(a,b,dims=1) for i = 1:EstimatorParams.n_simulations
                    logicalerrorrate_sweepdecoder(C, ActualRates[:,:,i], n_test; χ = χ, τ = τ) #Decode using the true error rates
                end
                end #elapsed
                @info "Decoding time 2: $t_decode_actual"
                t_write_LogicalRates = @elapsed begin
                Dest["n_est=$n_est"]["LogicalRatesPerfectKnowledge"] = LogicalRatesActual
                end # elapsed    
                @info "t_write_LogicalRates: $t_write_LogicalRates"
        end
    end
end

"""
Takes a file generated by estimateanddecode_simulation for a time dependent channel and decodes with the averaged channel
"""
function decodewithaveragechannel_simulation(Filepath::String)
    Dest = h5open(Filepath,"r+")
    return decodewithaveragechannel_simulation(Dest)
end
function decodewithaveragechannel_simulation(Dest::Union{HDF5.File,HDF5.Group})
    #l = attributes(Dest)["l"][]
    Channeltype = attributes(Dest["Channel"])["Channelsamplertype"][]
    #C = qecc_surfacecode_regular(l)
    Code = QECC(attributes(Dest["Code"])["Type"][], Dest["Code"]["Stabilizers"][], attributes(Dest["Code"])["n_data"][])
    C = QECCGraph(Code,Dest["Code"]["CheckEmbedding"][],Dest["Code"]["QubitEmbedding"][],Dest["Code"]["LogicalOperators"][])
    n,m = size(C.H)

    if Channeltype == "TVAPDTwirled"
        t = attributes(Dest["Channel"])["t"][]
        muT1 = attributes(Dest["Channel"])["mu_T1"][]
        muTphi = attributes(Dest["Channel"])["mu_Tphi"][]
        sigmaT1 = attributes(Dest["Channel"])["sigma_T1"][]
        sigmaTphi = attributes(Dest["Channel"])["sigma_Tphi"][]
        SampleP = sample_TVAPD_twirled(t,muT1, sigmaT1, muTphi,sigmaTphi, 10^7)
        AverageP = dropdims(mean(SampleP;dims=2);dims=2)
        AverageP = expand_errorrates(AverageP,n)
    else
        throw("Unsupported channel type")
    end
    #TODO: other channel types
    n_estimate = attributes(Dest)["n_estimate"][]
    n_test = attributes(Dest)["n_test"][]
    χ=attributes(Dest)["chi"][]
    τ=attributes(Dest)["tau"][]
    for n_est in n_estimate
        ActualRates = Dest["n_est=$n_est"]["ActualErrorRates"][]
        n_simulations = size(ActualRates,3)
        t_decode = @elapsed begin
        #Could be optimized by pre-allocating memory on each process and sending error rates there in one chunk?
        LogicalRates = @distributed (a,b) -> cat(a,b,dims=1) for i = 1:n_simulations
            logicalerrorrate_sweepdecoder(C, ActualRates[:,:,i], n_test; ErrorRatesDecoder = AverageP, χ = χ, τ = τ)
        end
        end #elapsed
        @info "Decoding time: $t_decode"
        t_write_LogicalRates = @elapsed begin
        Dest["n_est=$n_est"]["LogicalRatesAveragedChannel"] = LogicalRates
        end # elapsed    
        @info "t_write_LogicalRates: $t_write_LogicalRates"
    end
end