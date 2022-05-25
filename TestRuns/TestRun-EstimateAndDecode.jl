include("../Main.jl")
n_simulation = 1
l = 3
project = true
muT1 = 80e-6
muTphi = 57e-6
sigmaT1 = 35e-6
sigmaTphi = 26e-6
t = 5e-6 #15e-6
beta=0.0001
n_step=2
SamplePerSimulation=true
regularize=false
fullcovariance=false
n_estimate = [10000]
Options = Optim.Options()

n_test = 10^4
χ=20
τ=60
Decode_actual = true
Decode_averaged = true

reset_timer!()

Debug=false
if Debug
    @everywhere global_logger(ConsoleLogger(Logging.Debug,show_limited=false))
else
    @everywhere global_logger(ConsoleLogger(Logging.Info,show_limited=false))
end

@info "Parameters" n_simulation l project muT1 muTphi sigmaT1 sigmaTphi t n_estimate beta n_step SamplePerSimulation regularize fullcovariance χ τ

n_qubit = size(qecc_surfacecode_regular(l).H,1)

Code = qeccgraph_surfacecode_regular(l)
LocalEstimator = Estimator_lsq_optim(β=beta, n_step=n_step, Options = Options)
#LocalEstimator = Estimator_pinv(β=beta)
ChannelSampler = ChannelSampler_TVAPDTwirled(t=t,μ_T1 = muT1, μ_Tphi = muTphi, σ_T1 = sigmaT1, σ_Tphi = sigmaTphi, SamplePerSimulation = SamplePerSimulation)
if regularize
    #Estimate the mean and variance of the moments for given channel params and use it to regularize, in principle this could also be computed analytically by averaging the TVAPD(T1,T2) channel over T1,T2
    SampleP = sample_TVAPD_twirled(t,muT1, sigmaT1, muTphi,sigmaTphi, 10^7)
    SampleMom = momentsfromrates(SampleP)[2:end,:]
    Mean = dropdims(mean(SampleMom;dims=2);dims=2)
    if !fullcovariance
	    Var = dropdims(var(SampleMom;dims=2);dims=2) #We could also estimate the full covariance matrix instead but it will be singular because it only has 2 parameters
	    Regularizer = TiledRegularizer_L2_Repeating(Mean,(1 ./ Var))
    else
	    Cov = cov(SampleMom')
        Regularizer = TiledRegularizer_L2_Repeating(Mean, pinv(Cov))
    end
    @info "Regularizer Weight" Regularizer.Wt
else
    Regularizer = nothing
end

EstimatorParams = SimulationParameters_estimator(C = Code,f_neighborhops = surfacecode_hops, LocalEstimator=LocalEstimator,ChannelSampler=ChannelSampler,Regularizer=Regularizer,n_simulations=n_simulation,n_estimate=n_estimate,project=project)
Params = SimulationParameters_EstimateAndDecode(EstimatorParams=EstimatorParams,n_test=n_test,χ=χ,τ=τ,Decode_actual=Decode_actual)
#Params=SimulationParameters_EstimateAndDecode(LocalEstimator=LocalEstimator,ChannelSampler=ChannelSampler,Regularizer=Regularizer,n_simulations=n_simulation,n_estimate=n_estimate,l=l,project=project, Codetype = "Surface code square regular", n_test = n_test, χ= χ, τ=τ, Decode_actual=Decode_actual)

estimateanddecode_simulation("Test-decode.hdf5", Params)	

#File = h5open("Test-decode.hdf5","r")

print_timer(allocations=false)

if Decode_averaged
    decodewithaveragechannel_simulation("Test-decode.hdf5")
end
File = h5open("Test-decode.hdf5","r")
for n_est in n_estimate
    Est = File["n_est=$n_est"]["EstimatedErrorRates"][]
    Actual = File["n_est=$n_est"]["ActualErrorRates"][]
    L1 = L1Error(Est[:,5],Actual[:,5])
    println("L1 error on qubit 5: $L1")
    println("Logical")
    display(File["n_est=$n_est"]["LogicalRates"][])
    if Decode_actual
        println("Logical Perfect Knowledge")
        display(File["n_est=$n_est"]["LogicalRatesPerfectKnowledge"][])
    end
    if Decode_averaged
        println("Logical with averaged channel")
        display(File["n_est=$n_est"]["LogicalRatesAveragedChannel"][])
    end
end
close(File)