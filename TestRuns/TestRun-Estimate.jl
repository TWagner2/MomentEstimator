include("../Main.jl")
n_simulation = 1
l = 3
project = true
muT1 = 80e-6
muTphi = 57e-6
sigmaT1 = 35e-6
sigmaTphi = 26e-6
t = 5e-6 #15e-6
beta=0.001
n_step=1
SamplePerSimulation=true
regularize=false
fullcovariance=false
select=  2000 #nothing #2000
n_estimate = [100000]
Options = Optim.Options()

#Code = qecc_surfacecode_regular(l)
#f_neighborhops = surfacecode_hops
#ChannelSampler = ChannelSampler_Constant(Channel_SingleQubit(expand_errorrates(0.1,13)))
Code = qecc_surfacecode_regular(l)
Code = qecc_repeat_measurements(Code,2)
f_neighborhops = surfacecode_hops_repeated
ChannelSampler = ChannelSampler_Constant(Channel_SingleQubit(expand_errorrates(0.01,13),expand_errorrates([0.98,0.02],24)))

#LocalEstimator = Estimator_pinv(β=beta, n_qubit = Code.n_d)
LocalEstimator = Estimator_lsq_optim(β=beta, n_step=n_step, Options=Options,select=select)


Debug=true

if Debug
    @everywhere global_logger(ConsoleLogger(Logging.Debug,show_limited=false))
else
    @everywhere global_logger(ConsoleLogger(Logging.Info,show_limited=false))
end

@info "Parameters" n_simulation l project muT1 muTphi sigmaT1 sigmaTphi t n_estimate beta n_step SamplePerSimulation select regularize fullcovariance

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

Params = SimulationParameters_estimator(C = Code, f_neighborhops = f_neighborhops, LocalEstimator=LocalEstimator,ChannelSampler=ChannelSampler,Regularizer=Regularizer,n_simulations=n_simulation,n_estimate=n_estimate,project=project)

reset_timer!()
estimator_simulation("Test.hdf5", Params;printintervall=1)	
print_timer(allocations=false)

File = h5open("Test.hdf5","r")
println("L1 errors")
for n_est in n_estimate
    Est = File["n_est=$n_est"]["EstimatedErrorRates"][]
    Actual = File["n_est=$n_est"]["ActualErrorRates"][]
    L1 = L1Error(Est[:,5],Actual[:,5])
    println("nest: $n_est L1: $L1")
end
close(File)

