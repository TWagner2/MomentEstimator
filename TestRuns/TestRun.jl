using TimerOutputs
include("../Main.jl")
n_simulation = 1
l = 3
project = true
muT1 = 100e-6
muTphi = 200e-6
sigmaT1 = 25e-6
sigmaTphi = 50e-6
t = 15e-6
beta=0.5
n_step=1
SamplePerSimulation=true
regularize=true
fullcovariance=false
Debug=false
n_estimate = [100]

reset_timer!()

if Debug
    @everywhere global_logger(ConsoleLogger(Logging.Debug,show_limited=false))
end

@info "Parameters" n_simulation l project muT1 muTphi sigmaT1 sigmaTphi t n_estimate beta n_step SamplePerSimulation regularize fullcovariance

n_qubit = size(h_surfacecode_regular(l),1)

LocalEstimator = Estimator_lsq_optim(β=beta, n_step=n_step)
ChannelSampler = ChannelSampler_TVAPDTwirled(t=t,μ_T1 = muT1, μ_Tphi = muTphi, σ_T1 = sigmaT1, σ_Tphi = sigmaTphi, SamplePerSimulation = SamplePerSimulation)
if regularize
    #Estimate the mean and variance of the moments for given channel params and use it to regularize, in principle this could also be computed analytically by averaging the TVAPD(T1,T2) channel over T1,T2
    SampleP = sample_TVAPD_twirled(t,muT1, sigmaT1, muTphi,sigmaTphi, 10^5)
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


Params = SimulationParameters_estimator(LocalEstimator=LocalEstimator,ChannelSampler=ChannelSampler,Regularizer=Regularizer,n_simulations=n_simulation,n_estimate=n_estimate,l=l,project=project, Codetype = "Surface code square regular")

estimator_surfacecode_simulation("Test.hdf5", Params;printintervall=1)	

print_timer(allocations=false)