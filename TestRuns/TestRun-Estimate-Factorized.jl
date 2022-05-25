include("../Main.jl")
using FreqTables
n_simulation = 1
l = 3
project = true
p1=0.01
p2=0.005
beta=0.0001
n_step=1
n_estimate = [10000]
SamplePerSimulation = true
Options = Optim.Options()

reset_timer!()
Debug=false
if Debug
    @everywhere global_logger(ConsoleLogger(Logging.Debug,show_limited=false))
else
    @everywhere global_logger(ConsoleLogger(Logging.Info,show_limited=false))
end

@info "Parameters" n_simulation l project n_estimate beta n_step SamplePerSimulation

n_qubit = size(h_surfacecode_regular(l),1)

LocalEstimator = Estimator_lsq_optim(β=beta, n_step=n_step, Options=Options)
#LocalEstimator = Estimator_pinv(β=beta)
ChannelSampler = ChannelSampler_Constant(channel_nnsurfacecode(l,p1,p2))
Regularizer = nothing

Params = SimulationParameters_estimator(LocalEstimator=LocalEstimator,ChannelSampler=ChannelSampler,Regularizer=Regularizer,n_simulations=n_simulation,n_estimate=n_estimate,l=l,project=project, Codetype = "Surface code square regular")

estimator_surfacecode_simulation("Test.hdf5", Params;printintervall=1)	

print_timer(allocations=false)

File = h5open("Test.hdf5","r")
Chan = sample_channel(ChannelSampler,1)
e = symplectictogf4(sample_errors(10^6,Chan);exchange=true)
n = size(h_surfacecode_regular(l),1)
ApproxRates = zeros(4,n)
for i in axes(e,1)
    for j in axes(e,2)
        ApproxRates[e[i,j]+1,i] += 1
    end
end
ApproxRates = ApproxRates / 10^6

for n_est in n_estimate
    println("Est")
    display(File["n_est=$n_est"]["EstimatedErrorRates"][])
    println("Actual")
    display(ApproxRates)
end

#Compare:
FG_mom_exact = ft_FG(Chan.FG)
FG_mom_marginals = ft_FG(fg_singlequbit(n,ApproxRates))
FG_mom_estimate = ft_FG(fg_singlequbit(n,File["n_est=$(n_estimate[1])"]["EstimatedErrorRates"][]))
close(File)

#Comparison in Fourier space: l1(FG_mom_exact, FG_mom_marginals/estimate)
# For p1 = 0.01, p2 = 0.005, n_est = 10000 i got l1(exact,marginal) ≈ 0.88, l1(exact, estimate) ≈ 0.18
#For the l1 distance of the probability distributions (different trial)  i got l1(exact,marginal) ≈ 0.18, l1(exact, estimate) ≈ 0.14