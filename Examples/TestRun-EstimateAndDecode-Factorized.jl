include("../Main.jl")
n_simulation = 1
l = 3
project = true
p1=0.01
p2=0.005
beta=0.001
n_step=1
n_estimate = [10^5]
SamplePerSimulation = true

n_test = 10^4
χ=20
τ=60

DecodeMarginals = true

Options = Optim.Options()

reset_timer!()
Debug=false
if Debug
    @everywhere global_logger(ConsoleLogger(Logging.Debug,show_limited=false))
else
    @everywhere global_logger(ConsoleLogger(Logging.Info,show_limited=false))
end

@info "Parameters" n_simulation l project n_estimate beta n_step SamplePerSimulation

LocalEstimator = Estimator_lsq_optim(β=beta, n_step=n_step, Options=Options)
ChannelSampler = ChannelSampler_Constant(channel_nnsurfacecode(l,p1,p2)) #A form of nearest neighbor noise includign 2 qubit errors
Code = qeccgraph_surfacecode_regular(l)
Regularizer = nothing

ParamsEst = SimulationParameters_estimator(C=Code,LocalEstimator=LocalEstimator,ChannelSampler=ChannelSampler,Regularizer=Regularizer,n_simulations=n_simulation,n_estimate=n_estimate,project=project, f_neighborhops = surfacecode_hops)
Params = SimulationParameters_EstimateAndDecode(EstimatorParams=ParamsEst,n_test=n_test,χ=χ,τ=τ,Decode_actual=false)

estimateanddecode_simulation("Test-Decode.hdf5", Params)	

print_timer(allocations=false)

File = h5open("Test-Decode.hdf5","r")
Chan = sample_channel(ChannelSampler,1)
e = sample_errors(10^6,Chan)
e = symplectictogf4(e,exchange=true)
n = size(qecc_surfacecode_regular(l).H,1)
ApproxRates = zeros(4,n)
for i in axes(e,1)
    for j in axes(e,2)
        ApproxRates[e[i,j]+1,i] += 1
    end
end
ApproxRates = ApproxRates / 10^6 #Approximate 1 qubit marginals of the actual noise

if DecodeMarginals
    @info "Decode Marginals"
    Params = SimulationParameters_Decode(C = Code,n_simulations=n_simulation,n_test=n_test,χ=χ,τ=τ,ChannelSampler=ChannelSampler,Decoder_Input_Rates=ApproxRates)
    decode_simulation("Test-Decode-Marginals.hdf5",Params)
end

for n_est in n_estimate
#Note that the estimate will not converge to the marginals. This is expected, since the marginals are not the best possible approximation as a convolution to the actual noise
println("Est")
display(File["n_est=$n_est"]["EstimatedErrorRates"][])
println("Actual Marginals")
display(ApproxRates)
println("Logical")
display(File["n_est=$n_est"]["LogicalRates"][])
    if DecodeMarginals
        println("Logical with Marginals")
        File2 = h5open("Test-Decode-Marginals.hdf5")
        display(File2["LogicalRates"][])
    end
end

close(File)

