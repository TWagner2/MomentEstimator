# Pauli channel estimation from syndromes

This is an implementation of the estimator described in the paper "Pauli channels can be estimated from syndrome measurements in quantum error correction" by T. Wagner, D. Bruß, H. Kampermann and M. Kliesch (https://doi.org/10.48550/arXiv.2107.14252). If you use this code for academic research, please cite this paper. A paper proving sample complexity guarantees and showing simulations (using this code) is currently being worked on.

## Basic Usage

You can activate the julia project in the Julia package manager from the main folder of the project.
```
pkg> activate .
pkg> instantiate
```
This should install all dependencies in a new environment.


Here is a basic example how to use this module.
```julia
include("Main.jl")
Code = qecc_repetitioncode(3)
Channel = Channel_SingleQubit(expand_errorrates(0.0,0), [0.9 0.8 0.7; 0.1 0.2 0.3])
Est = Estimator_lsq_optim(β=0.1)
Syndromes = sample_syndromes(10^4,Code,Channel)
Estimate = run_estimator(Syndromes,Code,Est)
display(Estimate[1])
```
Here, we first create a code, in this case the 3 qubit repetition code, which is described by its parity check matrix. 
Then, we create a physical error channel. Since we view the repetition code as a classical code we set the Pauli error rates to 0 and have only bit flip errors.
Then, we create an estimator object, in this case a least squares estimator. The $\beta$ parameter is a lower bound on the moments.
Finally, we sample a set of syndrome measurements and run the estimator on the syndromes to obtain an estimate of the error rates.

Internally, the estimator creates an equation system with one equation for each non-trivial stabilizer of the code, in this case 3 equations. 
It then solves this system for the moments describing the noise and returns the moments and the correspondign error rates, which are the Fourier transform of the moments.
However, this means that for larger codes exponentially many equations will be used, which quickly becomes intractable. 
One way around this is to consider for each qubit only a neighborhood of a certain size and estimate the error rates of this qubits only from this subcode.
```julia
include("Main.jl")
Code = qecc_surfacecode_regular(3) #A 3x3 surface code with a rough and a smooth boundary
Channel = Channel_SingleQubit(expand_errorrates(0.1,Code.n_d)) #The error rates on each qubit are [0.9,0.03,0.03,0.03]
EstLocal = Estimator_lsq_optim(β=0.1)
Est = Estimator_applyall_local_nhop(EstLocal,surfacecode_hops(Code.H), nothing) #An estimator which applies EstLocal to each local Region. The regions are n-hop neighborhoods of each qubit.
Estimate = run_estimator(10^4,Channel,Code,Est) #Sampling of syndromes and estimation
display(Estimate[1])
```

There is also basic support for codes with (phenomenological) measurement errors.
For this, the code has to be represented as a data-syndrome code acting on data and measurement errors, i.e. the parity check matrix has one part acting on data (Pauli) errors and one part acting on measurement (bit-flip) errors. 
Furthermore, the data-syndrome code must have distance at least 3.
The easiest way is to just repeat the stabilizer measurements of the underlying quantum code.
In this case, even the local equation systems can get quite large.
Therefore, a smarter selection of equations might be needed.
Currently, we only support random subselection of equations via the "select" option of estimators.
Here is an example:
```julia
include("Main.jl")
Code = qecc_surfacecode_regular(3) #A 3x3 surface code with a rough and a smooth boundary
Code = qecc_repeat_measurements(Code,2) #Each measurement is done 2 times
Channel = Channel_SingleQubit(expand_errorrates(0.01,Code.n_d), expand_errorrates([0.98,0.02],24)) #The error rates on each qubit are [0.99,0.003,0.003,0.003], and each measurement hast a 0.01 chance to give the wrong outcome
EstLocal = Estimator_lsq_optim(β=0.1,select=100) #Use 100 randomly selected equations in each local region
Est = Estimator_applyall_local_nhop(EstLocal,surfacecode_hops_repeated(Code.H), nothing) #An estimator which applies EstLocal to each local Region. The regions are n-hop neighborhoods of each qubit.
Syndromes = sample_syndromes(10^4,Code,Channel)
Estimate = run_estimator(Syndromes,Code,Est)
display(Estimate[1])
```

## Simulations and decoding

In the examples Folder, there are some more examples of Simulations, such as randomly sampling a channel for each qubit and then estimating.
To run an example from the main folder using the appropriate environment, use e.g.

```
julia -project=. Examples/TestRun-Estimate.jl
```

We also support decoding with the sweep decoder developed by Chubb in https://doi.org/10.48550/arXiv.2101.04125 , although this does not support measurement noise.
Using this, one can compare the logical error rates of the code when the decoder uses the actual or the estimated error rates.
