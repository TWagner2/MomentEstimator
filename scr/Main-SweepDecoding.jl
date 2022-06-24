"""
Methods for the sweep decoder https://github.com/chubbc/SweepContractor.jl
Does not support measurement noise (on 2D codes), since then the lattice will be 3D.
"""

"""
construct_tensornetwork(QECC, ErrorRates; Error = nothing)

Construct a tensor network representation of the given QECC for use in decoding. 
The resulting tensor network describes the probability of the logical class of which Error is a member.
Does not currently support measurement errors.

This assumes that there are no tensors with the same connectivity. Could rewrite such that stabilizers with same / subset connectivity are combined into one tensor.
"""
function construct_tensornetwork( C::QECCGraph, ErrorRates = nothing; Error = nothing)
    H = C.C.H
    CheckEmbedding = C.CheckEmbedding
    QubitEmbedding = C.QubitEmbedding
    #Only float positions are accepted by the Tensors class
    CheckEmbedding = Float64.(CheckEmbedding)
    QubitEmbedding = Float64.(QubitEmbedding)
    
    LocalDim = 2 #Either you apply the stabilizer or you dont
    NCheck = size(H,2)
    NQubit = size(H,1)
    NTensor = NCheck + NQubit
    if isnothing(Error)
        Error = zeros(Int,NQubit)
    end

    TN = TensorNetwork(undef, NTensor) #The first NCheck entries are CheckTensors, the other NQubit entries are QubitTensors
    for i=1:NCheck
        Neighbors = nonzeroindices(H[:,i])
        n = length(Neighbors)
        #Entries is a delta tensor
        Entries = zeros(Float64,ones(Int,n).*LocalDim...)
        Entries[CartesianIndex(ones(Int,n)...)] = 1
        Entries[CartesianIndex(ones(Int,n).*2...)] = 1
        Adjacency = Neighbors .+ NCheck #Offset because QubitTensors are numbered after CheckTensors
        Position = CheckEmbedding[:,i]
        TN[i] = Tensor(Adjacency,Entries,Position...)
    end
    for i=1:NQubit
        Neighbors = nonzeroindices(H[i,:])
        n = length(Neighbors)
        Entries = zeros(Float64,ones(Int,n).*LocalDim...)
        if ErrorRates !== nothing
            for j in cartesianproduct(0:1,n)
                E = Error[i]
                for k in 1:n 
                    if j[k] == 1
                        e = H[i,Neighbors[k]]
                        E = addgf4(E,e)
                    end
                end
                Entries[CartesianIndex(j.+1)] = ErrorRates[gf4toindex(E,exchange=true),i]
            end
        end
        Adjacency = Int.(Neighbors)
        Position = QubitEmbedding[:,i]
        TN[i+NCheck] = Tensor(Adjacency,Entries,Position...)
    end
    return TN
end
function tn_setentries!(H::PauliMatrix, TN::TensorNetwork, Error::PauliVector, ErrorRates)
    n_qubit,n_check = size(H)
    for i in 1:n_qubit #iterate over qubit tensors
        index = i + n_check
        Neighbors = TN[index].adj
        n = length(Neighbors)
        for j in cartesianproduct(0:1,n)
            E = Error[i]
            for k in 1:n 
                if j[k] == 1
                    e = H[i,Neighbors[k]]
                    E = addgf4(E,e)
                end
            end
            TN[index].arr[CartesianIndex(j.+1)] = ErrorRates[gf4toindex(E,exchange=true),i]
        end
    end
end

"""
decode_trivial_symplectic(H::SymplecticMatrix, S::AbstractVector{Bool})

 Return some vector E such that H*E = S % 2.
 
 Expects input in symplectic convention
 If pinv was pre-computed, pass it as H and set pre-inverted = true
"""
function decode_trivial_symplectic(H::SymplecticMatrix , S::AbstractVector{Bool}; preinverted = false)
    if !preinverted
        H_i = pinv_z2(H) 
    else
        H_i = H
    end
    E = H_i'*S.% 2 #This already gives E with Z part first because we did not introduce exchange of XZ part in symplectic prod
    return E
end
"""
decode_trivial(H, S)

Return an error E with syndrome S for parity check matrix H
"""
function decode_trivial(H::PauliMatrix,S::AbstractVector{Bool})
    H_s = gf4tosymplectic_cat(H)
    E = decode_trivial_symplectic(H_s,S)
    E = symplectictogf4(E;exchange=true)
    return E
end
function decode_trivial(H_i::SymplecticMatrix,S::AbstractVector{Bool})
    E = decode_trivial_symplectic(H_i,S;preinverted = true)
    E = symplectictogf4(E;exchange=true)
    return E
end

"""
decode_tn(C::QECCGraph, S, ErrorRates, χ, τ)

Determine the most likely logical error for the given syndrome S.

Uses a tensor network contraction algorithm that truncates bond dimensions to χ whenever they exceede τ
"""
function decode_tn_timed( C::QECCGraph, S, ErrorRates, TN::TensorNetwork, χ, τ;H_i::Union{SymplecticMatrix,Nothing} = nothing)
    t_decode_trivial = @elapsed begin
    if H_i === nothing
        H = gf4tosymplectic_cat(C.C.H)
        H_i = pinv_z2(H)
    end
    E = decode_trivial(H_i, S)
    end
    P_L = []
    #Timing
    t_construct_TN= 0
    t_decode_TN = 0
    for i=1:size(C.L,2)
        E_L = addgf4.(E,C.L[:,i])
        t_construct_TN += @elapsed begin
        tn_setentries!(C.C.H,TN,E_L,ErrorRates)
        end
        t_decode_TN += @elapsed begin
        P_CLASS = sweep_contract(TN, χ , τ; fast = true)
        end
        push!(P_L,P_CLASS)
    end
    a = argmax_ldexp(P_L)
    Correction = addgf4.(C.L[:,a], E)
    return Correction, t_decode_trivial, t_construct_TN, t_decode_TN
end
function decode_tn(C::QECCGraph, S, ErrorRates,  TN::TensorNetwork, χ, τ; H_i::Union{SymplecticMatrix,Nothing} = nothing)
    if H_i === nothing
        H = gf4tosymplectic_cat(C.C.H)
        H_i = pinv_z2(H)
    end
    E = decode_trivial(H_i,S)
    P_L = []
    for i=1:size(C.L,2)
        E_L = addgf4.(E,C.L[:,i])
        tn_setentries!(C.C.H, TN, E_L,ErrorRates)
        P_CLASS = sweep_contract(TN, χ , τ; fast = true)
        push!(P_L,P_CLASS)
    end
    a = argmax_ldexp(P_L)
    Correction = addgf4.(C.L[:,a], E)
    return Correction
end

#decoder simulation

"""
islogicalerror(C::QECCGraph, E)

Check if E is a logical error for the code.

Expects E to have trivial syndrome.
"""
function islogicalerror(C::QECCGraph, E::PauliVector)
    S = scalar_commutator_v(C.L, E)
    if all(S .== 0)
        return false #No logical error
    else
        return true #A logical error
    end
end
function islogicalerror(L::SymplecticMatrix, E::SymplecticVector)
    S = symplectic_prod(L, E)
    if all(S .== 0)
        return false #No logical error
    else
        return true #A logical error
    end
end

"""
logicalerrorrate(C::QECCGraph, ErrorRates, n; decoder = decode_tn, decoder_params = (20,40), classical = false)

Determine the logical error rate of given decoder.

The logical error rate is determined by decoding the syndromes of n sampled errors and checking for a logical error.
"""
function logicalerrorrate(C::QECCGraph, ErrorRates, n; ErrorRatesDecoder = ErrorRates, decoder::Function = decode_tn, decoder_args = (20,40), classical = false, unnormalized = false, rng=Random.GLOBAL_RNG, decoder_kwargs...)
    failures = 0
    H = gf4tosymplectic_cat(C.C.H)
    t_decode = 0
    for i=1:n
        if (i % 1000 == 0)
            @info "Decoding: $i Time: $t_decode"
            t_decode = 0
        end
        t_decode += @elapsed begin
        Es = sample_error(ErrorRates;rng=rng)
        S = symplectic_prod(H,Es)
        Correction = decoder(C,S,ErrorRatesDecoder, decoder_args...; decoder_kwargs...)
        Correction = gf4tosymplectic_cat(Correction,exchange=true)
        TotalError = (Es .⊻ Correction)
        if !all(symplectic_prod(H,TotalError) .== 0)
            error("Decoding failure")
            println(E)
            println(S)
            prinln(Correction)
            println(TotalError)
        end
        if !classical
            TotalError = symplectictogf4(TotalError,exchange=true)
            failures += islogicalerror(C,TotalError)
        else
            failures += any(TotalError .!= 0) #For a classical code the total error must be 0
        end
        end#elapsed
    end
    if !unnormalized
        return failures / n
    else
        return failures
    end
end
function logicalerrorrate_sweepdecoder(C::QECCGraph, ErrorRates, n; ErrorRatesDecoder = ErrorRates, χ = 20, τ = 40, classical = false,rng=Random.GLOBAL_RNG)
    t_setup = @elapsed begin
    TN = construct_tensornetwork(C)
    H = gf4tosymplectic_cat(C.C.H)
    H_i = pinv_z2(H)
    end
    @info ("t_setup: $t_setup")
    t_decode = @elapsed begin
    p_L = logicalerrorrate(C,ErrorRates,n;ErrorRatesDecoder=ErrorRatesDecoder,decoder = decode_tn,decoder_args=(TN,χ,τ), classical = classical,rng=rng, H_i = H_i)
    end
    @info ("t_decode: $t_decode")
    return p_L
end
function logicalerrorrate_sweepdecoder_multiprocess(C::QECCGraph, ErrorRates, n; ErrorRatesDecoder = ErrorRates, χ = 20, τ = 40, classical = false,rng=Random.GLOBAL_RNG)
    n_proc=Distributed.nprocs()
    t_setup = @elapsed begin
    TN = construct_tensornetwork(C)
    H = gf4tosymplectic_cat(C.C.H)
    H_i =pinv_z2(H)
    Samplesizes = divide_evenly(n,n_proc)
    end#elapsed
    @info ("t_setup: $t_setup")
    t_decode = @elapsed begin
    failures= @distributed (+) for i = 1:n_proc
        logicalerrorrate(C,ErrorRates,Samplesizes[i];ErrorRatesDecoder=ErrorRatesDecoder,decoder = decode_tn,decoder_args=(TN,χ,τ), classical = classical, unnormalized = true,rng=rng, H_i = H_i)
    end
    end #elapsed
    @info("t_decode: $t_decode")
    return failures / n
end