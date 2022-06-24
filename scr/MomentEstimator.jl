"""
grouptobinomial(H::PauliMatrix)

Construct the binomial system corresponding to Pauli matrix H.

Each Pauli is converted to a 4 element indicator vector indicating X,Z or Y and measurement errors.
Moments are ordered as first the X moment of each qubit, then Z moment of each qubit, then Y moment of each qubit. After that the moments for measurement errors.
"""
function grouptobinomial(G::PauliMatrix; n_d=size(G,1))
    G_q = G[1:n_d,:]
    G_m = G[n_d+1:end,:]
    G_x = (G_q .== 1)
    G_z = (G_q .== 2)
    G_y = (G_q .== 3)
    return vcat(G_x,G_z,G_y,G_m)
end

"""
expectations_from_syndromes(S::AbstractMatrix;expand=true)

Compute expectation values from a set of syndrome measurements.

If expand=true, the syndromes are first expanded with expand_syndromes.
"""
function expectations_from_syndromes(S::AbstractMatrix{Bool};expand=true, Chunksize = size(S,2))
    average = expand ? zeros(2^size(S,1)) : zeros(size(S,1))
    n = size(S,2)
    for i in Iterators.partition(axes(S,2),Chunksize)
        Sub = S[:,i]
        S_e = expand ? expand_syndromes(Sub) : Sub
        nz = count(S_e,dims=2)
        r = size(S_e,2) .- 2*nz # This is the same as first converting the syndromes to the 1,-1 convention and then taking the average
        average += r
    end
    average = average ./ n
    return dropdims(average,dims=2)
end
"""
binomexp(E,M)

Compute E^M in the sense of Vector^Matrix of binomial systems.

We have binomexp(E,M)[i] = prod_j E[j]^M[j,i]
"""
function binomexp(E::AbstractVector,M::AbstractMatrix)
    R = ones(eltype(E),size(M,2))
    for i in 1:size(M,2)
        for j in 1:size(M,1)
            R[i] *= E[j]^M[j,i]
        end
    end
    return R
end
"""
function binomexp_jacobian(E,B)

The derivative of the function binomexp(. , B) at E
"""
function binomexp_jacobian(E,B)
    f = binomexp(E,B)
    J = f .* B' .* (1 ./ E)' #Same as diagm(f) * B' * diagm(1./E)
    return J
end
"""
momentsfromrates(P)

Moments E corresponding to the error rates P, corresponds to inverse Fourier transform on Pauli group.

#Warning: It seems this can crash if used in multithreading because of calls in Hadamard library
"""
function momentsfromrates(P)
    return ifwht_natural(P,1)
end
"""
ratesfrommoments(E)

Error Rates P corresponding to moments E, corresponds to Fourier transform on Pauli group.
If Omit = true, then it is assumed that the first entry of E was ommited and it is assumed to be 1.
Note that moments should be ordered according to stabilizer convention II, XI, IX, XX, ZI, ZX,... = (0000, 1000, 0100, 1100, 0010,...) while Probabilities are ordered according to Error convention, i.e. with X and Z exchanged

#Warning: It seems this can crash if used in multithreading because of calls in Hadamard library
"""
function ratesfrommoments(E;Omit=true)
    if Omit
        I = ones(eltype(E),(1,size(E)[2:end]...))
        E = vcat(I,E)
    end
    return fwht_natural(E,1)
end

#projections

"""
project_moments_simple!(E; β=0.001)

Perform a simple cutoff to map moments larger than 1 to 1 and moments smaller than β to β
"""
function project_moments_simple!(E; β=0.001)
    for i in eachindex(E)
        if E[i] < β
            #@info "Projected: $(E[i])"
            E[i] = β
        elseif E[i] > 1
            E[i] = 1
        end
    end
end

"""
project_probabilitysimplex(v)

Project each column of v onto the probability simplex v_i > 0, sum_i(v_i) = 1

Algorithm from https://arxiv.org/pdf/1309.1541.pdf
"""
function project_probabilitysimplex!(v::AbstractArray{T}) where {T<:AbstractFloat}
    n = size(v)[1]
    u = sort(v;dims=1,rev = true)
    ucum = 1 .- cumsum(u;dims=1)
    for i in CartesianIndices(ucum)
        ucum[i] = ucum[i] / i[1]
    end
    utmp = u + ucum
    rho = zeros(Int,size(v)[2:end])
    for j in CartesianIndices(rho)
        rho[j] = findlast(utmp[:,j] .> 0)
        for i in 1:n
            v[i,j] = max(v[i,j] + ucum[rho[j],j],0)
        end
    end
end
function project_probabilitysimplex!(v::AbstractVector{T}) where {T<:AbstractFloat}
    v = reshape(v,(size(v)...,1))
    project_probabilitysimplex!(v)
    return dropdims(v,dims=2)
end
"""
function project_momentsrates(E, Omit=true)

Project the moments E onto the fourier transform of the probability simplex, i.e. onto the set of valid moments.

#Arguments:
-E: A matrix. Each row is projected separately.
"""
function project_momentsrates(E::AbstractMatrix, Omit=true)
    R = ratesfrommoments(E;Omit=Omit)
    project_probabilitysimplex!(R)
    E = momentsfromrates(R)
    return E,R
end
"""
function project_momentsrates(E, Omit=true)

Project the moments E onto the fourier transform of the probability simplex, i.e. onto the set of valid moments.

#Arguments:
-E: A vector in which the non-trivial moments for all qubits / measurements are stacked.
"""
function project_moments_cat(E::AbstractVector, n_d=size(E,1) ÷ 3)
    E_d, E_m = momentsbyqubits(E, n_d = n_d)
    E_d,_ = project_momentsrates(E_d) #otherwise the estimated covariance is not guaranteed to be positive
    E_m,_ = project_momentsrates(E_m)
    E_d = E_d[2:end,:]
    E_m = E_m[2:end,:]
    E_d = Vector(vec(E_d'))
    E_m = Vector(vec(E_m'))
    E = vcat(E_d,E_m)
    return E
end

#pinv estimation
"""
computemoments_pinv_snf( H::PauliMatrix, Es)

Solve the binomial system arising from H and the stabilizer expectations Es via Smith normal form.
"""
function solvemoments_pinv_snf( C::QECC, Es, Params::Estimator_pinv_snf)
    @unpack_Estimator_pinv_snf Params
     G= construct_group(C.H)
     G,Es = select_random(G,Es,select)
     B = Int.(grouptobinomial(G,n_d = C.n_d))
     E = solvemoments_pinv_snf_binom(B,Es;β=β)
     return E
end
#For some reason the solution for transposed matrix is better at least for toric code
function solvemoments_pinv_snf_binom(B,Es;β=0.001)
    Bt = permutedims(B)
    Bi = permutedims(pinv_int(Bt))
    n = size(B,1) ÷ 3
    γ = β^n #A lower bound on the stabilize rexpectations since they must be products of individual moments which are lower  bounded by beta
    project_moments_simple!(Es; β = γ) #To avoid zeros, or if a lower bound on moments is known. A better lower bound would be β^(wt(S)) for each stabilizer S
    E = binomexp(Es,Bi)
    return E
end
"""
estimatemoments_pinv( H::PauliMatrix, S)

Estimate moments from set of stabilizer measurements using normal pinv.
"""
function estimatemoments_pinv_snf( C::QECC, S; Params::Estimator_pinv_snf)
    @unpack_Estimator_pinv_snf Params
     Es = expectations_from_syndromes(S,Chunksize=Chunksize)
    return solvemoments_pinv_snf(C, Es,Params)
end

"""
solvemoments_pinv( H::PauliMatrix, Es)

Solve the binomial system arising from H and the stabilizer expectations Es via normal pinv.
"""
function solvemoments_pinv( C::QECC, Es, Params::Estimator_pinv)
    @unpack_Estimator_pinv Params
     @timeit "Construct group" begin G= construct_group(C.H) end
     G, Es = select_random(G,Es,select)
     @timeit "group to binomial" begin B = Int.(grouptobinomial(G,n_d = C.n_d)) end
     E = solvemoments_pinv_binom(B,Es; β = β)
     return E
end
function solvemoments_pinv_binom(B,Es; β = 0.001)
    @timeit "pinv" begin Bi = pinv(B) end
    n = size(B,1) ÷ 3
    γ = β^n #A lower bound on the stabilize rexpectations since they must be products of individual moments which are lower  bounded by beta
    project_moments_simple!(Es; β = γ) #To avoid zeros, or if a lower bound on moments is known
    E = binomexp(Es,Bi)
    return E
end
"""
estimatemoments_pinv( H::PauliMatrix, S)

Estimate moments from set of stabilizer measurements using normal pinv.
"""
function estimatemoments_pinv( C::QECC, S; Params::Estimator_pinv)
    @unpack_Estimator_pinv Params
    Es = expectations_from_syndromes(S,Chunksize = Chunksize)
    return solvemoments_pinv(C,Es, Params)
end

"""
Uses optim package instead of levenberg-marquardt, can also handle regularization
"""
function cost_optim(E, Es, B; Regularizer::Union{AbstractRegularizer,Nothing} = nothing, Wt=I, nsamples)
    @timeit "f" begin
    A = binomexp(E,B) - Es
    tmp = 0.5*sum(A'*Wt*A)
    if Regularizer !== nothing
        tmp = tmp + regularizer_cost(E,Regularizer, nsamples)
    end
    end
    return tmp
end
function cost_optim_g(E,Es,B; Regularizer::Union{AbstractRegularizer,Nothing} = nothing, Wt=I, nsamples)
    @timeit "g" begin
    f = binomexp(E,B) #replace by inplace
    J = binomexp_jacobian(E, B) #replace by inplace
    G = dropdims((f - Es)'*Wt*J;dims=1)
    if Regularizer !== nothing
        R = dropdims(regularizer_jacobian(E,Regularizer, nsamples); dims=1)
        G = G + R
    end
    end
    return G
end
function cost_optim_h(E,Es,B; Regularizer::Union{AbstractRegularizer,Nothing} = nothing, Wt=I, nsamples)
    @timeit "h" begin
    @timeit "h_binom" begin
    f = binomexp(E,B) #replace by inplace
    J = binomexp_jacobian(E, B) #replace by inplace
    diff = f-Es
    H = J'*Wt*J
    for l in axes(J,1)
        @views tmp1 = B[:,l] * f[l] ./ E
        @views tmp2 = tmp1 * (B[:,l] ./ E)'
        @views tmp3 = tmp2 - (diagm(tmp1) ./ E)
        H += tmp3 * dot(diff,Wt[:,l])
    end
    end
    @timeit "h_regularizer" begin
    if Regularizer !== nothing
        H_reg = regularizer_hessian(E,Regularizer, nsamples)
        H += H_reg
    end
    end
    end
    return H
end
function squareexpectation_syndromes!(V::AbstractMatrix,Es::AbstractVector)
    n = length(Es)
    for c in CartesianIndices(V)
        #V[i,j] should be expectation(S_iS_j), which is the moment corresponding to i + j as bitstrings
        ind1,ind2 = combinationofindex(c[1],n), combinationofindex(c[2],n)
        ind = ind1 .⊻ ind2
        ind = findall(ind)
        ind = indexofcombination(ind)
        V[c] = Es[ind]
    end
end
function covariance_syndromes!(V::AbstractMatrix,Es::AbstractVector)
    squareexpectation_syndromes!(V,Es)
    Eprod = Es*Es'
    for i in eachindex(V)
        V[i] = V[i] - Eprod[i]
    end
end
"""
The covariance of the residual of the cost function, i.e. Expectation( (Es - Eshat) (Es-Eshat)' )
"""
function covariance_residual(Es_sample::AbstractVector, Es_compute::AbstractVector)
    V=zeros((length(Es_sample), length(Es_sample)))
    squareexpectation_syndromes!(V,Es_sample)
    V += Es_compute*Es_compute'
    V -= (Es_sample*Es_compute' + Es_compute*Es_sample')
    return V
end
"""
solvemoments_lsq_optim(C::QECC, Es, n_samples, Params::Estimator_lsq_optim; initializer::Function = solvemoments_pinv_binom)

Solve the binomial system arising from C.H and measured moments Es using least-squares.
A regularizer adding a square term to the cost function can be used, which corresponds to a Gaussian prior on E. In this case n_samples is used to correctly weight the regularization.
If n_step > 1 is used, an n-step estimator is used that iteratively updates an estimate of the covariance matrix and then uses it as weights (as is usual in "generalized method of moments"). (Compare e.g. https://projecteuclid.org/journals/bayesian-analysis/volume-4/issue-2/Bayesian-generalized-method-of-moments/10.1214/09-BA407.full)
"""
function solvemoments_lsq_optim(C::QECC, Es, n_samples, Params::Estimator_lsq_optim; initializer::Function = solvemoments_pinv_binom)
    @unpack_Estimator_lsq_optim Params
    if select !== nothing && n_step > 1
        throw("Random subselection of equations is currently not compatible with reweighting. Needs to be implemented.")
    end
    G = construct_group(C.H)
    G,Es = select_random(G,Es,select)
    B = Int.(grouptobinomial(G, n_d = C.n_d))
    Estimate = initializer(B, Es; β = β)
    project_moments_simple!(Estimate,β = β)
    buffer = 0.0001 #Init is not allowed to be on the boundary
    lower = zeros(size(Estimate)) .+ (1-buffer)*β  
    upper = ones(size(Estimate)) .* (1+buffer)  # Lower and upper bounds
    inner_optimizer = IPNewton()
    if typeof(Wt) <: UniformScaling
        Wt = Matrix{Float64}(Wt,length(Es),length(Es)) #Expand to a matrix because we want to update it inplace later, and also access entries
    end
    for i=1:n_step
        @debug "Step=$i"
        @timeit "t_step" begin
        f(x) = cost_optim(x,Es,B; Regularizer=Regularizer, Wt=Wt, nsamples=n_samples)
        g(x) = cost_optim_g(x, Es, B; Regularizer=Regularizer, Wt=Wt, nsamples = n_samples)
        h(x) = cost_optim_h(x, Es, B; Regularizer=Regularizer, Wt=Wt, nsamples = n_samples)
        @debug " " InitCost = f(Estimate)
        df = TwiceDifferentiable(f,g,h,Estimate; inplace = false)
        constraints = TwiceDifferentiableConstraints(lower,upper)
        @timeit "Optimization" begin FitResult = optimize(df,constraints,Estimate,inner_optimizer, Options) end
        if !Optim.converged(FitResult)
            @warn "Fit did not converge"
        end
        Estimate = Optim.minimizer(FitResult)
        @debug " " FinalCost=(Optim.minimum(FitResult))
        @timeit "Proc" begin
        if  i < n_step
            @timeit "project" begin
            Estimate = project_moments_cat(Estimate, C.n_d)
            Es_est = binomexp(Estimate,B)
            end
            @timeit "cov" begin covariance_syndromes!(Wt,Es_est) end #Not compatible with selection of subgroup of full stabilizer group currently
            @timeit "pinv" begin Wt = pinv(Wt) end  # pinv because we can have 0 eigenvalues
            project_moments_simple!(Estimate,β = β) #Guarantee that initialization is within bounds
        end
        end # proc
        end # step
    end
    return Estimate
end
function estimatemoments_lsq_optim(C::QECC, S; Params::Estimator_lsq_optim)
   @unpack_Estimator_lsq_optim Params
    Es = expectations_from_syndromes(S,Chunksize = Chunksize)
    n_samples = size(S,2)
    return solvemoments_lsq_optim(C, Es, n_samples, Params)
end

#local estimation
"""
estimate_local_nhop(H::PauliMatrix,estimator::Function,i::Integer,S;n=2)

Estimate for qubit i using only the subcode defined by its N-hop neighborhood.
N_hop should be even.
"""
function estimate_local_nhop(Q::QECC, S, i::Integer; LocalEstimator::AbstractEstimator, n_hop::Integer=2, Regularizer::Union{AbstractTiledRegularizer,Nothing} = nothing)
    R,C = adjacent_nhop(Q.H,i,true,n_hop)
    isub =indexin(i,R)[1] #the index of qubit i in the subcode
    @debug "" i isub
    Hsub = @view Q.H[R,C]
    Ssub = selectdim(S,1,C)
    n_d_local = count(R .<= Q.n_d) #How many rows of Hsub describe qubits vs measurement errors
    if Regularizer !== nothing
        RegLocal = select_local_regularizer(R,Regularizer)
        LocalEstimator = set_regularizer(LocalEstimator,RegLocal)
    end
    LocalCode = QECC("Local", Hsub, n_d_local)
    Estimate = estimate(LocalCode, Ssub, LocalEstimator)
    Estimate_d, Estimate_m = momentsbyqubits(Estimate,n_d = n_d_local)
    if isub <= n_d_local
        return Estimate_d[:,isub]
    else
        return Estimate_m[:,isub - n_d_local]
    end
end
"""
function applyall_local_nhop(H::PauliMatrix,estimator::Function,S, n_hop::AbstractVector{T}) where {T<:Integer}

Estimate the rates of all qubits by applying a local estimator or solver for each qubit separately.
"""
function applyall_local_nhop(C::QECC, S; Params::Estimator_applyall_local_nhop) where {T<:Integer}
    @unpack_Estimator_applyall_local_nhop Params
    n,m = size(C.H)
    Estimates_d = Array{Float64}(undef,(3,C.n_d))
    Estimates_m = Array{Float64}(undef,(1,n-C.n_d))
    Hops = Iterators.cycle(n_hop)
    for (i,Hop) in Iterators.zip(axes(C.H,1),Hops)
        Est = estimate_local_nhop( C, S, i; LocalEstimator = LocalEstimator ,n_hop=Hop, Regularizer=Regularizer)
        if i <= C.n_d
            Estimates_d[:,i] = Est
        else
            Estimates_m[:,i-C.n_d] = Est
        end
    end
    return (Estimates_d, Estimates_m)
end

#convenience

"""
momentsbyqubits(E)

Take vector of moments with order as defined by grouptobinomial, and returns a 2D array and a vector.
The 2D array has as column i the X,Z,Y moments of qubit i, and the vector has the moments for the measurement errors.

Just returns E itself if it is already a Matrix.
"""
function momentsbyqubits(E::AbstractVector;n_d = size(E,1) ÷ 3)
    E_q = E[1:n_d*3]
    E_m = E[n_d*3+1:end]
    M_q = Matrix(reshape(E_q,(:,3))') #order by qubit
    M_m = Matrix(reshape(E_m,(:,1))')
    return M_q,M_m

end
function momentsbyqubits(E; n_d = 1)
    return E
end