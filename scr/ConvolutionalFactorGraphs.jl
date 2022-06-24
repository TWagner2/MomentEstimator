"""
Structs and functions to represent Factor Graphs over binary Variables.
Currently only used for noise models, but in principle estimator could / should be based on this for correlated / multi-qubit noise.
"""

"""
Factor in a Factor graph over binary variables.
"""
abstract type AbstractFactor end
struct Factor <: AbstractFactor
    Support::Vector{Int} #This should be sorted
    Values::Union{BitlabeledVector{Float64}, Nothing}
    function Factor(Support::Vector{Int}, Values::Union{BitlabeledVector, Nothing})
        sort!(Support)
        return new(Support,Values)
    end
    function Factor(Support::Vector{Int}, Values::Union{Vector, Nothing})
        sort!(Support)
        if Values !== nothing
            Values = BitlabeledVector(Values, Int(log2(length(Values))))
        end
        return new(Support,Values)
    end
end
function Base.getindex(F::Factor,v::BitVector)
    v_sup = v[F.Support]
    return F.Values[v_sup]
end
function Base.getindex(F::Factor,v::Vector{Int})
    v_sup = intersect(v,F.Support)
    v_sup = Int.(indexin(v_sup, F.Support)) #Convert to "local coordinates"
    return F.Values[v_sup]
end
"""
Sparse representation of a Canonical Factor of a product factor graph. There is only 1 non-trivial value.
"""
struct CanonicalFactor <: AbstractFactor
    Support::Vector{Int} #This should be sorted
    Value::Float64
    function CanonicalFactor(Support::Vector{Int}, Value::Float64)
        sort!(Support)
        return new(Support,Value)
    end
end
function Base.getindex(F::CanonicalFactor,v::BitVector)
    v_sup = v[F.Support]
    if all(v_sup) 
        return F.Value
    else
        return 1
    end
end
function Base.getindex(F::CanonicalFactor,v::Vector{Int})
    if issubset(F.Support,v)
        return F.Value
    else
        return 1
    end
end

"""
struct FactorGraph

A factor graph over n binary variables.
"""
struct FactorGraph
    n::Int #Number of binary variables
    Factors::Vector{AbstractFactor}
    function FactorGraph(n::Int, Factors::Vector)
        return new(n,Factors)
    end
    function FactorGraph(n::Int,Supports::Vector{Vector{Int}},Values::Union{Vector{BitlabeledVector{Float64}}, Vector{Vector{Float64}}})
        Factors = [Factor(Supports[i],Values[i]) for i in 1:length(Supports)]
        return new(n,Factors)
    end
end
"""
evaluate_prodFG(FG::FactorGraph, v::BitVector)

Evaluates the function (represented by the factor graph FG if FG is viewed as a product Factor Graph) at v.
v should have length FG.n
"""
function evaluate_prodFG(FG::FactorGraph, v::Union{BitVector, Vector{Int}})
    r = 1
    for F in FG.Factors
        r *= F[v]
    end
    return r
end
"""
ft_FG(FG::FactorGraph)

Fourier transform a convolutional factor graph to a product factor graph (probabilites -> moments)
"""
function ft_FG(FG::FactorGraph)
    Factors = []
    for F in FG.Factors
        v = ifwht_natural(getdata(F.Values))     
        push!(Factors,Factor(F.Support,v))
    end
    return FactorGraph(FG.n,Factors)
end
"""
ift_FG(FG::FactorGraph)

Fourier transform a product factor graph to a convolutional factor graph (moments -> probabilites)
"""
function ift_FG(FG::FactorGraph)
    Factors = []
    for F in FG.Factors
        v = fwht_natural(getdata(F.Values))     
        push!(Factors,Factor(F.Support,v))
    end
    return FactorGraph(FG.n,Factors)
end

"""
canonical_factorization(FG::FactorGraph)

Compute the canonical factorization of a (product) factor graph, see e.g. https://arxiv.org/abs/1207.1366 .
"""
function canonical_factorization(FG::FactorGraph)
    #First, we find all supports that are necessary
    Values = Dict(Int[])
    for F in FG.Factors #Would be more efficient to only iterate over maximal factors
        a = powerset(F.Support) #This is ordered by size
        for Supp in a
            if !(Supp in keys(Values))
                E = evaluate_prodFG(FG,Supp)
                b = powerset(Supp)
                deleteat!(b,findall(x -> x == Supp, b)) #Remove Supp itself
                for s in b
                    E /= Values[s]
                end
                Values[Supp] = E
            end
        end
    end
    Factors = [CanonicalFactor(Supp,Values[Supp]) for Supp in keys(Values)]
    return FactorGraph(FG.n,Factors)
end


#Utility
function fg_singlequbit(n,ErrorRates)
    Factors = [Factor([i, i+n],ErrorRates[:,i]) for i in 1:n]
    return FactorGraph(2*n,Factors)
end
"""
l1(FG1::FactorGraph, FG2::FactorGraph)

L1 distance between the functions represented by the two product factor graphs.
Note that this is by bruteforce and only works for small examples.
"""
function l1(FG1::FactorGraph, FG2::FactorGraph; printintervall = 10^5)
    @assert FG1.n == FG2.n
    n = FG1.n
    r = 0
    for i in 1:2^n
        if i % printintervall == 0
            @info "Step $i"
        end
        v = indextobitvector(i,n)
        r += abs(evaluate_prodFG(FG1,v) - evaluate_prodFG(FG2,v))
    end
    return r / 2^n
end

function l1_conv(FG1::FactorGraph, FG2::FactorGraph; printintervall = 10^5, ft = true)
    @assert FG1.n == FG2.n
    n = FG1.n
    #First, Fourier transform and compute moments
    if ft
        FG1 = ft_FG(FG1)
        FG2 = ft_FG(FG2)
    end
    mom1 = zeros(2^n)
    mom2 = zeros(2^n)
    for i in 1:2^n
        if i % printintervall == 0
            @info "Step $i"
        end
        v = indextobitvector(i,n)
        mom1[i] = evaluate_prodFG(FG1,v)
        mom2[i] = evaluate_prodFG(FG2,v)
    end
    P1 = fwht_natural(mom1)
    P2 = fwht_natural(mom2)
    return L1Error(P1,P2)
end