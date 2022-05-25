const PauliVector = AbstractVector{T} where {T<:Integer} #Can also contain measurement errors, in which case it is a n_data + n_measurement vector with the first part in 0-3 and the last part in 0-1
const PauliMatrix = AbstractMatrix{T} where {T<:Integer}
const PauliVecOrMat = Union{PauliVector, PauliMatrix}
const PauliArray = AbstractArray{T} where {T<:Integer}

const SparsePauliVector = SparseVector{T} where {T<:Integer}
const SparsePauliMatrix = SparseMatrixCSC{T} where {T<:Integer}
const SparsePauliVecOrMat = Union{SparsePauliVector, SparsePauliMatrix}

#Convention: Stabilizers have x part first, errors have z part

const SymplecticVector = AbstractVector{Bool}
const SymplecticMatrix = AbstractMatrix{Bool}
const SymplecticVecOrMat = Union{SymplecticVector, SymplecticMatrix}
const SymplecticArray = AbstractArray{Bool}

include("Utils-snf.jl")

"""
diagm(v)

Matrix with v on diagonal
"""
function diagm(v::AbstractVector)
    n = length(v)
    M = zeros(eltype(v), (n,n))
    for i=1:length(v)
        M[i,i] = v[i]
    end
    return M
end

"""
addgf4(x::Integer , y::Integer)

Addition in GF(4).

Can relate to Paulis via 0 = I, 1 = X, 2 = Z, 3 = Y
"""
function addgf4(x::Integer , y::Integer)
    if x < 0 || x > 3 || y < 0 || y > 3
        error("Invalid arguments")
    end
    return x ⊻ y 
end
"""
scalar_commutator(x::Integer, y::Integer)

Return 1 if x and y anti-commute as single Paulis and 0 otherwise.
"""
function scalar_commutator(x::Integer, y::Integer)
    if x < 0 || x > 3 || y < 0 || y > 3
        error("Invalid arguments")
    end
    if x == 0 || y == 0 || x == y
        return 0
    else
        return 1
    end
end
"""
symplectic_prod(x, y)

Return syndromes of measurements of cols of x on cols of y.

Note that this does not include exchange of x and z part, since we already have convention that measurements have X-part first and errors Z-part first (and measurement errors are always at the end)
"""
function symplectic_prod(x::SymplecticVecOrMat, y::SymplecticVecOrMat)
    r = (x'*y) .% 2
    r = Bool.(r)
    return r
end
function scalar_commutator_v(x::PauliVecOrMat, y::PauliVecOrMat; n_d = size(x,1))
    xb = gf4tosymplectic_cat(x,n_d=n_d)
    yb = gf4tosymplectic_cat(y,exchange=true,n_d=n_d)
    return symplectic_prod(xb,yb)
end
"""
gf4tox(x)

X-part of symplectic representation of Pauli x
"""
function gf4tox(x)
    if x < 0 || x > 3
        error("Not a valid Pauli")
    end
    return x == 1 || x == 3
end
"""
gf4toz(x)

Z-part of symplectic representation of Pauli x
"""
function gf4toz(x)
    if x < 0 || x > 3
        error("Not a valid Pauli")
    end
    return x == 2 || x == 3  
end
"""
gf4tosymplectic(x)

Convert a Pauli-string into symplectic representation.

If you want concatenated form, use gf4tosymplectic_cat
"""
function gf4tosymplectic(x::PauliArray;n_d=size(x,1))
    x_qubit = selectdim(x,1,1:n_d)
    x_measurement = BitArray(selectdim(x,1,n_d+1:size(x,1)))
    return (gf4tox.(x_qubit),gf4toz.(x_qubit),x_measurement)    
end
"""
gf4tosymplectic_cat(x)

Convert a Pauli-string into symplectic representation.

Exchange = true should be used for errors, false for stabilizers
"""
function gf4tosymplectic_cat(v;exchange=false,n_d=size(v,1))
    x,z,m = gf4tosymplectic(v,n_d=n_d)
    if !exchange
        return vcat(x,z,m)
    else
        return vcat(z,x,m)
    end
end
"""
xzpart(x)

Extract X, Z and measurement Part from concatenated symplectic vector
"""
function xzpart(x;n_d=size(x,1) ÷ 2)
    return (selectdim(x,1,1:n_d),selectdim(x,1,n_d+1:2*n_d), selectdim(x,1,2*n_d+1:size(x,1)))
end
function swapxzpart(v)
    x,z,m = xzpart(v)
    return vcat(z,x,m)
end
"""
symplectictogf4(x,z)

Convert symplectic representation to GF(4) representation.
"""
function symplectictogf4(x,z,m)
    return vcat(x + 2*z,m)
end
"""
symplectictogf4(x)

Convert concatenated symplectic representation to GF(4) representation.
"""
function symplectictogf4(v; exchange = false, n_d = size(v,1) ÷ 2)
    x,z,m = xzpart(v,n_d=n_d)
    if !exchange
        return symplectictogf4(x,z,m)
    else
        return symplectictogf4(z,x,m)
    end
end

function randPauli(i)
    return rand([1,2,3],i)
end
function randSparsePauli(n,m,d)
    return sprand(n,m,d,randPauli)
end

"""
function expand_errorrates(p)

Interpret p as an error rates vector on n qubits.

If p is scalar, the rates are 1-p,p/3 eevrywhere. If p is a vector, it is repeated on all qubits. If p is a matrix, it is not changed.
"""
function expand_errorrates(p::AbstractFloat,n::Integer)
    ErrorRates = repeat([1-p, p/3, p/3, p/3],outer=(1,n))
    return ErrorRates
end
function expand_errorrates(p::AbstractVector{T}, n::Integer) where {T<:AbstractFloat}
    ErrorRates = repeat(p,outer=(1,n))
    return ErrorRates    
end 
function expand_errorrates(p::AbstractMatrix{T}, n::Integer) where {T<:AbstractFloat}
    if size(p,1) != 4 || size(p,2) != n
        error("Invalid error rates")
    end
    return p
end

"""
function embed_measurmenterrorrates(R::AbstractVector)

Embed  measurement error rates such that they can be written into one array with Pauli error rates.

Arguments:
-R: A 2xn matrix of measurement error rates
"""
function embed_measurementerrorrates(R::AbstractMatrix)
return vcat(R,zeros(2,size(R,2)))
end
"""
function embed_measurmenterrorrates(R::AbstractVector)

Embed  measurement moments such that they can be written into one array with Pauli moments, and in a way that is compatible under Fourier traffo with the embedding of rates.

Arguments:
-E: A 2xn matrix of measurement error moments
"""
function embed_measurementmoments(E::AbstractMatrix)
return repeat(E, outer=(2,1))
end

"""
nonzeroindices( A)

Find indices of all nonzero elements
"""
function nonzeroindices( A)
    Indices = Int[]
    for i in eachindex(A)
        if A[i] != zero(A[i])
            push!(Indices,i)
        end
    end
    return Indices
end
function nonzeroindices( A::SparseVector)
    return A.nzind    
end
"""
cartesianproduct(A,n)

n times cartesian product of iterator A with itself
"""
function cartesianproduct(A,n)
    #Following https://stackoverflow.com/questions/56120583/n-dimensional-cartesian-product-of-a-set-in-julia
    return Iterators.product(ntuple(i->A, n)...)
end

#used to compare tuples of the form (f,i),  representing the value f*2^i (0 <= f < 1), as returned by sweep contractor
function issmaller_ldexp(a,b)
    if a[1] == 0
        return b[1] != 0
    end
    if a[2] != b[2]
        return a[2] < b[2]
    else
        return a[1]<b[1]
    end
end
function argmax_ldexp(A)
    r = 1
    for i in eachindex(A)
        if issmaller_ldexp(A[r],A[i])
            r = i
        end
    end
    return r
end

"""
construct_group(H::PauliMatrix)

Construct the stabilizer group corresponding to Pauli matrix H, i.e. the group generated by the columns of H.
If the columns of H are not independent duplicated elements will appear.
"""
function construct_group(H::PauliMatrix)
    n,m = size(H)
    G = zeros(Int, (n,2^m))
    index = 1
    for i in cartesianproduct([false,true],m)
        G[:,index] = reduce(addgf4,H[:,[i...]];dims=2,init=0)
        index += 1
    end
    return G
end
"""
select_random(G, Es, select)

Select "select" many random rows from G and elements from Es simultaneously.
"""
function select_random(G::PauliMatrix,Es::AbstractVector,select::Union{Int,Nothing})
    if select !== nothing && size(G,2) > select
        s = sample(1:size(G,2),select,replace=false)
        G = G[:,s]
        Es = Es[s]
    end
    return G,Es
end

"""
expand_syndromes(S)

Construct the expanded syndrome vectors corresponding to the stabilizer group.
"""
function expand_syndromes(S::AbstractVecOrMat{Bool})
    n = size(S,1)
    G = falses(2^n,size(S)[2:end]...)
    index = 2
    step = 1
    while step <= n
        previndex = index ÷ 2
        G[previndex+1 : index, :] = @views(G[1:previndex,:] .+ S[[step],:]) .% 2
        index *= 2
        step += 1
    end
    return G
end

shift(x) = 1 << (x-1)
"""
indexofcombination(C)

Return the index of the product of the stabilizer generators indexed by I in the generated group.
"""
function indexofcombination(C::AbstractArray{T}) where {T<:Integer}
    C = unique(C)
    C_bit = shift.(C)
    result = reduce(⊻, C_bit, init = 0) .+ 1
    return result
end
function indexofallcombinations(C::AbstractArray{T}) where {T<:Integer}
    C = unique(C)
    Indices = []
    for i in cartesianproduct([false,true], length(C))
        push!(Indices,indexofcombination(C[[i...]]))
    end
    return Indices
end
"""
indexofcombination(C)

Return the combination stabilizer generators (as a bitstring of length n) corresponding to index index in the generated group with n generators.

This is essentially the same as converting index to a bitstring
"""
function combinationofindex(index::Int,n::Int)
    result = falses(n)
    index = index - 1
    for i in 1:n
        result[i] = index & 1
        index = index >> 1
    end
    return result
end

"""
Index of a Pauli string according to bit-convention, use exchange = false for stabilizers and moments, exchange = true for errors and error rates
"""
function gf4toindex(r::PauliVector; exchange::Bool = false, n_d::Integer = size(r,1))
    rs = gf4tosymplectic_cat(r,exchange=exchange,n_d=n_d)
return bitvectortoindex(rs)
end
function gf4toindex(r::Integer; exchange::Bool = false, n_d::Integer = size(r,1))
    return gf4toindex([r],exchange=exchange,n_d=n_d)
end
function indextogf4(i::Integer; exchange=false)
    v = indextobitvector(i,2)
    return symplectictogf4(v,exchange=exchange)[1]
end

"""
pinv_Int(M::AbstractMatrix{T}) where T<:Integer

Pseudoinverse of M arising from Smith normal form.
"""
function pinv_int(M::AbstractMatrix{T}) where T<:Integer
    F=smith(M)
    S_inv,D,T_inv =F.Sinv,SmithNormalForm.diagm(F),F.Tinv
    D_inv = pinv_diag(D)
    M_inv = T_inv*D_inv*S_inv
    return M_inv
end
"""
pinv_diag(M::AbstractMatrix{T}) where {T}

Pseudoinverse of a diagonal matrix.

Invert non-zero elements and transpose.
"""
function pinv_diag(M::AbstractMatrix{T}) where {T}
    n = min(size(M)...)
    M_inv = zeros(size(M))
    for i=1:n
        M_inv[i,i] = (M[i,i] == 0 ? 0 : 1/M[i,i])
    end
    return transpose(M_inv)
end

"""
function divide_evenly(n::Integer, n_cells::Integer)

Divide n as evenly as possible into n_cells many parts.

Return n_cells many integers that sum to n and differ by at most 1.
"""
function divide_evenly(n::Integer, n_cells::Integer)
    Split = zeros(Int,n_cells)
    for i = 1:n_cells
        Split[i] = n ÷ n_cells
    end
    for i = 1:(n % n_cells)
        Split[i] += 1
    end
    return Split
end

"""
L1Error(Estimate,Actual)

Total variation distance between probability vectors.
"""
function L1Error(Estimate,Actual)
    Diff = Estimate .- Actual
    L1 = 0.5*sum(abs.(Diff),dims=1)
    return L1
end


#Bitstring labeled array
"""
struct BitlabeledVector{T}

Vector whose size is a power of 2 and which is indexed by BitVectors. It can also be indexed by Vector{Int}, which is interpreted as a BitVector with this support.
"""
struct BitlabeledVector{T}
    n::Int
    Data::Vector{T}
    function BitlabeledVector(Data::Vector{T}, n) where {T}
        n <= 63 || error("Only n <= 63 is supported") 
        length(Data) != 2^n ? error("Length must be equal to 2^n") : new{T}(n,Data)
    end
end
function bitvectortoindex(i::BitVector)
    ind = length(i.chunks) > 0 ? i.chunks[1] + 1 : 1 #If i is empty it is treated as 0
    return ind
end
function Base.getindex(V::BitlabeledVector, i::BitVector)
    ind = bitvectortoindex(i) #If i is empty it is treated as 0
    return V.Data[ind]
end
function Base.getindex(V::BitlabeledVector, i::Vector{Int})
    a = falses(V.n)
    a[i] .= 1
    return V[a]
end
function Base.setindex!(V::BitlabeledVector{T}, v::T, i::BitVector) where {T}
    ind = bitvectortoindex(i)
    V.Data[ind] = v
end
function Base.setindex!(V::BitlabeledVector{T}, v::T, i::Vector{Int}) where {T}
    a = falses(V.n)
    a[i] .= 1
    return V[a] = v 
end
function firstindex(V::BitlabeledVector)
    return falses(V.n)
end
function firstindex(V::BitlabeledVector)
    return trues(V.n)
end
"""
Return the BitVector of length n associated to index i in the Data of a BitlabeledVector
"""
function indextobitvector(i::Int, n::Int)
    v = BitVector(undef, n)
    v.chunks[1] = (i-1) # -1 because indexing starts with 1 in julia, but first bitstring label is 00000
    return v
end
function getdata(V::BitlabeledVector)
    return V.Data
end

function powerset(x::Vector{T}) where T
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    result
end