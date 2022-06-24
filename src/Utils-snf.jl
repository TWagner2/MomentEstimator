#TODO: Speed up by devectorizing? (especially add_rows! etc...)?

function add_rows!(M::AbstractMatrix{Bool}, i::Integer, j::Integer, r=1)
    M[i,:] = M[i,:] .⊻ (r.*M[j,:])
end
function add_cols!(M::AbstractMatrix{Bool}, i::Integer, j::Integer, r=1)
    M[:,i] = M[:,i] .⊻ (r.*M[:,j])
end
function swap_rows!(M::AbstractMatrix{Bool}, i::Integer, j::Integer)
    for k in eachindex(M[i,:])
        temp = M[i,k]
        M[i,k] = M[j,k]
        M[j,k] = temp
    end   
end
function swap_cols!(M::AbstractMatrix{Bool}, i::Integer, j::Integer)
    for k in eachindex(M[:,j])
        temp = M[k,i]
        M[k,i] = M[k,j]
        M[k,j] = temp
    end
end

"""
smith_normal_form_z2(M::AbstractMatrix{Bool})

Compute the smith normal form of a matrix over Z_2.

Returns D,S,T such that S*Mat*T = D. Note that this is in contrast to SmithNormalForm package, which return D,S,T with S*D*T = M.
following: https://github.com/sauln/smith-normal-form
This algorithm only works over Z_2
"""
function smith_normal_form_z2(M::AbstractMatrix{Bool})
    M = copy(M)
    return smith_normal_form_z2!(M)
end
function smith_normal_form_z2!(M::AbstractMatrix{Bool})
    n = size(M)[1]
    m = size(M)[2]
    S = falses((n,n)) #Tracks row operations
    for i=1:n
        S[i,i] = 1
    end
    T = falses((m,m)) #Tracks column operations
    for i=1:m
        T[i,i] = 1
    end    
    i = 1
    index = nonzeroindex_cartesian(M)
    while !isnothing(index)
        j,k = index[1] + (i-1), index[2] + (i-1)
        smith_iteration!(M,S,T,i,j,k,n,m)
        i += 1
        index = nonzeroindex_cartesian(M[i:end,i:end])
    end
    return M,S,T
end
function smith_iteration!(M::AbstractMatrix{Bool},S::AbstractMatrix{Bool},T::AbstractMatrix{Bool},i::Integer,j::Integer,k::Integer,n::Integer,m::Integer)
    if (j,k) != (i,i)
        swap_rows!(M,i,j)
        swap_rows!(S,i,j)
        swap_cols!(M,i,k)
        swap_cols!(T,i,k)
    end
    for h=i+1:n
        if M[h,i] == 1
            add_rows!(M,h,i)
            add_rows!(S,h,i)
        end
    end 
    for l=i+1:m
        if M[i,l]==1
            add_cols!(M,l,i)
            add_cols!(T,l,i)
        end
    end    
end

#Pseudoinverse over Z_2
function pinv_z2(M::AbstractMatrix{Bool})
    D,S,T = smith_normal_form_z2(M)
    R = T*transpose(D)*S .% 2
    R = Bool.(R)
    return R
end

#TODO: Is there a way to implement smith normal form algorithm efficiently on sparse matrices?
function pinv_z2(M::AbstractSparseMatrix{Bool})
    M = BitMatrix(M)
    D,S,T = smith_normal_form_z2(M)
    D = SparseMatrixCSC(D)
    S = SparseMatrixCSC(S)
    T = SparseMatrixCSC(T)
    R = T*transpose(D)*S .% 2
    R = Bool.(R)
    return R
end
