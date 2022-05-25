"""
Types describing regularizers for the least.squares estimator.
"""

abstract type AbstractRegularizer end
abstract type AbstractTiledRegularizer end

"""
Adds an additional square Term of the form (E - E_av)' * Wt * (E - E_av) to the cost function.
Corresponds to a Gaussian prior with mean E_av and precision Wt over the parameters E.
"""
struct Regularizer_L2{T<:Real}  <: AbstractRegularizer
    E_av::AbstractVector{T} #Average moments of the prior
    Wt::AbstractMatrix{T} #Strength of regularization, should be inversly proportional to variance of prior
    function Regularizer_L2(E_av::AbstractVector{T},Wt::AbstractMatrix{T}) where {T<:Real}
        return new{T}(E_av,Wt)
    end
    function Regularizer_L2(E_av::AbstractVector{T}, α::AbstractVector{T}) where {T<:Real}
        Wt = diagm(α)
        return new{T}(E_av, Wt)
    end
    function Regularizer_L2(E_av::AbstractVector{T}, α::T) where {T<:Real}
        n = length(E_av)
        v = ones(n) * α
        return Regularizer_L2(E_av,v)
    end
end
function Regularizer_L2_From_P(p,α,n)
    p = expand_errorrates(p,n)
    E_av = momentsfromrates(p)[2:end,:]
    E_av = vec(E_av') #flatten row wise
    E_av = Vector(E_av)
    return Regularizer_L2(E_av,α)
end
function  regularizer_cost(E, R::Regularizer_L2, nsamples::Int)
    return 0.5* ((E - R.E_av)'* R.Wt * (E .- R.E_av)) / nsamples #regularization should decay linearly with sample size since precision of cost is proportional to sample size
end
function regularizer_jacobian(E, R::Regularizer_L2, nsamples::Int)
    return ((E - R.E_av)' * R.Wt) / nsamples
end
function regularizer_hessian(E, R::Regularizer_L2, nsamples::Int)
    return R.Wt / nsamples
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, R::Regularizer_L2)
    FileParams = create_group(Dest,"Regularizer")
    FileParams["Name"] = "Regularizer_L2"
    FileParams["E_av"] = R.E_av
    FileParams["Wt"] = R.Wt
end

"""
Returns one regularizer for each region to make regularization compatible with local / tiled estimator.
"""
struct TiledRegularizer_L2{T <: Real} <: AbstractTiledRegularizer
    E_av::AbstractMatrix{T} #Average moments for each qubit as an 3xn_qubit array
    Wt::AbstractMatrix{T} #Strength of regularization, should be inversly proportional to variance of prior, 3n x 3n matrix with ordered as X-moments, then Z-moments, then Y-moments
    function TiledRegularizer_L2(E_av::AbstractVector{T},Wt::AbstractMatrix{T}) where {T<:Real}
        return new{T}(E_av,Wt)
    end
    function TiledRegularizer_L2(E_av::AbstractVector{T}, α::AbstractVector{T}) where {T<:Real}
        Wt = diagm(α)
        return new{T}(E_av, Wt)
    end
    function TiledRegularizer_L2(E_av::AbstractVector{T}, α::T) where {T<:Real}
        n = size(E_av,2)
        v = ones(3*n) * α
        return new{T}(E_av, v)
    end
end
function TiledRegularizer_L2_From_P(p,α,n)
    p = expand_errorrates(p,n)
    E_av = momentsfromrates(p)[2:end,:]
    return TiledRegularizer_L2(E_av,α)
end
function select_local_regularizer(QubitIndices,Regularizer::TiledRegularizer_L2)
    n = size(Regularizer.E_av,1)
    E_av = Regularizer.E_av[:,QubitIndices]
    E_av = vec(E_av') #Flatten according to correct order: First all X moments, then all Z moments etc. (thus Row wise)
    E_av = Vector(E_av)
    Indices = cat(QubitIndices, n+QubitIndices, 2*n + QubitIndices; dims=1)
    Wt = Regularizer.Wt[Indices,Indices]
    return Regularizer_L2(E_av, Wt)
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, R::TiledRegularizer_L2)
    FileParams = create_group(Dest,"Regularizer")
    FileParams["Name"] = "TiledRegularizer_L2"
    FileParams["E_av"] = R.E_av
    FileParams["Wt"] = R.Wt
end

"""
Same averages for each qubit and block diagonal weights over qubits
"""
struct TiledRegularizer_L2_Repeating{T <: Real} <: AbstractTiledRegularizer
    E_av::AbstractVector{T} #Average moments for one qubit as an 3 element array
    Wt::AbstractMatrix{T} #3x3 Weight matrix
    function TiledRegularizer_L2_Repeating(E_av::AbstractVector{T},Wt::AbstractMatrix{T}) where {T<:Real}
        return new{T}(E_av,Wt)
    end
    function TiledRegularizer_L2_Repeating(E_av::AbstractVector{T}, α::AbstractVector{T}) where {T<:Real}
        Wt = diagm(α)
        return new{T}(E_av, Wt)
    end
    function TiledRegularizer_L2_Repeating(E_av::AbstractVector{T}, α::T) where {T<:Real}
        n = size(E_av,2)
        v = ones(3*n) * α
        return new{T}(E_av, v)
    end
end
function TiledRegularizer_L2_Repeating_From_P(p,α)
    p = expand_errorrates(p,1)
    E_av = dropdims(momentsfromrates(p)[2:end,:];dims=2)
    return TiledRegularizer_L2_Repeating(E_av,α)
end
function select_local_regularizer(QubitIndices,Regularizer::TiledRegularizer_L2_Repeating)
    n = length(QubitIndices)
    E_av = repeat(Regularizer.E_av, inner=(n))
    Wt = zeros((3*n,3*n))
    for j in axes(Regularizer.Wt,2)
        for i in axes(Regularizer.Wt,1)
            for q in 1:n
                Wt[(i-1)*n + q,(j-1)*n + q] = Regularizer.Wt[i,j] #Essentially blockdiagonal, but we first have all X moments, then all Z moments etc.
            end
        end
    end
    return Regularizer_L2(E_av, Wt)
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, R::TiledRegularizer_L2_Repeating)
    FileParams = create_group(Dest,"Regularizer")
    FileParams["Name"] = "TiledRegularizer_L2_Repeating"
    FileParams["E_av"] = R.E_av
    FileParams["Wt"] = R.Wt
end

