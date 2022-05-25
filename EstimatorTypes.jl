"""
Types describing different estimators of the moments from syndrome expectations
"""

abstract type AbstractEstimator end

"""
Estimator using Moore-Penrose pseudo-inverse
"""
Parameters.@with_kw struct Estimator_pinv <: AbstractEstimator
    β::Float64=0.5
    Chunksize::Int=10^3 
    select::Union{Int,Nothing} = nothing
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, Estimator::Estimator_pinv)
    FileParams = attributes(Dest)
    FileParams["beta"] = Estimator.β
    FileParams["Estimator"] = "Estimator_pinv"
    FileParams["select"] = select !== nothing ? select : Inf
end
function estimate(C::QECC,S,Estimator::Estimator_pinv)
    return estimatemoments_pinv(C,S;Params=Estimator)
end

"""
Estimator using Smith normal form psuedo-inverse
"""
Parameters.@with_kw struct Estimator_pinv_snf <: AbstractEstimator
    β::Float64=0.5
    Chunksize::Int=10^3 
    select::Union{Int,Nothing} = nothing
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, Estimator::Estimator_pinv_snf)
    FileParams = attributes(Dest)
    FileParams["beta"] = Estimator.β
    FileParams["Estimator"] = "Estimator_pinv_snf"
    FileParams["select"] = select !== nothing ? select : Inf
end
function estimate(C::QECC,S,Estimator::Estimator_pinv_snf)
    return estimatemoments_pinv_snf(C,S;Params=Estimator)
end

"""
Estimator using least squares with Optim package
"""
Parameters.@with_kw struct Estimator_lsq_optim <: AbstractEstimator
    β::Float64 = 0.5
    Chunksize::Int = 10^3
    Regularizer::Union{AbstractRegularizer,Nothing} = nothing
    Wt = I
    n_step::Int = 1
    select::Union{Int,Nothing} = nothing
    Options::Optim.Options = Optim.Options()
end
function set_regularizer(Estimator::Estimator_lsq_optim, Reg::AbstractRegularizer)
    @unpack_Estimator_lsq_optim Estimator
    return Estimator_lsq_optim(β,Chunksize,Reg,Wt,n_step,select,Options)
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, Estimator::Estimator_lsq_optim)
    FileParams = attributes(Dest)
    FileParams["beta"] = Estimator.β
    if typeof(Estimator.Wt) <: UniformScaling
        FileParams["Wt"] = Estimator.Wt.λ
    else
        FileParams["Wt"] = Estimator.Wt
    end
    FileParams["n_step"] = Estimator.n_step
    FileParams["select"] = Estimator.select !== nothing ? Estimator.select : Inf
    FileParams["Estimator"] = "Estimator_lsq_optim"
    FileParams["Optim_Options"] = string(Estimator.Options)
    if Estimator.Regularizer !== nothing
        write_params(Dest,Estimator.Regularizer)
    end
end
function estimate(C::QECC,S,Estimator::Estimator_lsq_optim)
    return estimatemoments_lsq_optim(C,S; Params=Estimator)
end

"""
Estimate the rates of all qubits by applying a local estimator or solver for each qubit separately.
"""
Parameters.@with_kw struct Estimator_applyall_local_nhop <: AbstractEstimator
    LocalEstimator::AbstractEstimator = Estimator_lsq_optim()
    n_hop::AbstractVector{Int}
    Regularizer::Union{AbstractTiledRegularizer,Nothing} = nothing
end
function write_params(Dest::Union{HDF5.File,HDF5.Group}, Estimator::Estimator_applyall_local_nhop)
    Dest["n_hop"] = Estimator.n_hop
    FileParams = attributes(Dest)
    FileParams["Estimator"] = "Estimator_applyall_local_nhop"
    if Estimator.Regularizer !== nothing
        write_params(Dest,Estimator.Regularizer)
    end
    Local = create_group(Dest, "LocalEstimator")
    write_params(Local,Estimator.LocalEstimator)
end
function estimate(C::QECC,S,Estimator::Estimator_applyall_local_nhop)
    return applyall_local_nhop(C,S; Params=Estimator)
end

