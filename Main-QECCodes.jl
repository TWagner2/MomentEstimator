#Type definitions
"""
struct QECC

Represents a quantum (data-syndrome) code with parity check Matrix H.
Each column of H is a stabilizer, acting on n_d data errors and n_m = (size(H,1) - n_d) measurement errors.
For data-syndrome codes usually n_m = size(H,2).
For normal stabilizer codes n_m = 0, since we only have data errors.
"""
struct QECC
    Name::String
    H::PauliMatrix
    n_d::Integer
end
#Types for sweep decoding: Embedding of a QECC into 2D
const Embedding2D = AbstractMatrix{T} where {T<:AbstractFloat}
"""
struct QECCGraph

Embedding of a qecc into 2D, used for sweep decoding.
"""
struct QECCGraph
    C::QECC
    CheckEmbedding::Embedding2D
    QubitEmbedding::Embedding2D
    H::PauliMatrix #The parity check matrix of C, for convenience and compatible interface e.g. in Union{QECC,QECCGraph}
    L::PauliMatrix #A set of Logical operators for the code, must also explicitly include a logical identity operator
    function QECCGraph(C::QECC, CheckEmbedding::Embedding2D, QubitEmbedding::Embedding2D, L::PauliMatrix)
        return new(C,CheckEmbedding,QubitEmbedding,C.H,L)
    end
end
function QECCGraph(H::PauliMatrix,CheckEmbedding::Embedding2D,QubitEmbedding::Embedding2D,L::PauliMatrix,Name::String, n_d::Integer)
    return QECCGraph( QECC(Name,H,n_d), CheckEmbedding, QubitEmbedding, L)
end
function write_params(Dest::Union{HDF5.File,HDF5.Group},C::QECC)
    attr = attributes(Dest)
    attr["Type"] = C.Name
    attr["n_data"] = C.n_d
    Dest["Stabilizers"] = Matrix(C.H) #Convert to dense matrix if necessary
end
function write_params(Dest::Union{HDF5.File,HDF5.Group},C::QECCGraph)
    write_params(Dest, C.C)
    Dest["CheckEmbedding"] = C.CheckEmbedding
    Dest["QubitEmbedding"] = C.QubitEmbedding
    Dest["LogicalOperators"] = Matrix(C.L)
end


"""
qecc_repetitioncode(n::Integer)

n-qubit repetition code.
Stabilizers are of X-type
"""
function qecc_repetitioncode(n::Integer)
    H = zeros( Int, (n,n-1))
    for j=1:n-1
        H[j+1,j] = 1
        H[j,j] = 1            
    end
    return QECC("Repetition Code", SparseMatrixCSC(H), n)
end
function qeccgraph_repetitioncode(n::Integer)
    H = qecc_repetitioncode(n)
    QubitEmbedding = Matrix{Float64}([ (1+2i)*j for j=1:-1:0,i=0:n-1 ] )
    CheckEmbedding = Matrix{Float64}([ (2+2i)*j for j=1:-1:0,i=0:n-2 ])
    L = ones(Int,(n,1)).*2
    L = hcat(zeros(Int,(n,1)), L)
    return QECCGraph(H,CheckEmbedding,QubitEmbedding,L)
end
function tn_repetitioncode(n::Integer, ErrorRates; Error = nothing)
    rep = qecc_repetitioncode(n)
    TN = construct_tensornetwork( rep, ErrorRates; Error = Error)
    return TN    
end
function tn_repetitioncode(n::Integer, p::Float64; Error = nothing)
    ErrorRates = repeat([1-p, 0, p, 0], outer = [1,n])
    return tn_repetitioncode(n,ErrorRates; Error = Error)
end

"""
qecc_surfacecode_regular(l::Integer)

Regular surface code on a square lattice of linear dimension l with smooth boundaries at the top and bottom and rough boundaries left and right.

X-stabilizers are on vertices, Z-stabilizers on faces, qubits on edges. 
Qubits are enumerated top left to bottom right row wise, first all horizontal qubits then all vertical ones.
Stabilizers are also enumerated top left to bottom right row wise, first all vertex then all face stabilizers.
"""
function qecc_surfacecode_regular(l::Integer)
    NCheckZ = (l-1)*l
    NCheckX = (l-1)*l 
    NQubitHorizontal = l^2
    NQubitVertical = (l-1)^2
    NQubit = NQubitHorizontal + NQubitVertical
    H = zeros(Int,(NQubit, NCheckX + NCheckZ))
    #Connections of horizontal qubits
    for i=1:l
        for j=1:l
            QubitIndex = l*(i-1) + j #account for the fact that we skip vertical rows 
            VertexIndex = (l-1)*(i-1) + j
            FaceIndex = l*(i-1) + j + NCheckX
            if j < l
                H[QubitIndex, VertexIndex] = 1 #Vertex to the right
            end
            if j > 1
                H[QubitIndex, VertexIndex-1] = 1 #Vertex to the left
            end
            if i < l
                H[QubitIndex, FaceIndex] = 2 #Face below
            end
            if i > 1
                H[QubitIndex, FaceIndex-l] = 2 #Face above
            end
                
        end
    end
    #Connections of vertical qubits
    for i=1:(l-1)
        for j=1:(l-1)
            QubitIndex = (l-1)*(i-1) + l^2 + j
            VertexIndex = (l-1)*(i-1) + j
            FaceIndex = l*(i-1) + j + NCheckX
            H[QubitIndex, VertexIndex] = 1 #Vertex above
            H[QubitIndex, VertexIndex + (l-1)] = 1 #Vertex below
            H[QubitIndex, FaceIndex] = 2 #Face to the left
            H[QubitIndex, FaceIndex+1] = 2 #Face to the right
        end
    end
    return QECC("Surface code square regular", SparseMatrixCSC(H), NQubit)
end
function qeccgraph_surfacecode_regular(l::Integer)
    H = qecc_surfacecode_regular(l)
    N_S = l*(l-1)
    NQubitHorizontal = l^2
    NQubitVertical = (l-1)^2
    NQubit = NQubitHorizontal + NQubitVertical
    CheckEmbeddingVertex = zeros(Float64,(2,N_S))
    index = 1
    for i=1:l
        for j=1:l-1
            CheckEmbeddingVertex[:,index] = [1.0*j, -1.0*i]
            index += 1
        end
    end
    CheckEmbeddingFaces = zeros(Float64,(2,N_S))
    index = 1
    for i=1:l-1
        for j=1:l
            CheckEmbeddingFaces[:,index] = [-0.5+1.0*j, -0.5-1.0*i]
            index += 1
        end
    end
    CheckEmbedding = hcat(CheckEmbeddingVertex,CheckEmbeddingFaces)
    QubitEmbeddingHorizontal = zeros(Float64,(2,NQubitHorizontal))
    index = 1
    for i=1:l
        for j=1:l
            QubitEmbeddingHorizontal[:,index] = [-0.5+1.0*j, -1.0*i]
            index += 1
        end
    end
    QubitEmbeddingVertical = zeros(Float64,(2,NQubitVertical))
    index = 1
    for i=1:l-1
        for j=1:l-1
            QubitEmbeddingVertical[:,index] = [1.0*j, -0.5-1.0*i]
            index += 1
        end
    end
    QubitEmbedding = hcat(QubitEmbeddingHorizontal,QubitEmbeddingVertical)
    L_X =  zeros(Int, (NQubit,1)) #Logical X-Operator = From Top to Bottom 
    for i=1:l
        L_X[l*i,1] = 1
    end
    L_Z = zeros(Int,(NQubit,1)) #Logical Z-Operator = From Left to Right
    for j=1:l
        L_Z[j,1] = 2
    end
    L_Y = addgf4.(L_X,L_Z)
    L_I = zeros(Int,(NQubit,1))
    L = SparseMatrixCSC(hcat(L_I,L_X,L_Z,L_Y))
    return QECCGraph(H,CheckEmbedding,QubitEmbedding,L)
end


"""
qecc_repeat_measurements(C::QECC, l::Integer)

Simplest data syndrome code version of any qecc. Each measurement is repeated l times.
"""
function qecc_repeat_measurements(C::QECC, l::Integer)
    H_d = C.H
    H = repeat(H_d,outer = (1,l))
    H = vcat(H,Matrix(I,size(H,2),size(H,2)))
    return QECC(C.Name * "_repeated", H, size(H_d,1))
end

#Subcodes etc.
"""
adjacent(H::AbstractMatrix,Q)


Returns all column / Row indices such that the column / row is non-zero on at least 1 element in the index set I.

Return the indices in H of all stabilizers / Qubits that are adjacent to at least one Qubit / Stabilizer in I.
I can be any type of index for the second dimension of H (scalar, logical).
"""
function adjacent(H::AbstractMatrix,I::AbstractArray,Row)
    Adjacent = Int[]
    if !Row
        H = H'
    end
    HI = @view H[I,:]
    for j in axes(HI,2)
        for i in axes(HI,1)
            if HI[i,j] != zero(HI[i,j])
                push!(Adjacent,j)
                break
            end
        end
    end
    return Adjacent
end
function adjacent(H::AbstractMatrix,I::Integer,Row)
    I = [I]
    return adjacent(H,I,Row)
end
"""
adjacent_nhop(H,I,Row,n)

Return the N-Hop neighbourhodd of the given rows / columns.

E.g. for Row = true and n = 2, it returns all columns adjacent to the given rows and all rows adjacent to these columns.
"""
function adjacent_nhop(H::AbstractMatrix,I::AbstractArray,Row,n::Integer)
    R = Int[]
    C = Int[]
    Row ? R = I : C = I
    for i=1:n
        if Row
            C = adjacent(H,R,Row)
        else
            R = adjacent(H,C,Row)
        end
        Row = !Row
    end
    return R,C
end
function adjacent_nhop(H::AbstractMatrix,I::Integer,Row, n::Integer)
    I = [I]
    return adjacent_nhop(H,I,Row,n)
end

