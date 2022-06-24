using Test
include("Main.jl")

@testset "Misc Utils" begin
    a = (1.3,-5)
    b = (1.8,-6)
    @test issmaller_ldexp(a,b) == false
    a = (0,9)
    b = (1.3,5)
    @test issmaller_ldexp(a,b) == true
end
@testset "Moments" begin
    P = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    E = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    for i in axes(P,1)
        e = momentsfromrates(P[i])
        @test e == E[i]
        @test ratesfrommoments(e[2:end],Omit=true) == P[i]
    end

    E = Vector(LinRange(1,0,9))
    @test momentsbyqubits(E)[1] == [1 0.875 0.75; 0.625 00.5 0.375; 0.25 0.125 0.0]
    @test isempty(momentsbyqubits(E)[2])
end

@testset "Error Rates expand" begin
    p = 0.3
    @test expand_errorrates(p,3) ≈ [0.7 0.7 0.7; 0.1 0.1 0.1; 0.1 0.1 0.1; 0.1 0.1 0.1]
    p = [0.9,0.05,0.03,0.02]
    @test expand_errorrates(p,3) == [0.9 0.9 0.9; 0.05 0.05 0.05; 0.03 0.03 0.03; 0.02 0.02 0.02]
    p = [0.9 0.6; 0.05 0.3; 0.03 0.04; 0.02 0.06]
    @test expand_errorrates(p,2) == [0.9 0.6; 0.05 0.3; 0.03 0.04; 0.02 0.06]

end
@testset "Stabilizer group expansion" begin
    H = qecc_surfacecode_regular(2).H
    G = construct_group(H)
    @test G == [0 0 0 0 0; 1 1 0 0 1; 0 0 1 1 1; 1 1 1 1 0; 2 0 2 0 2; 3 1 2 0 3; 2 0 3 1 3; 3 1 3 1 2; 0 2 0 2 2; 1 3 0 2 3; 0 2 1 3 3; 1 3 1 3 2; 2 2 2 2 0; 3 3 2 2 1; 2 2 3 3 1; 3 3 3 3 0]'
    S = BitMatrix([0 1 0; 1 0 0; 0 1 1])
    Expected = [0 0 0
                0 1 0
                1 0 0
                1 1 0
                0 1 1
                0 0 1
                1 1 1
                1 0 1]
    @test expand_syndromes(S) == Expected
    @test expectations_from_syndromes(S) ≈ [3, 1, 1, -1, -1, 1, -3, -1] ./ 3
    @test expectations_from_syndromes(S,Chunksize=2) ≈ [3, 1, 1, -1, -1, 1, -3, -1] ./ 3
    @test expectations_from_syndromes(S,Chunksize=1) ≈ [3, 1, 1, -1, -1, 1, -3, -1] ./ 3
    S = SparseMatrixCSC(S)
    @test expand_syndromes(S) == Expected
    @test expectations_from_syndromes(S) ≈ [3, 1, 1, -1, -1, 1, -3, -1] ./ 3

    S = BitVector([0,1,0])
    @test expand_syndromes(S) == [0,0,1,1,0,0,1,1]
 
    M = [1 2 3; 3 1 2; 2 2 2]
    B = grouptobinomial(M)
    @test B == [1 0 0; 0 1 0; 0 0 0; 0 1 0; 0 0 1; 1 1 1; 0 0 1; 1 0 0; 0 0 0]
    M = [1 2 3; 3 1 2; 2 2 2; 1 0 1; 1 1 0]
    @test grouptobinomial(M,n_d=3) == [1 0 0; 0 1 0; 0 0 0; 0 1 0; 0 0 1; 1 1 1; 0 0 1; 1 0 0; 0 0 0; 1 0 1; 1 1 0]

    M = [1,2,3,4,5,6]
    indices = []
    combinations = []
    for i in cartesianproduct([false,true],6)
        push!(indices, indexofcombination(M[[i...]]))
        push!(combinations,i)
    end
    @test indices == 1:64
    @test all(combinationofindex.(indices,6) .== BitVector.(combinations))
    M = [1,3,6]
    Indices = indexofallcombinations(M)
    @test Indices == [1,2,5,6,33,34,37,38]

    Es = [1.0,0.9,0.8,0.7]
    V = zeros(Float64,(4,4))
    S = zeros(Float64,(4,4))
    covariance_syndromes!(V,Es)
    squareexpectation_syndromes!(S,Es)
    @test S == [1.0 0.9 0.8 0.7; 0.9 1.0 0.7 0.8; 0.8 0.7 1.0 0.9; 0.7 0.8 0.9 1.0]
    @test V == S - Es*Es'
end

@testset "Integer snf" begin
    M = [2 4 4; -6 6 12; 10 -4 -16]
    H = qecc_surfacecode_regular(3).H
    Mi = pinv_int(M)
    Hi = pinv_int(H)
    @test M*Mi*M ≈ M
    @test H*Hi*H ≈ H
    @test Mi*M*Mi ≈ Mi
    @test Hi*H*Hi ≈ Hi
end

@testset "Symplectic" begin
    H = transpose([3 2 1; 1 2 3; 2 2 2])
    E = [1, 0, 2]
    E_symp = vcat(gf4tosymplectic(E)...)
    H_symp = vcat(gf4tosymplectic(H)...)

    @test symplectictogf4(gf4tosymplectic([1,3,2,3,1,2,1,0])...) == [1,3,2,3,1,2,1,0]
    @test symplectictogf4(gf4tosymplectic(E)...) == E
    @test symplectictogf4(gf4tosymplectic(H)...) == H

    @test scalar_commutator_v(H,E) == [0, 1, 1]

    @test E_symp == [1,0,0,0,0,1]
    @test H_symp == transpose([1 0 1 1 1 0; 1 0 1 0 1 1; 0 0 0 1 1 1])

    @test xzpart(E_symp) == (Bool[1,0,0], Bool[0,0,1], BitArray([]))
    @test xzpart(H_symp) == (Bool[1 1 0; 0 0 0; 1 1 0], Bool[1 0 1; 1 1 1; 0 1 1], BitMatrix(zeros((0,3))))

    H_s = SparseMatrixCSC(H)
    E_s = SparseVector(E)
    E_symp_s = vcat(gf4tosymplectic(E_s)...)
    H_symp_s = vcat(gf4tosymplectic(H_s)...)

    @test symplectictogf4(gf4tosymplectic(SparseVector([1,3,2,3,1,2,1,0]))...) == SparseVector([1,3,2,3,1,2,1,0])
    @test symplectictogf4(gf4tosymplectic(E_s)...) == E_s
    @test symplectictogf4(gf4tosymplectic(H_s)...) == H_s

    @test scalar_commutator_v(H_s,E_s) == [0, 1, 1]

    @test E_symp_s == [1,0,0,0,0,1]
    @test H_symp_s == transpose([1 0 1 1 1 0; 1 0 1 0 1 1; 0 0 0 1 1 1])

    @test xzpart(E_symp_s) == ([1,0,0], [0,0,1], BitArray([]))
    @test xzpart(H_symp_s) == (transpose([1 0 1; 1 0 1; 0 0 0]), transpose([1 1 0; 0 1 1; 1 1 1]), BitMatrix(zeros((0,3))))
end

@testset "Symplectic-MeasurementErrors" begin
    n_d = 3
    H = transpose([3 2 1 1 0 0 0; 1 2 3 0 1 0 0; 2 2 2 0 0 1 0; 3 1 1 0 0 0 1])
    E = [1, 0, 2, 1, 0, 1, 0]
    E_symp = gf4tosymplectic_cat(E;exchange=true,n_d=n_d)
    H_symp = gf4tosymplectic_cat(H;n_d=n_d)

    @test symplectictogf4(gf4tosymplectic([1,3,2,3,1,2,1,0,1,0,1,0],n_d = 8)...) == [1,3,2,3,1,2,1,0,1,0,1,0]
    @test symplectictogf4(gf4tosymplectic(E,n_d=n_d)...) == E
    @test symplectictogf4(gf4tosymplectic(H,n_d=n_d)...) == H
    @test scalar_commutator_v(H,E,n_d=n_d) == [1, 1, 0, 0]
    @test E_symp == [0,0,1,1,0,0,1,0,1,0]
    @test H_symp == transpose([1 0 1 1 1 0 1 0 0 0; 1 0 1 0 1 1 0 1 0 0; 0 0 0 1 1 1 0 0 1 0; 1 1 1 1 0 0 0 0 0 1])
    @test xzpart(E_symp,n_d=n_d) == (Bool[0,0,1], Bool[1,0,0], Bool[1,0,1,0])
    @test xzpart(H_symp,n_d=n_d) == (Bool[1 1 0 1; 0 0 0 1; 1 1 0 1], Bool[1 0 1 1; 1 1 1 0; 0 1 1 0], Bool[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1])
end

#Smith normal form
@testset "smith normal form" begin
M = BitArray([1 1 0 0 0 0 0; 1 0 1 1 1 0 0; 0 1 1 0 0 1 0; 0 0 0 1 0 1 1; 0 0 0 0 1 0 1])
Expected = [1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 1 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 0 0 0]
D,S,T = smith_normal_form_z2(M)
@test D == Expected
@test S*M*T .% 2 == D

M = SparseMatrixCSC(M)
Expected = SparseMatrixCSC(Expected)
D,S,T = smith_normal_form_z2(M)
@test D == Expected
@test S*M*T .% 2 == D
end

@testset "Codes" begin
#Code generation
Rep = qeccgraph_repetitioncode(5)
@test Rep.C.H == [1 0 0 0; 1 1 0 0; 0 1 1 0; 0 0 1 1; 0 0 0 1]
@test Rep.CheckEmbedding == Float64.([2 4 6 8; 0 0 0 0])
@test Rep.QubitEmbedding == Float64.([1 3 5 7 9; 0 0 0 0 0])
@test Rep.L == [0 2; 0 2; 0 2; 0 2; 0 2]

C3 = qeccgraph_surfacecode_regular(3)
@test C3.C.H == transpose([1 1 0 0 0 0 0 0 0 1 0 0 0
            0 1 1 0 0 0 0 0 0 0 1 0 0
            0 0 0 1 1 0 0 0 0 1 0 1 0
            0 0 0 0 1 1 0 0 0 0 1 0 1 
            0 0 0 0 0 0 1 1 0 0 0 1 0
            0 0 0 0 0 0 0 1 1 0 0 0 1
            2 0 0 2 0 0 0 0 0 2 0 0 0
            0 2 0 0 2 0 0 0 0 2 2 0 0
            0 0 2 0 0 2 0 0 0 0 2 0 0
            0 0 0 2 0 0 2 0 0 0 0 2 0
            0 0 0 0 2 0 0 2 0 0 0 2 2
            0 0 0 0 0 2 0 0 2 0 0 0 2])
@test C3.CheckEmbedding == Float64.([1 2 1 2 1 2 0.5 1.5 2.5 0.5 1.5 2.5; -1 -1 -2 -2 -3 -3 -1.5 -1.5 -1.5 -2.5 -2.5 -2.5])
@test C3.QubitEmbedding == Float64.([0.5 1.5 2.5 0.5 1.5 2.5 0.5 1.5 2.5 1 2 1 2; -1 -1 -1 -2 -2 -2 -3 -3 -3 -1.5 -1.5 -2.5 -2.5])
@test C3.L == transpose([0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 1 0 0 1 0 0 1 0 0 0 0; 2 2 2 0 0 0 0 0 0 0 0 0 0; 2 2 3 0 0 1 0 0 1 0 0 0 0])

Q = [1,12]
AdjQ = adjacent(C3.C.H,Q,true)
S = [2,8]
AdjS = adjacent(C3.C.H,S,false)
@test Set(AdjQ) == Set([1,7,3,5,10,11])
@test Set(AdjS) == Set([2,3,11,5,10])
Q = 1
AdjQ2 = adjacent_nhop(C3.C.H,Q,true,2)
@test Set(AdjQ2[2]) == Set([1,7])
@test Set(AdjQ2[1]) == Set([1,2,10,1,4,10])
S = 2
AdjS2 = adjacent_nhop(C3.C.H,S,false,2)
@test Set(AdjS2[2]) == Set([1,2,8,9,4])
@test Set(AdjS2[1]) == Set([2,3,11])
end

@testset "Codes Data Syndrome" begin
   C3 = qecc_repeat_measurements(qecc_surfacecode_regular(3), 3)
   H_d = qecc_surfacecode_regular(3).H
   @test C3.H[1:13,1:12] == H_d
   @test C3.H[1:13,13:24] == H_d
   @test C3.H[1:13,25:36] == H_d
   @test C3.H[14:end,:] == Matrix(I,36,36)
   @test C3.n_d == 13
end

@testset "Decoding Setup" begin
    C = qeccgraph_surfacecode_regular(3)
    E = [2,2,2,0,0,0,0,0,0,0,0,0,0]
    @test islogicalerror(C,E) == true
    E = [1,1,0,0,0,0,0,0,0,1,0,0,0]
    @test islogicalerror(C,E) == false
end

@testset "Trivial decoding" begin
H = qecc_surfacecode_regular(3).H
S = Bool[1,0,0,0,0,0,0,0,0,0,0,0]
E = decode_trivial(H,S)
@test scalar_commutator_v(H,E) == S
S = rand([false,true], (12))
E = decode_trivial(H,S)
@test scalar_commutator_v(H,E) == S    
end

@testset "TN decoding" begin
    Rep = qeccgraph_repetitioncode(5)
    S = Bool[0,1,0,0]
    ErrorRates = repeat([0.9,0.1,0,0],outer = [1,5]) #Note that error rates are passed in order Z,X,Y while stabilizers are have order X,Z,Y
    TN = construct_tensornetwork(Rep)
    @test decode_tn(Rep,S,ErrorRates, TN, 2,32) == [2,2,0,0,0]
    S = Bool[0,0,1,0]
    @test decode_tn(Rep,S,ErrorRates, TN,2,32) == [0,0,0,2,2] #Trivial decoder fails for this one
    #TODO: More sophisticated tests, surface codes
end

@testset "Projections" begin
    P = Float64[1,0,0,0]
    project_probabilitysimplex!(P)
    @test P == [1,0,0,0]
    P = Float64[0.7,-1,0,1]
    project_probabilitysimplex!(P)
    @test P == [0.35,0,0,0.65]
    P = Float64[1,1,0,0]
    project_probabilitysimplex!(P)
    @test P == [0.5,0.5,0,0]
    P = Float64[0.9,-0.1,0.1,0.1]
    project_probabilitysimplex!(P)
    @test P ≈ [0.8 + 2/30,0, 2/30, 2/30]
    P = Float64[1 0.7; 0 -1; 0 0; 0 1]
    project_probabilitysimplex!(P)
    @test P == [1 0.35; 0 0; 0 0; 0 0.65]
    P = Float64[1;1;0;0;; 0.7;-1;0;1;;; 0.9;-0.1;0.1;0.1;; 1;0;0;0]
    project_probabilitysimplex!(P)
    @test P ≈ [0.5;0.5;0;0;; 0.35;0;0;0.65;;; 0.8 + 2/30;0; 2/30; 2/30;; 1;0;0;0]
end

@testset "Binomials" begin
    M = [0 1 2 3; 4 5 6 7; 8 9 10 11]
    E = [4, 3, 2]
    R = binomexp(E,M)
    @test R == [ 20736, 497664, 11943936, 286654464]

end
@testset "Optimization" begin
    M = [0 1 2 3;
         4 5 6 7;
         8 9 10 11]
    E = [4, 3, 2]
    Es = [20737, 497665, 11943937, 286654465]
    @test cost_optim(E,Es,M;nsamples=1) == 0.5*[ 1,1,1,1]'*[1,1,1,1]
    J = [0.0 4*3^3*2^8 3^4*8*2^7;
         3^5*2^9 4*5*3^4*2^9 4*3^5*9*2^8;
         2*4*3^6*2^10 4^2*6*3^5*2^10 4^2*3^6*10*2^9;
         3*4^2*3^7*2^11 4^3*7*3^6*2^11 4^3*3^7*11*2^10]
    @test binomexp_jacobian(E,M) == J
    @test cost_optim_g(E,Es,M,nsamples=1) ≈ ((binomexp(E,M)-Es)'* J)'
    H2 = [0.0+0.0+2*4^0*3^6*2^10+3*2*4^1*3^7*2^11 0.0+4^1*5*3^5*2^9+2*4^1*6*3^5*2^10+3*4^2*7*3^6*2^11 0.0+4^1*3^6*9*2^8+2*4^1*3^6*10*2^9+3*4^2*3^7*11*2^10;
          0.0+4^1*3^6*9*2^8+2*4^1*3^6*10*2^9+3*4^2*3^7*11*2^10 1*4*3*3^2*2^8+4^1*5*4*3^3*2^9+4^2*6*5*3^4*2^10+4^3*7*6*3^5*2^11 1*4*3^3*8*2^7+4^1*5*3^4*8*2^8+4^2*6*3^5*10*2^9+4^3*7*3^6*11*2^10;
          0.0+4^1*3^6*9*2^8+2*4^1*3^6*10*2^9+3*4^2*3^7*11*2^10 1*4*3^3*8*2^7+4^1*5*3^4*8*2^8+4^2*6*3^5*10*2^9+4^3*7*3^6*11*2^10 1*3^4*8*7*2^6+4^1*3^5*9*8*2^7+4^2*3^6*10*9*2^8+4^3*3^7*11*10*2^9]
    @test cost_optim_h(E,Es,M;nsamples=1, Wt=Matrix(I,4,4)) ≈ J'*J + cost_optim(E,Es,M;nsamples=1)*H2
end

@testset "Noisemodels" begin
Chan = channel_nnsurfacecode(3,0.3,0.1)
ExpectedSupports = Set([[1,2],[1,10],[2,10],[2,3],[2,11],[3,11],[4,5],[4,10],[4,12],[5,10],[5,12],[10,12],[5,6],[5,11],[5,13],[6,11],[6,13],[11,13],[7,8],[7,12],[8,12],[8,9],[8,13],[9,13],[1,4],[2,5],[10,11],[3,6],[4,7],[5,8],[12,13],[6,9]])
ExpectedSupports = Set([cat(S,S.+13,dims=1) for S in ExpectedSupports])
Supports = Set([F.Support for F in Chan.FG.Factors])
@test Supports == ExpectedSupports

ErrorRates = [0; 1; 0; 0;;]
MeasurementErrorRates = [0; 1;;]
E = sample_errors(5,ErrorRates,MeasurementErrorRates)
@test E == [1 1 1 1 1; 0 0 0 0 0; 1 1 1 1 1]
Chan = Channel_SingleQubit(expand_errorrates([0,1.0,0,0],2), expand_errorrates([0,1.0],3))
@test sample_errors(2,Chan) == [1 1;1 1;0 0;0 0;1 1;1 1; 1 1]
Chan = Channel_SingleQubit(expand_errorrates([0,0,1.0,0],2), expand_errorrates([1.0,0],3))
@test sample_errors(2,Chan) == [0 0;0 0;1 1;1 1;0 0;0 0; 0 0]

end

@testset "BitlabeledVector" begin
    a = BitlabeledVector([1.0,2.0,3.0,4.0],2)
    @test a[BitVector([0,0])] == 1.0
    @test a[Int[]] == 1.0
    @test a[BitVector([1,0])] == 2.0
    @test a[[1]] == 2.0
    @test a[BitVector([0,1])] == 3.0
    @test a[[2]] == 3.0
    @test a[BitVector([1,1])] == 4.0
    @test a[[1,2]] == 4.0
    @test a[BitVector([])] == 1.0
    
    a[BitVector([1,1])] = 5.0
    @test a[BitVector([1,1])] == 5.0
    
    b = BitlabeledVector([1.0], 0)
    @test b[BitVector([])] == 1.0
    @test b[BitVector([0])] == 1.0
end
@testset "FactorGraphs" begin
    Supports = [[1,2],[2,3]]
    Values = [[1.0,2.0,3.0,4.0], [5.0,6.0,7.0,8.0]]
    F = FactorGraph(3,Supports,Values)
    F_c = canonical_factorization(F)
    @test evaluate_prodFG(F,BitVector([0,0,0])) == 5.0
    @test evaluate_prodFG(F,BitVector([0,1,0])) == 18.0
    @test evaluate_prodFG(F,BitVector([0,0,1])) == 7.0
    @test evaluate_prodFG(F,BitVector([1,1,0])) == 24.0
    @test evaluate_prodFG(F_c,BitVector([0,0,0])) == 5.0
    @test evaluate_prodFG(F_c,BitVector([0,1,0])) == 18.0
    @test evaluate_prodFG(F_c,BitVector([0,0,1])) == 7.0
    @test evaluate_prodFG(F_c,BitVector([1,1,0])) == 24.0
end