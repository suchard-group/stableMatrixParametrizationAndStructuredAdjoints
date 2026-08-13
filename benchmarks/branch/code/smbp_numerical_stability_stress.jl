#!/usr/bin/env julia

# Numerical-stability stress tests for the SMBP block kernels.
#
# These are not timing benchmarks. They target regimes where reviewers are
# likely to worry about numerical behavior: repeated-root boundaries, slow mean
# reversion, cancellation in branch covariances, ill-conditioned bases, and
# stressed adjoint/gradient calculations.

using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "schur_vs_smbp_exp_adjoint.jl"))
include(joinpath(@__DIR__, "smbp_lyapunov_plan.jl"))

const STABILITY_K = 8
const STABILITY_PRECISION = 256
const STABILITY_SEED = 91
const BOUNDARY_N_SEEDS = 16
const BOUNDARY_DISTANCES = 10.0 .^ collect(-16:0)
const SLOW_EPSILONS = 10.0 .^ collect(0:-1:-8)
const BRANCH_DT_VALUES = 10.0 .^ collect(-14:1)
const BASIS_COND_VALUES = 10.0 .^ collect(0:2:10)
const LYAP_PARAM_COND_VALUES = 10.0 .^ collect(0:2:10)
const LYAP_PARAM_SKEW_VALUES = 10.0 .^ collect(-6:2:6)
const LYAP_PARAM_EPSILONS = 10.0 .^ collect(0:-1:-8)
const LYAP_PARAM_DT = 0.2
const MATCHED_BOUNDARY_DISTANCES = 10.0 .^ collect(-16:0)
const MATCHED_BRANCH_DT_VALUES = 10.0 .^ collect(-14:1)
const MATCHED_BASIS_COND_VALUES = 10.0 .^ collect(0:2:10)
const MATCHED_SLOW_EPSILONS = 10.0 .^ collect(0:-1:-8)
const MATCHED_DT = 0.2

fmt_stability(x::Integer) = string(x)
fmt_stability(x::AbstractFloat) = isfinite(x) ? @sprintf("%.6g", x) : string(x)
fmt_stability(x::Bool) = string(x)
fmt_stability(x::AbstractString) = x

function write_rows(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(fmt_stability.(row), ","))
        end
    end
    println("wrote ", path)
end

function big_identity(K::Int)
    Ibig = zeros(BigFloat, K, K)
    for i in 1:K
        Ibig[i, i] = BigFloat(1)
    end
    return Ibig
end

function matrix_inf_norm_big(A::AbstractMatrix{BigFloat})
    value = BigFloat(0)
    for i in axes(A, 1)
        row_sum = BigFloat(0)
        for j in axes(A, 2)
            row_sum += abs(A[i, j])
        end
        value = max(value, row_sum)
    end
    return value
end

function big_frobenius_norm(A)
    total = BigFloat(0)
    for x in A
        total += x * x
    end
    return sqrt(total)
end

function big_matrix_exp(A::Matrix{BigFloat}; max_terms::Int = 600)
    n = size(A, 1)
    n == size(A, 2) || error("matrix exponential requires a square matrix")
    scale_norm = matrix_inf_norm_big(A)
    scale_power = max(0, ceil(Int, log2(Float64(max(scale_norm, BigFloat(1))))))
    B = A ./ BigFloat(2)^scale_power
    result = big_identity(n)
    term = big_identity(n)
    tolerance = eps(BigFloat) * BigFloat(64)

    for k in 1:max_terms
        term = (term * B) ./ BigFloat(k)
        result .+= term
        if matrix_inf_norm_big(term) <= tolerance * max(matrix_inf_norm_big(result), BigFloat(1))
            break
        end
        k == max_terms && error("BigFloat matrix exponential Taylor series did not converge")
    end

    for _ in 1:scale_power
        result = result * result
    end
    return result
end

function big_frechet_exp(X::Matrix{BigFloat}, E::Matrix{BigFloat})
    n = size(X, 1)
    M = zeros(BigFloat, 2n, 2n)
    M[1:n, 1:n] .= X
    M[1:n, n + 1:2n] .= E
    M[n + 1:2n, n + 1:2n] .= X
    EM = big_matrix_exp(M)
    return Matrix(EM[1:n, n + 1:2n])
end

function big_exp_equal_diag_block(a::BigFloat, u::BigFloat, l::BigFloat)
    product = u * l
    if abs(product) < big"1.0e-90"
        scale = exp(a)
        return scale, scale
    elseif product < 0
        root = sqrt(-product)
        scale = exp(a)
        return scale * cos(root), scale * sin(root) / root
    else
        root = sqrt(product)
        scale = exp(a)
        return scale * cosh(root), scale * sinh(root) / root
    end
end

function block_matrix(diag::Float64, upper::Float64, lower::Float64)
    return [diag upper; lower diag]
end

function big_block_matrix(diag::Float64, upper::Float64, lower::Float64)
    return BigFloat[diag upper; lower diag]
end

function big_block_exp(diag::Float64, upper::Float64, lower::Float64, dt::Float64)
    a = -BigFloat(dt) * BigFloat(diag)
    u = -BigFloat(dt) * BigFloat(upper)
    l = -BigFloat(dt) * BigFloat(lower)
    c, s = big_exp_equal_diag_block(a, u, l)
    return BigFloat[c s * u; s * l c]
end

function float_block_exp(diag::Float64, upper::Float64, lower::Float64, dt::Float64)
    D = block_matrix(diag, upper, lower)
    I2 = Matrix(I, 2, 2)
    inst = SMBPInstance(2, I2, I2, D, [diag], [upper], [lower], D, zeros(2, 2))
    E = zeros(2, 2)
    smbp_block_exp!(E, inst, dt)
    return E
end

function big_solve_lyapunov(A::Matrix{BigFloat}, H::Matrix{BigFloat})
    K = size(A, 1)
    L = kron(big_identity(K), A) + kron(A, big_identity(K))
    x = L \ vec(H)
    return reshape(x, K, K)
end

function float_solve_lyapunov(diag::Float64, upper::Float64, lower::Float64,
                              H::Matrix{Float64})
    plan = SMBPLyapunovEqualDiagonalPlan([diag], [upper], [lower])
    out = zeros(2, 2)
    apply_smbp_lyapunov_symmetric!(out, plan, H)
    return out
end

function float_solve_lyapunov_block(diag::Vector{Float64}, upper::Vector{Float64},
                                    lower::Vector{Float64}, H::Matrix{Float64})
    plan = SMBPLyapunovEqualDiagonalPlan(diag, upper, lower)
    out = zeros(2length(diag), 2length(diag))
    apply_smbp_lyapunov_symmetric!(out, plan, H)
    return out
end

function float_exp_adjoint(diag::Float64, upper::Float64, lower::Float64,
                           Fbar::Matrix{Float64}, dt::Float64)
    D = block_matrix(diag, upper, lower)
    I2 = Matrix(I, 2, 2)
    inst = SMBPInstance(2, I2, I2, D, [diag], [upper], [lower], D, Fbar)
    Dbar = zeros(2, 2)
    try
        smbp_block_exp_adjoint!(Dbar, Fbar, inst, dt)
        return Dbar, true
    catch
        fill!(Dbar, NaN)
        return Dbar, false
    end
end

function big_exp_adjoint(diag::Float64, upper::Float64, lower::Float64,
                         Fbar::Matrix{BigFloat}, dt::Float64)
    D = big_block_matrix(diag, upper, lower)
    X = -BigFloat(dt) .* D
    return -BigFloat(dt) .* big_frechet_exp(Matrix(transpose(X)), Fbar)
end

function float_lyapunov_adjoint(diag::Float64, upper::Float64, lower::Float64,
                                seed::Matrix{Float64})
    return float_solve_lyapunov(diag, lower, upper, seed)
end

function big_lyapunov_adjoint(diag::Float64, upper::Float64, lower::Float64,
                              seed::Matrix{BigFloat})
    D = big_block_matrix(diag, upper, lower)
    return big_solve_lyapunov(Matrix(transpose(D)), seed)
end

function relerr_float_big(X::Matrix{Float64}, Y::Matrix{BigFloat})
    Xbig = BigFloat.(X)
    denom = max(big_frobenius_norm(Xbig), big_frobenius_norm(Y), eps(BigFloat))
    return Float64(big_frobenius_norm(Xbig .- Y) / denom)
end

function relerr_big(X::Matrix{BigFloat}, Y::Matrix{BigFloat})
    denom = max(big_frobenius_norm(X), big_frobenius_norm(Y), eps(BigFloat))
    return Float64(big_frobenius_norm(X .- Y) / denom)
end

function abs_fro_error_float_big(X::Matrix{Float64}, Y::Matrix{BigFloat})
    return Float64(big_frobenius_norm(BigFloat.(X) .- Y))
end

function log10_big(x::BigFloat)
    x == 0 && return -Inf
    return Float64(log10(abs(x)))
end

function cholesky_success(A::Matrix{Float64})
    S = Symmetric(0.5 .* (A .+ A'))
    try
        cholesky(S)
        return true
    catch
        return false
    end
end

function min_eig_symmetric(A::Matrix{Float64})
    return eigmin(Symmetric(0.5 .* (A .+ A')))
end

function symmetry_error(A::Matrix{Float64})
    return norm(A - A') / max(norm(A), eps(Float64))
end

function random_spd(rng::AbstractRNG, K::Int; ridge::Float64 = 0.5)
    G = randn(rng, K, K)
    return Symmetric((G' * G) / K + ridge * I) |> Matrix
end

function random_spd_with_condition(rng::AbstractRNG, K::Int,
                                   condition_number::Float64)
    U = random_orthogonal(rng, K)
    values = 10.0 .^ collect(range(0.0, -log10(condition_number), length = K))
    return Symmetric(U * Diagonal(values) * U') |> Matrix
end

function random_symmetric(rng::AbstractRNG, K::Int)
    G = randn(rng, K, K)
    return 0.5 .* (G .+ G')
end

function random_skew_symmetric(rng::AbstractRNG, K::Int)
    G = randn(rng, K, K)
    return 0.5 .* (G .- G')
end

function random_conditioned_basis(rng::AbstractRNG, K::Int, condition_number::Float64)
    U = random_orthogonal(rng, K)
    V = random_orthogonal(rng, K)
    exponents = range(0.0, -log10(condition_number), length = K)
    S = Diagonal(10.0 .^ collect(exponents))
    return U * S * V'
end

function branch_covariance_float(diag::Float64, upper::Float64, lower::Float64,
                                 V::Matrix{Float64}, dt::Float64)
    Sigma = float_solve_lyapunov(diag, upper, lower, V)
    F = float_block_exp(diag, upper, lower, dt)
    Q = Sigma - F * Sigma * F'
    return Sigma, F, Q
end

function branch_covariance_big(diag::Float64, upper::Float64, lower::Float64,
                               V::Matrix{BigFloat}, dt::Float64)
    D = big_block_matrix(diag, upper, lower)
    Sigma = big_solve_lyapunov(D, V)
    F = big_block_exp(diag, upper, lower, dt)
    Q = Sigma - F * Sigma * transpose(F)
    return Sigma, F, Q
end

function branch_gradient_float(diag::Float64, upper::Float64, lower::Float64,
                               V::Matrix{Float64}, Qbar::Matrix{Float64},
                               dt::Float64)
    Sigma, F, Q = branch_covariance_float(diag, upper, lower, V, dt)
    Fbar = -(Qbar * F * Sigma' + Qbar' * F * Sigma)
    Sigma_bar = Qbar - F' * Qbar * F
    Dbar_exp, exp_ok = float_exp_adjoint(diag, upper, lower, Fbar, dt)
    Y = float_lyapunov_adjoint(diag, upper, lower, Sigma_bar)
    Dbar_lyap = -(Y * Sigma' + Y' * Sigma)
    return Dbar_exp + Dbar_lyap, Q, exp_ok
end

function branch_gradient_big(diag::Float64, upper::Float64, lower::Float64,
                             V::Matrix{BigFloat}, Qbar::Matrix{BigFloat},
                             dt::Float64)
    Sigma, F, Q = branch_covariance_big(diag, upper, lower, V, dt)
    Fbar = -(Qbar * F * transpose(Sigma) + transpose(Qbar) * F * Sigma)
    Sigma_bar = Qbar - transpose(F) * Qbar * F
    Dbar_exp = big_exp_adjoint(diag, upper, lower, Fbar, dt)
    Y = big_lyapunov_adjoint(diag, upper, lower, Sigma_bar)
    Dbar_lyap = -(Y * transpose(Sigma) + transpose(Y) * Sigma)
    return Dbar_exp + Dbar_lyap, Q
end

function block_matrix(diag::Vector{Float64}, upper::Vector{Float64},
                      lower::Vector{Float64})
    K = 2length(diag)
    D = zeros(K, K)
    for block in eachindex(diag)
        i = 2block - 1
        D[i, i] = diag[block]
        D[i + 1, i + 1] = diag[block]
        D[i, i + 1] = upper[block]
        D[i + 1, i] = lower[block]
    end
    return D
end

function branch_gradient_float_block(diag::Vector{Float64}, upper::Vector{Float64},
                                     lower::Vector{Float64}, V::Matrix{Float64},
                                     Qbar::Matrix{Float64}, dt::Float64)
    K = 2length(diag)
    D = block_matrix(diag, upper, lower)
    Sigma = float_solve_lyapunov_block(diag, upper, lower, V)
    F = zeros(K, K)
    IK = Matrix(I, K, K)
    inst = SMBPInstance(K, IK, IK, D, diag, upper, lower, D, Qbar)
    smbp_block_exp!(F, inst, dt)
    Q = Sigma - F * Sigma * F'

    Fbar = -(Qbar * F * Sigma' + Qbar' * F * Sigma)
    Sigma_bar = Qbar - F' * Qbar * F
    Dbar_exp = zeros(K, K)
    smbp_block_exp_adjoint!(Dbar_exp, Fbar, inst, dt)
    Y = float_solve_lyapunov_block(diag, lower, upper, Sigma_bar)
    Dbar_lyap = -(Y * Sigma' + Y' * Sigma)
    return Dbar_exp + Dbar_lyap, Q
end

function branch_gradient_big_block(D::Matrix{BigFloat}, V::Matrix{BigFloat},
                                   Qbar::Matrix{BigFloat}, dt::Float64)
    dt_big = BigFloat(dt)
    Sigma = big_solve_lyapunov(D, V)
    F = big_matrix_exp(-dt_big .* D)
    Q = Sigma - F * Sigma * transpose(F)

    Fbar = -(Qbar * F * transpose(Sigma) + transpose(Qbar) * F * Sigma)
    Sigma_bar = Qbar - transpose(F) * Qbar * F
    Dbar_exp = -dt_big .* big_frechet_exp(Matrix(transpose(-dt_big .* D)), Fbar)
    Y = big_solve_lyapunov(Matrix(transpose(D)), Sigma_bar)
    Dbar_lyap = -(Y * transpose(Sigma) + transpose(Y) * Sigma)
    return Dbar_exp + Dbar_lyap, Q
end

# Block-basis adjoint helpers used to test the exponential-adjoint and
# Lyapunov-adjoint maps directly (with an arbitrary seed), as opposed to
# branch_gradient_*_block above which only exposes their sum.
function float_exp_adjoint_block(diag::Vector{Float64}, upper::Vector{Float64},
                                 lower::Vector{Float64}, Fbar::Matrix{Float64},
                                 dt::Float64)
    K = 2length(diag)
    D = block_matrix(diag, upper, lower)
    IK = Matrix(I, K, K)
    inst = SMBPInstance(K, IK, IK, D, diag, upper, lower, D, Fbar)
    Dbar = zeros(K, K)
    try
        smbp_block_exp_adjoint!(Dbar, Fbar, inst, dt)
        return Dbar, true
    catch
        fill!(Dbar, NaN)
        return Dbar, false
    end
end

function big_exp_adjoint_block(D::Matrix{BigFloat}, Fbar::Matrix{BigFloat}, dt::Float64)
    dt_big = BigFloat(dt)
    return -dt_big .* big_frechet_exp(Matrix(transpose(-dt_big .* D)), Fbar)
end

function float_lyapunov_adjoint_block(diag::Vector{Float64}, upper::Vector{Float64},
                                      lower::Vector{Float64}, seed::Matrix{Float64})
    return float_solve_lyapunov_block(diag, lower, upper, seed)
end

function big_lyapunov_adjoint_block(D::Matrix{BigFloat}, seed::Matrix{BigFloat})
    return big_solve_lyapunov(Matrix(transpose(D)), seed)
end

function boundary_rows(rng::AbstractRNG, first_seed::Int, n_seeds::Int)
    dt = 0.7
    rows = Vector{Vector{Any}}()

    setprecision(STABILITY_PRECISION) do
        for draw in 0:(n_seeds - 1)
            current_seed = first_seed + draw
            Random.seed!(rng, current_seed)
            V = random_spd(rng, 2; ridge = 0.5)
            Fbar = randn(rng, 2, 2)
            lyap_seed = random_symmetric(rng, 2)
            Qbar_seed = random_symmetric(rng, 2)

            for side in ("real", "complex")
                sign = side == "real" ? 1.0 : -1.0
                for distance in BOUNDARY_DISTANCES
                    diag = 2.0
                    upper = 1.0
                    lower = sign * distance

                    E_float = float_block_exp(diag, upper, lower, dt)
                    E_big = big_block_exp(diag, upper, lower, dt)
                    Dbar_float, exp_adj_ok = float_exp_adjoint(diag, upper, lower, Fbar, dt)
                    Dbar_big = big_exp_adjoint(diag, upper, lower, BigFloat.(Fbar), dt)
                    Sigma_float = float_solve_lyapunov(diag, upper, lower, V)
                    Sigma_big = big_solve_lyapunov(
                        big_block_matrix(diag, upper, lower), BigFloat.(V))
                    Y_float = float_lyapunov_adjoint(diag, upper, lower, lyap_seed)
                    Y_big = big_lyapunov_adjoint(diag, upper, lower, BigFloat.(lyap_seed))
                    Dbar_branch_float, Q_float, branch_exp_ok =
                        branch_gradient_float(diag, upper, lower, V, Qbar_seed, dt)
                    Dbar_branch_big, Q_big =
                        branch_gradient_big(diag, upper, lower, BigFloat.(V),
                                            BigFloat.(Qbar_seed), dt)

                    push!(rows, Any[
                        current_seed,
                        side,
                        distance,
                        relerr_float_big(E_float, E_big),
                        exp_adj_ok ? relerr_float_big(Dbar_float, Dbar_big) : Inf,
                        relerr_float_big(Sigma_float, Sigma_big),
                        relerr_float_big(Y_float, Y_big),
                        exp_adj_ok,
                        relerr_float_big(Q_float, Q_big),
                        branch_exp_ok ? relerr_float_big(Dbar_branch_float, Dbar_branch_big) : Inf,
                    ])
                end
            end
        end
    end
    return rows
end

function slow_lyapunov_rows(rng::AbstractRNG)
    K = STABILITY_K
    smbp = make_instance(rng, K)
    V_block = random_spd(rng, K; ridge = 0.5)
    adjoint_seed = random_symmetric(rng, K)
    rows = Vector{Vector{Any}}()

    setprecision(STABILITY_PRECISION) do
        for epsilon in SLOW_EPSILONS
            diag = epsilon .* smbp.diag
            upper = epsilon .* smbp.upper
            lower = epsilon .* smbp.lower
            D = epsilon .* smbp.D
            A = smbp.R * D * smbp.R'
            V_dense = smbp.R * V_block * smbp.R'

            Sigma_big_block = big_solve_lyapunov(BigFloat.(D), BigFloat.(V_block))
            Rbig = BigFloat.(smbp.R)
            Sigma_big = Rbig * Sigma_big_block * transpose(Rbig)
            reference_log10 = log10_big(big_frobenius_norm(Sigma_big))

            try
                Sigma_block = float_solve_lyapunov_block(diag, upper, lower, V_block)
                Sigma_float = smbp.R * Sigma_block * smbp.R'
                residual = norm(A * Sigma_float + Sigma_float * A' - V_dense) / norm(V_dense)
                Y_float = float_solve_lyapunov_block(diag, lower, upper, adjoint_seed)
                Y_big = big_solve_lyapunov(
                    Matrix(transpose(BigFloat.(D))), BigFloat.(adjoint_seed))

                push!(rows, Any[
                    epsilon,
                    relerr_float_big(Sigma_float, Sigma_big),
                    residual,
                    relerr_float_big(Y_float, Y_big),
                    min_eig_symmetric(Sigma_float),
                    cholesky_success(Sigma_float),
                    reference_log10,
                ])
            catch
                push!(rows, Any[
                    epsilon,
                    Inf,
                    Inf,
                    Inf,
                    NaN,
                    false,
                    reference_log10,
                ])
            end
        end
    end
    return rows
end

function branch_covariance_rows(rng::AbstractRNG)
    K = STABILITY_K
    smbp = make_instance(rng, K)
    V_block = random_spd(rng, K; ridge = 0.5)
    Qbar_block = random_symmetric(rng, K)
    Sigma_block = float_solve_lyapunov_block(smbp.diag, smbp.upper, smbp.lower, V_block)
    rows = Vector{Vector{Any}}()

    setprecision(STABILITY_PRECISION) do
        Dbig = BigFloat.(smbp.D)
        Vbig = BigFloat.(V_block)
        Qbar_big = BigFloat.(Qbar_block)
        Sigma_big = big_solve_lyapunov(Dbig, Vbig)
        for dt in BRANCH_DT_VALUES
            E_float = zeros(K, K)
            smbp_block_exp!(E_float, smbp, dt)
            Q_float = Sigma_block - E_float * Sigma_block * E_float'

            E_big = zeros(BigFloat, K, K)
            for block in eachindex(smbp.diag)
                i = 2block - 1
                Eb = big_block_exp(smbp.diag[block], smbp.upper[block], smbp.lower[block], dt)
                E_big[i:i + 1, i:i + 1] .= Eb
            end
            Q_big = Sigma_big - E_big * Sigma_big * transpose(E_big)
            Dbar_branch_float, _ =
                branch_gradient_float_block(smbp.diag, smbp.upper, smbp.lower,
                                            V_block, Qbar_block, dt)
            Dbar_branch_big, _ =
                branch_gradient_big_block(Dbig, Vbig, Qbar_big, dt)

            push!(rows, Any[
                dt,
                relerr_float_big(Q_float, Q_big),
                abs_fro_error_float_big(Q_float, Q_big),
                relerr_float_big(Dbar_branch_float, Dbar_branch_big),
                abs_fro_error_float_big(Dbar_branch_float, Dbar_branch_big),
                symmetry_error(Q_float),
                min_eig_symmetric(Q_float),
                cholesky_success(Q_float),
                log10_big(big_frobenius_norm(Q_big)),
            ])
        end
    end
    return rows
end

function basis_conditioning_rows(rng::AbstractRNG)
    K = STABILITY_K
    smbp = make_instance(rng, K)
    V_block = random_spd(rng, K; ridge = 0.5)
    Qbar_dense = random_symmetric(rng, K)
    Fbar_dense = randn(rng, K, K)
    lyap_seed_dense = random_symmetric(rng, K)
    dt = 0.2
    rows = Vector{Vector{Any}}()

    setprecision(STABILITY_PRECISION) do
        E_float = zeros(K, K)
        smbp_block_exp!(E_float, smbp, dt)
        Sigma_block = float_solve_lyapunov_block(smbp.diag, smbp.upper, smbp.lower, V_block)
        E_big = zeros(BigFloat, K, K)
        for block in eachindex(smbp.diag)
            i = 2block - 1
            Eb = big_block_exp(smbp.diag[block], smbp.upper[block], smbp.lower[block], dt)
            E_big[i:i + 1, i:i + 1] .= Eb
        end
        Sigma_big_block = big_solve_lyapunov(BigFloat.(smbp.D), BigFloat.(V_block))
        Q_block = Sigma_block - E_float * Sigma_block * E_float'
        Q_big_block = Sigma_big_block - E_big * Sigma_big_block * transpose(E_big)
        Vbig = BigFloat.(V_block)

        for condition_number in BASIS_COND_VALUES
            R = condition_number == 1.0 ?
                random_orthogonal(rng, K) :
                random_conditioned_basis(rng, K, condition_number)
            invR = inv(R)
            F_float = R * E_float * invR
            Sigma_float = R * Sigma_block * R'
            Q_float = R * Q_block * R'

            Rbig = BigFloat.(R)
            invRbig = Rbig \ big_identity(K)
            F_big = Rbig * E_big * invRbig
            Sigma_big = Rbig * Sigma_big_block * transpose(Rbig)
            Q_big = Rbig * Q_big_block * transpose(Rbig)
            Qbar_block = R' * Qbar_dense * R
            Qbar_big = transpose(Rbig) * BigFloat.(Qbar_dense) * Rbig
            Dbar_branch_float, _ =
                branch_gradient_float_block(smbp.diag, smbp.upper, smbp.lower,
                                            V_block, Qbar_block, dt)
            Dbar_branch_big, _ =
                branch_gradient_big_block(BigFloat.(smbp.D), Vbig, Qbar_big, dt)

            Fbar_block = R' * Fbar_dense * invR'
            Fbar_big = transpose(Rbig) * BigFloat.(Fbar_dense) * transpose(invRbig)
            Dbar_exp_float, exp_adj_ok =
                float_exp_adjoint_block(smbp.diag, smbp.upper, smbp.lower, Fbar_block, dt)
            Dbar_exp_big = big_exp_adjoint_block(BigFloat.(smbp.D), Fbar_big, dt)

            lyap_seed_block = R' * lyap_seed_dense * R
            lyap_seed_big = transpose(Rbig) * BigFloat.(lyap_seed_dense) * Rbig
            Y_float = float_lyapunov_adjoint_block(smbp.diag, smbp.upper, smbp.lower,
                                                    lyap_seed_block)
            Y_big = big_lyapunov_adjoint_block(BigFloat.(smbp.D), lyap_seed_big)

            push!(rows, Any[
                condition_number,
                cond(R),
                relerr_float_big(F_float, F_big),
                exp_adj_ok ? relerr_float_big(Dbar_exp_float, Dbar_exp_big) : Inf,
                relerr_float_big(Sigma_float, Sigma_big),
                relerr_float_big(Y_float, Y_big),
                relerr_float_big(Q_float, Q_big),
                relerr_float_big(Dbar_branch_float, Dbar_branch_big),
                min_eig_symmetric(Q_float),
                cholesky_success(Q_float),
            ])
        end
    end
    return rows
end

function lyapunov_param_selection(Q::AbstractMatrix, Sigma::AbstractMatrix,
                                  M::AbstractMatrix)
    return (0.5 .* Q .+ M) / Sigma
end

function lyapunov_param_metrics(B::Matrix{Float64}, Q::Matrix{Float64},
                                Sigma::Matrix{Float64}, dt::Float64)
    F = exp(-dt .* B)
    branch_cov = Sigma - F * Sigma * F'
    residual = norm(B * Sigma + Sigma * B' - Q) / max(norm(Q), eps(Float64))

    setprecision(STABILITY_PRECISION) do
        Bbig = BigFloat.(B)
        Sigmabig = BigFloat.(Sigma)
        Fbig = big_matrix_exp(-BigFloat(dt) .* Bbig)
        branch_big = Sigmabig - Fbig * Sigmabig * transpose(Fbig)
        return (
            exp_relerr = relerr_float_big(F, Fbig),
            branch_covariance_relerr = relerr_float_big(branch_cov, branch_big),
            branch_covariance_abs_fro_error =
                abs_fro_error_float_big(branch_cov, branch_big),
            lyapunov_residual_relerr = residual,
            min_stationary_eigenvalue = min_eig_symmetric(Sigma),
            min_branch_eigenvalue = min_eig_symmetric(branch_cov),
            branch_cholesky_success = cholesky_success(branch_cov),
            selection_condition = cond(B),
            spectral_abscissa = minimum(real.(eigvals(B))),
        )
    end
end

function lyapunov_param_push!(rows, stress::AbstractString,
                              x_value::Float64, B::Matrix{Float64},
                              Q::Matrix{Float64}, Sigma::Matrix{Float64},
                              dt::Float64)
    metrics = lyapunov_param_metrics(B, Q, Sigma, dt)
    push!(rows, Any[
        stress,
        x_value,
        metrics.exp_relerr,
        metrics.branch_covariance_relerr,
        metrics.branch_covariance_abs_fro_error,
        metrics.lyapunov_residual_relerr,
        metrics.min_stationary_eigenvalue,
        metrics.min_branch_eigenvalue,
        metrics.branch_cholesky_success,
        metrics.selection_condition,
        metrics.spectral_abscissa,
    ])
end

function lyapunov_param_stability_rows(rng::AbstractRNG)
    K = STABILITY_K
    rows = Vector{Vector{Any}}()
    base_sigma = random_spd_with_condition(rng, K, 1.0e2)
    base_q = random_spd_with_condition(rng, K, 1.0e2)
    base_skew = random_skew_symmetric(rng, K)
    base_skew ./= max(norm(base_skew), eps(Float64))

    for condition_number in LYAP_PARAM_COND_VALUES
        Sigma = random_spd_with_condition(rng, K, condition_number)
        Q = base_q
        M = 0.25 * norm(Q) * base_skew
        B = lyapunov_param_selection(Q, Sigma, M)
        lyapunov_param_push!(
            rows, "stationary-covariance-conditioning",
            Float64(condition_number), B, Q, Sigma, LYAP_PARAM_DT)
    end

    for condition_number in LYAP_PARAM_COND_VALUES
        Sigma = base_sigma
        Q = random_spd_with_condition(rng, K, condition_number)
        M = 0.25 * norm(Q) * base_skew
        B = lyapunov_param_selection(Q, Sigma, M)
        lyapunov_param_push!(
            rows, "diffusion-conditioning",
            Float64(condition_number), B, Q, Sigma, LYAP_PARAM_DT)
    end

    for skew_scale in LYAP_PARAM_SKEW_VALUES
        Sigma = base_sigma
        Q = base_q
        M = skew_scale * norm(Q) * base_skew
        B = lyapunov_param_selection(Q, Sigma, M)
        lyapunov_param_push!(
            rows, "skew-strength",
            Float64(skew_scale), B, Q, Sigma, LYAP_PARAM_DT)
    end

    for epsilon in LYAP_PARAM_EPSILONS
        Sigma = base_sigma
        Q = epsilon .* base_q
        M = epsilon .* 0.25 .* norm(base_q) .* base_skew
        B = lyapunov_param_selection(Q, Sigma, M)
        lyapunov_param_push!(
            rows, "slow-mean-reversion",
            Float64(epsilon), B, Q, Sigma, LYAP_PARAM_DT)
    end

    return rows
end

function symmetrize_matrix(A::Matrix{Float64})
    return Matrix(Symmetric(0.5 .* (A .+ A')))
end

function smbp_instance_from_parts(R::Matrix{Float64}, Rinv::Matrix{Float64},
                                  diag::Vector{Float64},
                                  upper::Vector{Float64},
                                  lower::Vector{Float64})
    D = block_matrix(diag, upper, lower)
    A = R * D * Rinv
    return SMBPInstance(
        size(D, 1),
        R,
        Rinv,
        D,
        copy(diag),
        copy(upper),
        copy(lower),
        A,
        zeros(size(D)),
    )
end

function big_block_exp_matrix(diag::Vector{Float64}, upper::Vector{Float64},
                              lower::Vector{Float64}, dt::Float64)
    K = 2 * length(diag)
    E = zeros(BigFloat, K, K)
    for block in eachindex(diag)
        i = 2 * block - 1
        E[i:i + 1, i:i + 1] .=
            big_block_exp(diag[block], upper[block], lower[block], dt)
    end
    return E
end

function matched_param_method_metrics(inst::SMBPInstance,
                                      V_block::Matrix{Float64},
                                      dt::Float64)
    K = inst.K
    E_block = zeros(K, K)
    smbp_block_exp!(E_block, inst, dt)
    Sigma_block = float_solve_lyapunov_block(
        inst.diag, inst.upper, inst.lower, V_block)
    branch_block = Sigma_block - E_block * Sigma_block * E_block'

    F_smbp = zeros(K, K)
    tmp = zeros(K, K)
    materialize_exp_from_blocks_inverse!(
        F_smbp, tmp, inst.R, inst.Rinv, inst, dt)
    Sigma_dense = symmetrize_matrix(inst.R * Sigma_block * inst.R')
    branch_smbp = symmetrize_matrix(inst.R * branch_block * inst.R')
    Q_dense = symmetrize_matrix(inst.R * V_block * inst.R')
    M_dense = 0.5 .* (inst.A * Sigma_dense - Sigma_dense * inst.A')
    M_dense = 0.5 .* (M_dense .- M_dense')

    B_lyap = lyapunov_param_selection(Q_dense, Sigma_dense, M_dense)
    F_lyap = exp(-dt .* B_lyap)
    branch_lyap = symmetrize_matrix(Sigma_dense - F_lyap * Sigma_dense * F_lyap')

    setprecision(STABILITY_PRECISION) do
        Rbig = BigFloat.(inst.R)
        Rinvbig = Rbig \ big_identity(K)
        Dbig = BigFloat.(inst.D)
        Vbig = BigFloat.(V_block)
        Ebig = big_block_exp_matrix(inst.diag, inst.upper, inst.lower, dt)
        Bbig = Rbig * Dbig * Rinvbig
        Fbig = Rbig * Ebig * Rinvbig
        Sigma_block_big = big_solve_lyapunov(Dbig, Vbig)
        Sigma_big = Rbig * Sigma_block_big * transpose(Rbig)
        branch_big = Sigma_big - Fbig * Sigma_big * transpose(Fbig)

        method_inputs = (
            ("SSBP", inst.A, F_smbp, Sigma_dense, branch_smbp),
            ("Lyapunov parametrization", B_lyap, F_lyap, Sigma_dense, branch_lyap),
        )
        metrics = Vector{NamedTuple}()
        for (method, B_float, F_float, Sigma_float, branch_float) in method_inputs
            residual = norm(B_float * Sigma_dense + Sigma_dense * B_float' - Q_dense) /
                max(norm(Q_dense), eps(Float64))
            push!(metrics, (
                method = method,
                drift_relerr = relerr_float_big(B_float, Bbig),
                exp_relerr = relerr_float_big(F_float, Fbig),
                stationary_covariance_relerr =
                    relerr_float_big(Sigma_float, Sigma_big),
                branch_covariance_relerr =
                    relerr_float_big(branch_float, branch_big),
                branch_covariance_abs_fro_error =
                    abs_fro_error_float_big(branch_float, branch_big),
                lyapunov_residual_relerr = residual,
                min_stationary_eigenvalue = min_eig_symmetric(Sigma_float),
                min_branch_eigenvalue = min_eig_symmetric(branch_float),
                branch_cholesky_success = cholesky_success(branch_float),
                selection_condition = cond(B_float),
                spectral_abscissa = minimum(real.(eigvals(B_float))),
            ))
        end
        return metrics
    end
end

function matched_param_push!(rows, stress::AbstractString, group::AbstractString,
                             x_value::Float64, inst::SMBPInstance,
                             V_block::Matrix{Float64}, dt::Float64)
    try
        for metrics in matched_param_method_metrics(inst, V_block, dt)
            push!(rows, Any[
                stress,
                group,
                x_value,
                metrics.method,
                dt,
                metrics.drift_relerr,
                metrics.exp_relerr,
                metrics.stationary_covariance_relerr,
                metrics.branch_covariance_relerr,
                metrics.branch_covariance_abs_fro_error,
                metrics.lyapunov_residual_relerr,
                metrics.min_stationary_eigenvalue,
                metrics.min_branch_eigenvalue,
                metrics.branch_cholesky_success,
                metrics.selection_condition,
                metrics.spectral_abscissa,
            ])
        end
    catch
        for method in ("SSBP", "Lyapunov parametrization")
            push!(rows, Any[
                stress,
                group,
                x_value,
                method,
                dt,
                Inf,
                Inf,
                Inf,
                Inf,
                Inf,
                Inf,
                NaN,
                NaN,
                false,
                Inf,
                NaN,
            ])
        end
    end
end

function matched_param_stability_rows(rng::AbstractRNG)
    K = STABILITY_K
    rows = Vector{Vector{Any}}()

    boundary_base = make_instance(rng, K)
    boundary_V = random_spd(rng, K; ridge = 0.5)
    for side in ("real", "complex")
        sign = side == "real" ? 1.0 : -1.0
        for distance in MATCHED_BOUNDARY_DISTANCES
            diag = copy(boundary_base.diag)
            upper = copy(boundary_base.upper)
            lower = copy(boundary_base.lower)
            diag[1] = 2.0
            upper[1] = 1.0
            lower[1] = sign * distance
            inst = smbp_instance_from_parts(
                boundary_base.R, boundary_base.Rinv, diag, upper, lower)
            matched_param_push!(
                rows, "repeated-root-boundary", side, Float64(distance),
                inst, boundary_V, MATCHED_DT)
        end
    end

    branch_inst = make_instance(rng, K)
    branch_V = random_spd(rng, K; ridge = 0.5)
    for dt in MATCHED_BRANCH_DT_VALUES
        matched_param_push!(
            rows, "branch-length", "matched", Float64(dt),
            branch_inst, branch_V, Float64(dt))
    end

    basis_base = make_instance(rng, K)
    basis_V = random_spd(rng, K; ridge = 0.5)
    for condition_number in MATCHED_BASIS_COND_VALUES
        R = condition_number == 1.0 ?
            random_orthogonal(rng, K) :
            random_conditioned_basis(rng, K, condition_number)
        Rinv = inv(R)
        inst = smbp_instance_from_parts(
            R, Rinv,
            basis_base.diag, basis_base.upper, basis_base.lower)
        matched_param_push!(
            rows, "basis-conditioning", "matched",
            Float64(condition_number), inst, basis_V, MATCHED_DT)
    end

    slow_base = make_instance(rng, K)
    slow_V = random_spd(rng, K; ridge = 0.5)
    for epsilon in MATCHED_SLOW_EPSILONS
        inst = smbp_instance_from_parts(
            slow_base.R,
            slow_base.Rinv,
            epsilon .* slow_base.diag,
            epsilon .* slow_base.upper,
            epsilon .* slow_base.lower)
        matched_param_push!(
            rows, "slow-mean-reversion", "matched",
            Float64(epsilon), inst, slow_V, MATCHED_DT)
    end

    return rows
end

function gradient_case_rows(rng::AbstractRNG)
    V = [1.2 0.2; 0.2 0.8]
    Fbar = [0.3 -0.7; 0.5 0.2]
    lyap_seed = [0.4 -0.1; -0.1 0.7]
    Qbar_base = [0.6 -0.3; -0.3 0.5]
    cases = [
        ("well-conditioned", 1.0, 0.7, -0.8, 0.2, 1.0),
        ("near-boundary", 1.0, 1.0, -1.0e-12, 0.2, 1.0),
        ("slow-mean-reversion", 1.0e-6, 0.7e-6, -0.8e-6, 0.2, 1.0),
        ("ill-conditioned-basis", 1.0, 0.7, -0.8, 0.2, 1.0e8),
        ("small-branch-length", 1.0, 0.7, -0.8, 1.0e-6, 1.0),
    ]
    rows = Vector{Vector{Any}}()

    setprecision(STABILITY_PRECISION) do
        for (name, diag, upper, lower, dt, condition_number) in cases
            R = condition_number == 1.0 ?
                Matrix(I, 2, 2) :
                random_conditioned_basis(rng, 2, condition_number)
            invR = inv(R)
            Rbig = BigFloat.(R)
            invRbig = Rbig \ big_identity(2)

            Fbar_block = R' * Fbar * invR'
            Fbar_big = transpose(Rbig) * BigFloat.(Fbar) * transpose(invRbig)
            try
                Dbar_exp_float, exp_adj_ok =
                    float_exp_adjoint(diag, upper, lower, Fbar_block, dt)
                Dbar_exp_big = big_exp_adjoint(diag, upper, lower, Fbar_big, dt)

                lyap_seed_block = R' * lyap_seed * R
                lyap_seed_big = transpose(Rbig) * BigFloat.(lyap_seed) * Rbig
                Y_float = float_lyapunov_adjoint(diag, upper, lower, lyap_seed_block)
                Y_big = big_lyapunov_adjoint(diag, upper, lower, lyap_seed_big)

                V_block = R \ V / R'
                Qbar_block = R' * Qbar_base * R
                V_big = invRbig * BigFloat.(V) * transpose(invRbig)
                Qbar_big = transpose(Rbig) * BigFloat.(Qbar_base) * Rbig
                Dbar_branch_float, Q_float, branch_exp_ok =
                    branch_gradient_float(diag, upper, lower, V_block, Qbar_block, dt)
                Dbar_branch_big, Q_big =
                    branch_gradient_big(diag, upper, lower, V_big, Qbar_big, dt)

                push!(rows, Any[
                    name,
                    condition_number,
                    exp_adj_ok ? relerr_float_big(Dbar_exp_float, Dbar_exp_big) : Inf,
                    relerr_float_big(Y_float, Y_big),
                    branch_exp_ok ? relerr_float_big(Dbar_branch_float, Dbar_branch_big) : Inf,
                    cholesky_success(Q_float),
                ])
            catch
                push!(rows, Any[
                    name,
                    condition_number,
                    Inf,
                    Inf,
                    Inf,
                    false,
                ])
            end
        end
    end
    return rows
end

function main()
    configure_blas_threads()
    seed = env_int("SEED", STABILITY_SEED)
    boundary_n_seeds = env_int("BOUNDARY_N_SEEDS", BOUNDARY_N_SEEDS)
    boundary_rng = MersenneTwister(seed)
    rng = MersenneTwister(seed)
    out_dir = get(ENV, "OUT_DIR", joinpath(@__DIR__, "..", "results"))
    isdir(out_dir) || mkpath(out_dir)

    write_rows(
        joinpath(out_dir, "smbp_boundary_stability_results.csv"),
        ["seed", "side", "boundary_distance", "exp_relerr", "exp_adjoint_relerr",
         "lyapunov_relerr", "lyapunov_adjoint_relerr", "exp_adjoint_success",
         "branch_covariance_relerr", "branch_gradient_relerr"],
        boundary_rows(boundary_rng, seed, boundary_n_seeds))

    write_rows(
        joinpath(out_dir, "smbp_slow_lyapunov_results.csv"),
        ["epsilon", "sigma_relerr", "residual_relerr", "lyapunov_adjoint_relerr",
         "min_eigenvalue", "cholesky_success", "reference_fro_log10"],
        slow_lyapunov_rows(rng))

    write_rows(
        joinpath(out_dir, "smbp_branch_covariance_stability_results.csv"),
        ["dt", "q_relerr", "q_abs_fro_error", "branch_gradient_relerr",
         "branch_gradient_abs_fro_error", "symmetry_error", "min_eigenvalue",
         "cholesky_success", "reference_fro_log10"],
        branch_covariance_rows(rng))

    write_rows(
        joinpath(out_dir, "smbp_basis_conditioning_results.csv"),
        ["target_condition", "observed_condition", "exp_relerr", "exp_adjoint_relerr",
         "lyapunov_relerr", "lyapunov_adjoint_relerr",
         "branch_covariance_relerr", "branch_gradient_relerr",
         "branch_min_eigenvalue", "branch_cholesky_success"],
        basis_conditioning_rows(rng))

    write_rows(
        joinpath(out_dir, "lyapunov_param_stability_results.csv"),
        ["stress", "x_value", "exp_relerr", "branch_covariance_relerr",
         "branch_covariance_abs_fro_error", "lyapunov_residual_relerr",
         "min_stationary_eigenvalue", "min_branch_eigenvalue",
         "branch_cholesky_success", "selection_condition", "spectral_abscissa"],
        lyapunov_param_stability_rows(rng))

    write_rows(
        joinpath(out_dir, "matched_param_stability_results.csv"),
        ["stress", "group", "x_value", "method", "dt", "drift_relerr",
         "exp_relerr", "stationary_covariance_relerr",
         "branch_covariance_relerr", "branch_covariance_abs_fro_error",
         "lyapunov_residual_relerr", "min_stationary_eigenvalue",
         "min_branch_eigenvalue", "branch_cholesky_success",
         "selection_condition", "spectral_abscissa"],
        matched_param_stability_rows(rng))

    write_rows(
        joinpath(out_dir, "smbp_gradient_stress_results.csv"),
        ["regime", "basis_condition", "exp_adjoint_relerr", "lyapunov_adjoint_relerr",
         "branch_gradient_relerr", "branch_cholesky_success"],
        gradient_case_rows(rng))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
