#!/usr/bin/env julia

# Dense-Schur-style matrix exponential and Frechet-adjoint benchmark against
# SMBP block kernels for equal-diagonal 2x2 blocks.
#
# The benchmark uses the same dense matrix A = R * D * R^{-1} for both paths.
# The main timing rows use orthogonal R, so R^{-1} = R':
#
#   dense:
#     F    = exp(-dt * A)
#     Abar = -dt * L_exp((-dt * A)')[Fbar]
#
#   SMBP:
#     E    = exp(-dt * D) by closed-form 2x2 block exponentials
#     F    = R * E * R'
#     Ebar = R' * Fbar * R
#     Dbar = -dt * L_exp((-dt * D)')[Ebar] by 2x2 block-pair kernels
#     Abar = R * Dbar * R'
#
# The block Frechet kernel mirrors the plan/coefficient idea in
# BlockDiagonalFrechetEqualDiagonalPlan.java: for each 2x2 block pair it caches
# a 16-entry coefficient matrix that applies
#
#   G -> alpha*G + beta*N_left*G + gamma*G*N_right + eta*N_left*G*N_right
#
# with alpha,beta,gamma,eta computed from divided differences of exp at the
# two equal-diagonal block eigenvalue pairs. The generated random blocks use
# imaginary roots away from zero; near-degenerate cases are intentionally
# outside this small timing probe.

using LinearAlgebra
using Printf
using Random
using Statistics

const DEFAULT_K_VALUES = [4, 8, 16, 32, 64]
const DEFAULT_DT_VALUES = [0.1]
const BENCHMARK_BLAS_THREADS = 1
const SMBP_FRECHET_FALLBACK_PRODUCT_TOL = 1.0e-2

struct SMBPInstance
    K::Int
    R::Matrix{Float64}
    Rinv::Matrix{Float64}
    D::Matrix{Float64}
    diag::Vector{Float64}
    upper::Vector{Float64}
    lower::Vector{Float64}
    A::Matrix{Float64}
    Fbar::Matrix{Float64}
end

mutable struct SMBPFrechetAdjointPlan
    K::Int
    block_count::Int
    dt::Float64
    coeffs::Vector{Float64}
end

SMBPFrechetAdjointPlan(K::Int) =
    SMBPFrechetAdjointPlan(K, div(K, 2), NaN, zeros(16 * div(K, 2) * div(K, 2)))

struct TimingStats
    median_ms::Float64
    mean_ms::Float64
    min_ms::Float64
    max_ms::Float64
    batch::Int
end

struct DenseSchurPlan
    K::Int
    Z::Matrix{Float64}
    T::Matrix{Float64}
end

function DenseSchurPlan(A::Matrix{Float64})
    factor = schur(A)
    return DenseSchurPlan(size(A, 1), Matrix(factor.Z), Matrix(factor.T))
end

mutable struct DenseComplexSchurParlettPlan
    K::Int
    Z::Matrix{ComplexF64}
    T::Matrix{ComplexF64}
    scaledT::Matrix{ComplexF64}
    ET::Matrix{ComplexF64}
    tmp::Matrix{ComplexF64}
    tmp2::Matrix{ComplexF64}
    out::Matrix{Float64}
end

function DenseComplexSchurParlettPlan(A::Matrix{Float64})
    factor = schur(complex.(A))
    K = size(A, 1)
    return DenseComplexSchurParlettPlan(
        K,
        Matrix(factor.Z),
        Matrix(factor.T),
        zeros(ComplexF64, K, K),
        zeros(ComplexF64, K, K),
        zeros(ComplexF64, K, K),
        zeros(ComplexF64, K, K),
        zeros(Float64, K, K),
    )
end

function parse_int_list(value::Union{Nothing,String}, default::Vector{Int})
    value === nothing && return default
    isempty(strip(value)) && return default
    return [parse(Int, strip(x)) for x in split(value, ",")]
end

function parse_float_list(value::Union{Nothing,String}, default::Vector{Float64})
    value === nothing && return default
    isempty(strip(value)) && return default
    return [parse(Float64, strip(x)) for x in split(value, ",")]
end

function env_int(name::String, default::Int)
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return parse(Int, value)
end

function env_float(name::String, default::Float64)
    value = get(ENV, name, "")
    isempty(strip(value)) && return default
    return parse(Float64, value)
end

function configure_blas_threads()
    BLAS.set_num_threads(BENCHMARK_BLAS_THREADS)
    return BLAS.get_num_threads()
end

function random_orthogonal(rng::AbstractRNG, K::Int)
    Q = Matrix(qr(randn(rng, K, K)).Q)
    return Q
end

function random_ldu_rotation_pair(rng::AbstractRNG, K::Int)
    L = Matrix{Float64}(I, K, K)
    U = Matrix{Float64}(I, K, K)
    scale = 0.06
    @inbounds for j in 1:(K - 1)
        for i in (j + 1):K
            L[i, j] = scale * randn(rng)
            U[j, i] = scale * randn(rng)
        end
    end
    diagonal = exp.(0.10 .* randn(rng, K))
    R = L * Diagonal(diagonal) * U
    Rinv = Matrix(inv(UnitUpperTriangular(U)) * Diagonal(1.0 ./ diagonal) *
                  inv(UnitLowerTriangular(L)))
    return R, Rinv
end

function make_instance(rng::AbstractRNG, K::Int)
    iseven(K) || error("K must be even; got K=$K")
    block_count = div(K, 2)
    diag = Vector{Float64}(undef, block_count)
    upper = Vector{Float64}(undef, block_count)
    lower = Vector{Float64}(undef, block_count)
    D = zeros(K, K)

    for b in 1:block_count
        i = 2b - 1
        rho = 0.30 + 1.40 * rand(rng)
        decay = 0.35 + 1.25 * rand(rng)
        up = 0.45 + 1.35 * rand(rng)
        lo = -(0.45 + 1.35 * rand(rng))

        diag[b] = rho * decay
        upper[b] = rho * up
        lower[b] = rho * lo

        D[i, i] = diag[b]
        D[i + 1, i + 1] = diag[b]
        D[i, i + 1] = upper[b]
        D[i + 1, i] = lower[b]
    end

    R = random_orthogonal(rng, K)
    Rinv = Matrix(R')
    A = R * D * Rinv
    Fbar = randn(rng, K, K)
    return SMBPInstance(K, R, Rinv, D, diag, upper, lower, A, Fbar)
end

function make_ldu_instance_like(rng::AbstractRNG, inst::SMBPInstance)
    R, Rinv = random_ldu_rotation_pair(rng, inst.K)
    A = R * inst.D * Rinv
    return SMBPInstance(
        inst.K,
        R,
        Rinv,
        copy(inst.D),
        copy(inst.diag),
        copy(inst.upper),
        copy(inst.lower),
        A,
        copy(inst.Fbar),
    )
end

function exp_equal_diag_block(a::Float64, u::Float64, l::Float64)
    product = u * l
    if abs(product) < 1.0e-24
        scale = exp(a)
        # exp(a) * (I + N) is enough at exactly zero root; the random benchmark
        # avoids this branch except for pathological manual inputs.
        return scale, scale
    elseif product < 0.0
        root = sqrt(-product)
        scale = exp(a)
        return scale * cos(root), scale * sin(root) / root
    else
        root = sqrt(product)
        scale = exp(a)
        return scale * cosh(root), scale * sinh(root) / root
    end
end

function smbp_block_exp!(E::Matrix{Float64}, inst::SMBPInstance, dt::Float64)
    fill!(E, 0.0)
    for b in eachindex(inst.diag)
        i = 2b - 1
        a = -dt * inst.diag[b]
        u = -dt * inst.upper[b]
        l = -dt * inst.lower[b]
        c, s = exp_equal_diag_block(a, u, l)
        E[i, i] = c
        E[i, i + 1] = s * u
        E[i + 1, i] = s * l
        E[i + 1, i + 1] = c
    end
    return E
end

function materialize_from_block!(F::Matrix{Float64}, tmp::Matrix{Float64},
                                 R::Matrix{Float64}, E::Matrix{Float64})
    K = size(R, 1)
    @inbounds for b in 1:2:K
        e00 = E[b, b]
        e01 = E[b, b + 1]
        e10 = E[b + 1, b]
        e11 = E[b + 1, b + 1]
        for i in 1:K
            r0 = R[i, b]
            r1 = R[i, b + 1]
            tmp[i, b] = r0 * e00 + r1 * e10
            tmp[i, b + 1] = r0 * e01 + r1 * e11
        end
    end
    mul!(F, tmp, R')
    return F
end

function materialize_exp_from_blocks!(F::Matrix{Float64}, tmp::Matrix{Float64},
                                      R::Matrix{Float64}, inst::SMBPInstance,
                                      dt::Float64)
    K = size(R, 1)
    @inbounds for block in eachindex(inst.diag)
        b = 2block - 1
        a = -dt * inst.diag[block]
        u = -dt * inst.upper[block]
        l = -dt * inst.lower[block]
        c, s = exp_equal_diag_block(a, u, l)
        e00 = c
        e01 = s * u
        e10 = s * l
        e11 = c
        for i in 1:K
            r0 = R[i, b]
            r1 = R[i, b + 1]
            tmp[i, b] = r0 * e00 + r1 * e10
            tmp[i, b + 1] = r0 * e01 + r1 * e11
        end
    end
    mul!(F, tmp, R')
    return F
end

function materialize_exp_from_blocks_inverse!(F::Matrix{Float64}, tmp::Matrix{Float64},
                                              R::Matrix{Float64}, Rinv::Matrix{Float64},
                                              inst::SMBPInstance, dt::Float64)
    K = size(R, 1)
    @inbounds for block in eachindex(inst.diag)
        b = 2block - 1
        a = -dt * inst.diag[block]
        u = -dt * inst.upper[block]
        l = -dt * inst.lower[block]
        c, s = exp_equal_diag_block(a, u, l)
        e00 = c
        e01 = s * u
        e10 = s * l
        e11 = c
        for i in 1:K
            r0 = R[i, b]
            r1 = R[i, b + 1]
            tmp[i, b] = r0 * e00 + r1 * e10
            tmp[i, b + 1] = r0 * e01 + r1 * e11
        end
    end
    mul!(F, tmp, Rinv)
    return F
end

function divided_difference_exp(left::ComplexF64, right::ComplexF64)
    if abs(left - right) < 1.0e-10
        return exp(left)
    end
    return (exp(left) - exp(right)) / (left - right)
end

function integral_moment0(c_re::Float64, c_im::Float64)
    radius2 = c_re * c_re + c_im * c_im
    if radius2 < 1.0e-14
        total_re = 0.0
        total_im = 0.0
        power_re = 1.0
        power_im = 0.0
        factorial = 1.0
        @inbounds for n in 0:29
            if n > 0
                factorial *= n
            end
            term_scale = 1.0 / (factorial * (n + 1.0))
            total_re += term_scale * power_re
            total_im += term_scale * power_im

            next_re = power_re * c_re - power_im * c_im
            next_im = power_re * c_im + power_im * c_re
            power_re = next_re
            power_im = next_im
        end
        return total_re, total_im
    end

    scale = exp(c_re)
    exp_re = scale * cos(c_im)
    exp_im = scale * sin(c_im)
    num_re = exp_re - 1.0
    num_im = exp_im
    return (
        (num_re * c_re + num_im * c_im) / radius2,
        (num_im * c_re - num_re * c_im) / radius2,
    )
end

function phi_integral(lambda_re::Float64, lambda_im::Float64,
                      mu_re::Float64, mu_im::Float64)
    moment_re, moment_im = integral_moment0(mu_re - lambda_re, mu_im - lambda_im)
    scale = exp(lambda_re)
    c = cos(lambda_im)
    s = sin(lambda_im)
    return (
        scale * (c * moment_re - s * moment_im),
        scale * (s * moment_re + c * moment_im),
    )
end

function equal_diag_coefficients_both_imag(a::Float64, b::Float64,
                                           r::Float64, q::Float64)
    same_re, same_im = phi_integral(a, r, b, q)
    opposite_re, opposite_im = phi_integral(a, r, b, -q)

    alpha = 0.5 * (same_re + opposite_re)
    beta = 0.5 * (same_im + opposite_im) / r
    gamma = 0.5 * (same_im - opposite_im) / q
    eta = -0.5 * (same_re - opposite_re) / (r * q)
    return alpha, beta, gamma, eta
end

function equal_diag_coefficients(a::Float64, u::Float64, l::Float64,
                                 b::Float64, v::Float64, w::Float64)
    left_product = u * l
    right_product = v * w
    if left_product < 0.0 && right_product < 0.0
        r = sqrt(-left_product)
        q = sqrt(-right_product)
        if r >= 1.0e-10 && q >= 1.0e-10
            return equal_diag_coefficients_both_imag(a, b, r, q)
        end
    end

    r = sqrt(complex(u * l))
    q = sqrt(complex(v * w))
    if abs(r) < 1.0e-10 || abs(q) < 1.0e-10
        error("Near-zero equal-diagonal root encountered; regenerate benchmark parameters.")
    end

    phi_pp = divided_difference_exp(complex(a) + r, complex(b) + q)
    phi_pm = divided_difference_exp(complex(a) + r, complex(b) - q)
    phi_mp = divided_difference_exp(complex(a) - r, complex(b) + q)
    phi_mm = divided_difference_exp(complex(a) - r, complex(b) - q)

    alpha = 0.25 * (phi_pp + phi_pm + phi_mp + phi_mm)
    beta = 0.25 * (phi_pp + phi_pm - phi_mp - phi_mm) / r
    gamma = 0.25 * (phi_pp - phi_pm + phi_mp - phi_mm) / q
    eta = 0.25 * (phi_pp - phi_pm - phi_mp + phi_mm) / (r * q)
    return real(alpha), real(beta), real(gamma), real(eta)
end

function fill_coefficient_matrix!(coeffs::Vector{Float64}, offset::Int,
                                  alpha::Float64, beta::Float64,
                                  gamma::Float64, eta::Float64,
                                  lu::Float64, ll::Float64,
                                  ru::Float64, rl::Float64,
                                  scale::Float64)
    @inbounds begin
        coeffs[offset] = scale * alpha
        coeffs[offset + 1] = scale * gamma * rl
        coeffs[offset + 2] = scale * beta * lu
        coeffs[offset + 3] = scale * eta * lu * rl

        coeffs[offset + 4] = scale * gamma * ru
        coeffs[offset + 5] = scale * alpha
        coeffs[offset + 6] = scale * eta * lu * ru
        coeffs[offset + 7] = scale * beta * lu

        coeffs[offset + 8] = scale * beta * ll
        coeffs[offset + 9] = scale * eta * ll * rl
        coeffs[offset + 10] = scale * alpha
        coeffs[offset + 11] = scale * gamma * rl

        coeffs[offset + 12] = scale * eta * ll * ru
        coeffs[offset + 13] = scale * beta * ll
        coeffs[offset + 14] = scale * gamma * ru
        coeffs[offset + 15] = scale * alpha
    end
    return coeffs
end

function dense_rectangular_frechet_exp(left::Matrix{Float64},
                                       right::Matrix{Float64},
                                       direction::Matrix{Float64})
    M = zeros(4, 4)
    @views begin
        M[1:2, 1:2] .= left
        M[1:2, 3:4] .= direction
        M[3:4, 3:4] .= right
    end
    EM = exp(M)
    return @views Matrix(EM[1:2, 3:4])
end

function fill_coefficient_matrix_dense_frechet!(coeffs::Vector{Float64},
                                                offset::Int,
                                                la::Float64, lu::Float64,
                                                ll::Float64,
                                                ra::Float64, ru::Float64,
                                                rl::Float64,
                                                scale::Float64)
    left = [la lu; ll la]
    right = [ra ru; rl ra]
    directions = (
        [1.0 0.0; 0.0 0.0],
        [0.0 1.0; 0.0 0.0],
        [0.0 0.0; 1.0 0.0],
        [0.0 0.0; 0.0 1.0],
    )

    @inbounds for col in 1:4
        Y = scale .* dense_rectangular_frechet_exp(left, right, directions[col])
        coeffs[offset + col - 1] = Y[1, 1]
        coeffs[offset + 4 + col - 1] = Y[1, 2]
        coeffs[offset + 8 + col - 1] = Y[2, 1]
        coeffs[offset + 12 + col - 1] = Y[2, 2]
    end
    return coeffs
end

function needs_dense_frechet_fallback(lu::Float64, ll::Float64,
                                      ru::Float64, rl::Float64)
    # The closed-form divided-difference coefficients lose accuracy when a
    # block root is small.  The fallback is local to this 2x2 block pair.
    return min(abs(lu * ll), abs(ru * rl)) < SMBP_FRECHET_FALLBACK_PRODUCT_TOL
end

function evaluate_plan!(plan::SMBPFrechetAdjointPlan,
                        inst::SMBPInstance,
                        dt::Float64)
    plan.dt = dt
    block_count = plan.block_count
    coeffs = plan.coeffs
    @inbounds for left in 1:block_count
        # The adjoint uses L_exp(X')[Ebar], so transpose each scaled block.
        la = -dt * inst.diag[left]
        lu = -dt * inst.lower[left]
        ll = -dt * inst.upper[left]

        for right in 1:block_count
            ra = -dt * inst.diag[right]
            ru = -dt * inst.lower[right]
            rl = -dt * inst.upper[right]
            offset = 16 * ((left - 1) * block_count + (right - 1)) + 1
            if needs_dense_frechet_fallback(lu, ll, ru, rl)
                fill_coefficient_matrix_dense_frechet!(
                    coeffs, offset, la, lu, ll, ra, ru, rl, -dt)
            else
                alpha, beta, gamma, eta =
                    equal_diag_coefficients(la, lu, ll, ra, ru, rl)
                fill_coefficient_matrix!(
                    coeffs, offset, alpha, beta, gamma, eta, lu, ll, ru, rl, -dt)
            end
        end
    end
    return plan
end

function apply_plan!(Dbar::Matrix{Float64}, Ebar::Matrix{Float64},
                     plan::SMBPFrechetAdjointPlan)
    block_count = plan.block_count
    K = plan.K
    coeffs = plan.coeffs
    @inbounds for left in 1:block_count
        li = 2left - 1
        for right in 1:block_count
            rj = 2right - 1
            e00 = Ebar[li, rj]
            e01 = Ebar[li, rj + 1]
            e10 = Ebar[li + 1, rj]
            e11 = Ebar[li + 1, rj + 1]
            c = 16 * ((left - 1) * block_count + (right - 1)) + 1

            Dbar[li, rj] =
                coeffs[c] * e00 + coeffs[c + 1] * e01 +
                coeffs[c + 2] * e10 + coeffs[c + 3] * e11
            Dbar[li, rj + 1] =
                coeffs[c + 4] * e00 + coeffs[c + 5] * e01 +
                coeffs[c + 6] * e10 + coeffs[c + 7] * e11
            Dbar[li + 1, rj] =
                coeffs[c + 8] * e00 + coeffs[c + 9] * e01 +
                coeffs[c + 10] * e10 + coeffs[c + 11] * e11
            Dbar[li + 1, rj + 1] =
                coeffs[c + 12] * e00 + coeffs[c + 13] * e01 +
                coeffs[c + 14] * e10 + coeffs[c + 15] * e11
        end
    end
    return Dbar
end

function smbp_block_exp_adjoint!(Dbar::Matrix{Float64}, Ebar::Matrix{Float64},
                                 inst::SMBPInstance, dt::Float64)
    plan = SMBPFrechetAdjointPlan(inst.K)
    evaluate_plan!(plan, inst, dt)
    return apply_plan!(Dbar, Ebar, plan)
end

function rotate_upstream_adjoint!(Ebar::Matrix{Float64}, tmp::Matrix{Float64},
                                  R::Matrix{Float64}, Fbar::Matrix{Float64})
    mul!(tmp, R', Fbar)
    mul!(Ebar, tmp, R)
    return Ebar
end

function rotate_downstream_adjoint!(Abar::Matrix{Float64}, tmp::Matrix{Float64},
                                    R::Matrix{Float64}, Dbar::Matrix{Float64})
    mul!(tmp, R, Dbar)
    mul!(Abar, tmp, R')
    return Abar
end

function rotate_upstream_adjoint_inverse!(Ebar::Matrix{Float64}, tmp::Matrix{Float64},
                                          R::Matrix{Float64}, Rinv::Matrix{Float64},
                                          Fbar::Matrix{Float64})
    mul!(tmp, R', Fbar)
    mul!(Ebar, tmp, Rinv')
    return Ebar
end

function rotate_downstream_adjoint_inverse!(Abar::Matrix{Float64}, tmp::Matrix{Float64},
                                            R::Matrix{Float64}, Rinv::Matrix{Float64},
                                            Dbar::Matrix{Float64})
    mul!(tmp, Rinv', Dbar)
    mul!(Abar, tmp, R')
    return Abar
end

function dense_frechet_exp(X::Matrix{Float64}, E::Matrix{Float64})
    n = size(X, 1)
    M = zeros(2n, 2n)
    @views begin
        M[1:n, 1:n] .= X
        M[1:n, n + 1:2n] .= E
        M[n + 1:2n, n + 1:2n] .= X
    end
    EM = exp(M)
    return @views Matrix(EM[1:n, n + 1:2n])
end

function dense_exp_adjoint(A::Matrix{Float64}, Fbar::Matrix{Float64}, dt::Float64)
    X = -dt * A
    return -dt * dense_frechet_exp(Matrix(X'), Fbar)
end

function schur_basis_exp(plan::DenseSchurPlan, dt::Float64)
    # The real Schur factor is quasi-triangular when complex eigenvalue pairs
    # are present. UpperTriangular would drop the 2x2 real Schur subdiagonal
    # entries; UpperHessenberg preserves them while still exposing the cached
    # Schur structure to Julia's matrix exponential.
    return exp(UpperHessenberg(-dt * plan.T))
end

function schur_exp(plan::DenseSchurPlan, dt::Float64)
    ET = schur_basis_exp(plan, dt)
    return plan.Z * ET * plan.Z'
end

function parlett_exp_triangular!(F::Matrix{ComplexF64},
                                 T::AbstractMatrix{ComplexF64})
    n = size(T, 1)
    size(T, 2) == n || error("T must be square")
    fill!(F, 0.0 + 0.0im)
    @inbounds for i in 1:n
        F[i, i] = exp(T[i, i])
    end

    @inbounds for j in 2:n
        for i in (j - 1):-1:1
            y = T[i, j] * (F[i, i] - F[j, j])
            for k in (i + 1):(j - 1)
                y += T[k, j] * F[i, k] - T[i, k] * F[k, j]
            end
            denom = T[i, i] - T[j, j]
            if abs(denom) < 1.0e-12
                error("Parlett recurrence encountered repeated or nearly repeated diagonal entries")
            end
            F[i, j] = y / denom
        end
    end
    return F
end

function parlett_exp_triangular(T::AbstractMatrix{ComplexF64})
    F = zeros(ComplexF64, size(T, 1), size(T, 2))
    return parlett_exp_triangular!(F, T)
end

function schur_parlett_exp!(out::Matrix{Float64},
                            plan::DenseComplexSchurParlettPlan,
                            dt::Float64)
    @. plan.scaledT = -dt * plan.T
    parlett_exp_triangular!(plan.ET, plan.scaledT)
    mul!(plan.tmp, plan.Z, plan.ET)
    mul!(plan.tmp2, plan.tmp, plan.Z')
    @. out = real(plan.tmp2)
    return out
end

function schur_parlett_exp(plan::DenseComplexSchurParlettPlan, dt::Float64)
    return schur_parlett_exp!(plan.out, plan, dt)
end

function schur_exp_adjoint(plan::DenseSchurPlan, Fbar::Matrix{Float64}, dt::Float64)
    Ebar = plan.Z' * Fbar * plan.Z
    Y = -dt * dense_frechet_exp(Matrix((-dt * plan.T)'), Ebar)
    return plan.Z * Y * plan.Z'
end

function relative_error(X::Matrix{Float64}, Y::Matrix{Float64})
    denom = max(norm(X), norm(Y), eps(Float64))
    return norm(X - Y) / denom
end

function batched_elapsed(f, batch::Int)
    sink = 0.0
    elapsed = @elapsed begin
        for _ in 1:batch
            sink += f()
        end
    end
    return elapsed, sink
end

function calibrate_batch(f, batch::Int, target_seconds::Float64, max_batch::Int)
    target_seconds <= 0.0 && return batch
    batch = max(1, batch)
    max_batch = max(batch, max_batch)
    while true
        elapsed, sink = batched_elapsed(f, batch)
        sink == -Inf && println("unreachable sink: ", sink)
        if elapsed >= target_seconds || batch >= max_batch
            return batch
        end
        if elapsed <= 0.0
            batch = min(max_batch, 10 * batch)
        else
            scale = clamp(ceil(Int, 1.10 * target_seconds / elapsed), 2, 100)
            batch = min(max_batch, batch * scale)
        end
    end
end

function median_time_ms(f; warmup::Int, reps::Int, batch::Int,
                        min_sample_ms::Float64 = 0.0, max_batch::Int = batch)
    batch = max(1, batch)
    for _ in 1:warmup
        batched_elapsed(f, batch)
    end
    batch = calibrate_batch(f, batch, min_sample_ms / 1000.0, max_batch)
    GC.gc()
    times = Vector{Float64}(undef, reps)
    sink = 0.0
    for i in 1:reps
        elapsed, sample_sink = batched_elapsed(f, batch)
        sink += sample_sink
        times[i] = 1000.0 * elapsed / batch
    end
    sink == -Inf && println("unreachable sink: ", sink)
    return TimingStats(median(times), mean(times), minimum(times), maximum(times), batch)
end

function reps_for_K(K::Int, base_reps::Int)
    if haskey(ENV, "REPS")
        return base_reps
    end
    K <= 8 && return max(base_reps, 25)
    K <= 16 && return max(div(base_reps, 1), 15)
    K <= 32 && return max(div(base_reps, 2), 8)
    K <= 64 && return max(div(base_reps, 4), 4)
    return max(div(base_reps, 8), 2)
end

function run_case(inst::SMBPInstance, ldu_inst::SMBPInstance, dt::Float64; warmup::Int, reps::Int,
                  batch::Int, min_sample_ms::Float64, max_batch::Int,
                  blas_threads::Int)
    K = inst.K
    E = zeros(K, K)
    F_smbp = zeros(K, K)
    tmp = zeros(K, K)
    tmp2 = zeros(K, K)
    Ebar = zeros(K, K)
    Dbar = zeros(K, K)
    Abar_smbp = zeros(K, K)
    F_ldu = zeros(K, K)
    Abar_ldu = zeros(K, K)
    plan = SMBPFrechetAdjointPlan(K)
    schur_plan = DenseSchurPlan(inst.A)
    schur_parlett_plan = DenseComplexSchurParlettPlan(inst.A)
    schur_adjoint_argument = Matrix((-dt * schur_plan.T)')
    schur_adjoint_seed = schur_plan.Z' * inst.Fbar * schur_plan.Z

    F_dense = exp(-dt * inst.A)
    F_schur = schur_exp(schur_plan, dt)
    F_schur_parlett = schur_parlett_exp(schur_parlett_plan, dt)
    smbp_block_exp!(E, inst, dt)
    materialize_exp_from_blocks!(F_smbp, tmp, inst.R, inst, dt)
    materialize_exp_from_blocks_inverse!(F_ldu, tmp, ldu_inst.R, ldu_inst.Rinv, ldu_inst, dt)

    Abar_dense = dense_exp_adjoint(inst.A, inst.Fbar, dt)
    Abar_schur = schur_exp_adjoint(schur_plan, inst.Fbar, dt)
    rotate_upstream_adjoint!(Ebar, tmp, inst.R, inst.Fbar)
    evaluate_plan!(plan, inst, dt)
    apply_plan!(Dbar, Ebar, plan)
    rotate_downstream_adjoint!(Abar_smbp, tmp2, inst.R, Dbar)
    rotate_upstream_adjoint_inverse!(Ebar, tmp, ldu_inst.R, ldu_inst.Rinv, ldu_inst.Fbar)
    apply_plan!(Dbar, Ebar, plan)
    rotate_downstream_adjoint_inverse!(Abar_ldu, tmp2, ldu_inst.R, ldu_inst.Rinv, Dbar)

    exp_relerr = relative_error(F_dense, F_smbp)
    adj_relerr = relative_error(Abar_dense, Abar_smbp)
    ldu_exp_relerr = relative_error(exp(-dt * ldu_inst.A), F_ldu)
    ldu_adj_relerr = relative_error(dense_exp_adjoint(ldu_inst.A, ldu_inst.Fbar, dt), Abar_ldu)
    schur_exp_relerr = relative_error(F_dense, F_schur)
    schur_parlett_exp_relerr = relative_error(F_dense, F_schur_parlett)
    schur_adj_relerr = relative_error(Abar_dense, Abar_schur)

    dense_exp_stats = median_time_ms(
        () -> begin
            F = exp(-dt * inst.A)
            return F[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_block_exp_stats = median_time_ms(
        () -> begin
            smbp_block_exp!(E, inst, dt)
            return E[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_exp_with_rotation_stats = median_time_ms(
        () -> begin
            materialize_exp_from_blocks!(F_smbp, tmp, inst.R, inst, dt)
            return F_smbp[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_exp_ldu_with_rotation_stats = median_time_ms(
        () -> begin
            materialize_exp_from_blocks_inverse!(F_ldu, tmp, ldu_inst.R, ldu_inst.Rinv, ldu_inst, dt)
            return F_ldu[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    schur_exp_stats = median_time_ms(
        () -> begin
            F = schur_exp(schur_plan, dt)
            return F[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    schur_basis_exp_stats = median_time_ms(
        () -> begin
            F = schur_basis_exp(schur_plan, dt)
            return F[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    schur_parlett_exp_stats = median_time_ms(
        () -> begin
            F = schur_parlett_exp(schur_parlett_plan, dt)
            return F[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    dense_adj_stats = median_time_ms(
        () -> begin
            Abar = dense_exp_adjoint(inst.A, inst.Fbar, dt)
            return Abar[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    schur_adj_stats = median_time_ms(
        () -> begin
            Abar = schur_exp_adjoint(schur_plan, inst.Fbar, dt)
            return Abar[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    schur_basis_adj_stats = median_time_ms(
        () -> begin
            Tbar = -dt * dense_frechet_exp(schur_adjoint_argument, schur_adjoint_seed)
            return Tbar[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_coeff_eval_stats = median_time_ms(
        () -> begin
            evaluate_plan!(plan, inst, dt)
            return plan.coeffs[1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_adj_apply_stats = median_time_ms(
        () -> begin
            apply_plan!(Dbar, Ebar, plan)
            return Dbar[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_adj_build_apply_stats = median_time_ms(
        () -> begin
            evaluate_plan!(plan, inst, dt)
            apply_plan!(Dbar, Ebar, plan)
            return Dbar[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_adj_with_rotation_stats = median_time_ms(
        () -> begin
            rotate_upstream_adjoint!(Ebar, tmp, inst.R, inst.Fbar)
            apply_plan!(Dbar, Ebar, plan)
            rotate_downstream_adjoint!(Abar_smbp, tmp2, inst.R, Dbar)
            return Abar_smbp[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_adj_ldu_with_rotation_stats = median_time_ms(
        () -> begin
            rotate_upstream_adjoint_inverse!(Ebar, tmp, ldu_inst.R, ldu_inst.Rinv, ldu_inst.Fbar)
            apply_plan!(Dbar, Ebar, plan)
            rotate_downstream_adjoint_inverse!(Abar_ldu, tmp2, ldu_inst.R, ldu_inst.Rinv, Dbar)
            return Abar_ldu[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    smbp_adj_uncached_with_rotation_stats = median_time_ms(
        () -> begin
            rotate_upstream_adjoint!(Ebar, tmp, inst.R, inst.Fbar)
            evaluate_plan!(plan, inst, dt)
            apply_plan!(Dbar, Ebar, plan)
            rotate_downstream_adjoint!(Abar_smbp, tmp2, inst.R, Dbar)
            return Abar_smbp[1, 1]
        end;
        warmup = warmup,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        max_batch = max_batch)

    return (
        K = K,
        blocks = div(K, 2),
        dt = dt,
        reps = reps,
        batch = batch,
        min_sample_ms = min_sample_ms,
        blas_threads = blas_threads,
        exp_relerr = exp_relerr,
        adj_relerr = adj_relerr,
        ldu_exp_relerr = ldu_exp_relerr,
        ldu_adj_relerr = ldu_adj_relerr,
        schur_exp_relerr = schur_exp_relerr,
        schur_parlett_exp_relerr = schur_parlett_exp_relerr,
        schur_adj_relerr = schur_adj_relerr,
        dense_exp_ms = dense_exp_stats.median_ms,
        schur_exp_ms = schur_exp_stats.median_ms,
        schur_basis_exp_ms = schur_basis_exp_stats.median_ms,
        schur_parlett_exp_ms = schur_parlett_exp_stats.median_ms,
        smbp_block_exp_ms = smbp_block_exp_stats.median_ms,
        smbp_exp_with_rotation_ms = smbp_exp_with_rotation_stats.median_ms,
        smbp_exp_ldu_with_rotation_ms = smbp_exp_ldu_with_rotation_stats.median_ms,
        dense_adj_ms = dense_adj_stats.median_ms,
        schur_adj_ms = schur_adj_stats.median_ms,
        schur_basis_adj_ms = schur_basis_adj_stats.median_ms,
        smbp_coeff_eval_ms = smbp_coeff_eval_stats.median_ms,
        smbp_adj_apply_ms = smbp_adj_apply_stats.median_ms,
        smbp_adj_build_apply_ms = smbp_adj_build_apply_stats.median_ms,
        smbp_adj_with_rotation_ms = smbp_adj_with_rotation_stats.median_ms,
        smbp_adj_ldu_with_rotation_ms = smbp_adj_ldu_with_rotation_stats.median_ms,
        smbp_adj_uncached_with_rotation_ms = smbp_adj_uncached_with_rotation_stats.median_ms,
        dense_exp_batch = dense_exp_stats.batch,
        schur_exp_batch = schur_exp_stats.batch,
        schur_basis_exp_batch = schur_basis_exp_stats.batch,
        schur_parlett_exp_batch = schur_parlett_exp_stats.batch,
        smbp_block_exp_batch = smbp_block_exp_stats.batch,
        smbp_exp_with_rotation_batch = smbp_exp_with_rotation_stats.batch,
        smbp_exp_ldu_with_rotation_batch = smbp_exp_ldu_with_rotation_stats.batch,
        dense_adj_batch = dense_adj_stats.batch,
        schur_adj_batch = schur_adj_stats.batch,
        schur_basis_adj_batch = schur_basis_adj_stats.batch,
        smbp_coeff_eval_batch = smbp_coeff_eval_stats.batch,
        smbp_adj_apply_batch = smbp_adj_apply_stats.batch,
        smbp_adj_build_apply_batch = smbp_adj_build_apply_stats.batch,
        smbp_adj_with_rotation_batch = smbp_adj_with_rotation_stats.batch,
        smbp_adj_ldu_with_rotation_batch = smbp_adj_ldu_with_rotation_stats.batch,
        smbp_adj_uncached_with_rotation_batch = smbp_adj_uncached_with_rotation_stats.batch,
    )
end

function write_header(io)
    println(io,
        "K,blocks,dt,reps,batch,min_sample_ms,blas_threads,exp_relerr,adj_relerr," *
        "ldu_exp_relerr,ldu_adj_relerr," *
        "schur_exp_relerr,schur_parlett_exp_relerr,schur_adj_relerr," *
        "dense_exp_ms,schur_exp_ms,schur_basis_exp_ms,schur_parlett_exp_ms," *
        "smbp_block_exp_ms,smbp_exp_with_rotation_ms,smbp_exp_ldu_with_rotation_ms," *
        "dense_adj_ms,schur_adj_ms,schur_basis_adj_ms,smbp_coeff_eval_ms,smbp_adj_apply_ms," *
        "smbp_adj_build_apply_ms,smbp_adj_with_rotation_ms,smbp_adj_ldu_with_rotation_ms," *
        "smbp_adj_uncached_with_rotation_ms," *
        "dense_exp_batch,schur_exp_batch,schur_basis_exp_batch,schur_parlett_exp_batch," *
        "smbp_block_exp_batch,smbp_exp_with_rotation_batch,smbp_exp_ldu_with_rotation_batch," *
        "dense_adj_batch,schur_adj_batch,schur_basis_adj_batch,smbp_coeff_eval_batch,smbp_adj_apply_batch," *
        "smbp_adj_build_apply_batch,smbp_adj_with_rotation_batch,smbp_adj_ldu_with_rotation_batch," *
        "smbp_adj_uncached_with_rotation_batch," *
        "dense_over_smbp_exp_block,dense_over_smbp_exp_with_rotation," *
        "dense_over_schur_exp,dense_over_schur_parlett_exp," *
        "schur_over_smbp_exp_with_rotation,schur_parlett_over_smbp_exp_with_rotation," *
        "adj_speedup_apply,adj_speedup_build_apply," *
        "adj_speedup_with_rotation,adj_speedup_uncached_with_rotation," *
        "dense_over_schur_adj,schur_over_smbp_adj_with_rotation," *
        "schur_over_smbp_adj_uncached_with_rotation")
end

fmt_csv(x::Integer) = string(x)
fmt_csv(x::AbstractFloat) = isfinite(x) ? @sprintf("%.6g", x) : string(x)

function write_row(io, row)
    exp_speedup_block = row.dense_exp_ms / row.smbp_block_exp_ms
    exp_speedup_rot = row.dense_exp_ms / row.smbp_exp_with_rotation_ms
    dense_over_schur_exp = row.dense_exp_ms / row.schur_exp_ms
    dense_over_schur_parlett_exp = row.dense_exp_ms / row.schur_parlett_exp_ms
    schur_over_smbp_exp_rot = row.schur_exp_ms / row.smbp_exp_with_rotation_ms
    schur_parlett_over_smbp_exp_rot = row.schur_parlett_exp_ms / row.smbp_exp_with_rotation_ms
    adj_speedup_apply = row.dense_adj_ms / row.smbp_adj_apply_ms
    adj_speedup_build_apply = row.dense_adj_ms / row.smbp_adj_build_apply_ms
    adj_speedup_rot = row.dense_adj_ms / row.smbp_adj_with_rotation_ms
    adj_speedup_uncached_rot = row.dense_adj_ms / row.smbp_adj_uncached_with_rotation_ms
    dense_over_schur_adj = row.dense_adj_ms / row.schur_adj_ms
    schur_over_smbp_adj_rot = row.schur_adj_ms / row.smbp_adj_with_rotation_ms
    schur_over_smbp_adj_uncached_rot = row.schur_adj_ms / row.smbp_adj_uncached_with_rotation_ms

    values = Any[
        row.K, row.blocks, row.dt, row.reps, row.batch, row.min_sample_ms, row.blas_threads,
        row.exp_relerr, row.adj_relerr,
        row.ldu_exp_relerr, row.ldu_adj_relerr,
        row.schur_exp_relerr, row.schur_parlett_exp_relerr, row.schur_adj_relerr,
        row.dense_exp_ms, row.schur_exp_ms, row.schur_basis_exp_ms, row.schur_parlett_exp_ms,
        row.smbp_block_exp_ms, row.smbp_exp_with_rotation_ms, row.smbp_exp_ldu_with_rotation_ms,
        row.dense_adj_ms, row.schur_adj_ms, row.schur_basis_adj_ms, row.smbp_coeff_eval_ms, row.smbp_adj_apply_ms,
        row.smbp_adj_build_apply_ms, row.smbp_adj_with_rotation_ms, row.smbp_adj_ldu_with_rotation_ms,
        row.smbp_adj_uncached_with_rotation_ms,
        row.dense_exp_batch, row.schur_exp_batch, row.schur_basis_exp_batch, row.schur_parlett_exp_batch,
        row.smbp_block_exp_batch, row.smbp_exp_with_rotation_batch, row.smbp_exp_ldu_with_rotation_batch,
        row.dense_adj_batch, row.schur_adj_batch, row.schur_basis_adj_batch, row.smbp_coeff_eval_batch, row.smbp_adj_apply_batch,
        row.smbp_adj_build_apply_batch, row.smbp_adj_with_rotation_batch, row.smbp_adj_ldu_with_rotation_batch,
        row.smbp_adj_uncached_with_rotation_batch,
        exp_speedup_block, exp_speedup_rot,
        dense_over_schur_exp, dense_over_schur_parlett_exp,
        schur_over_smbp_exp_rot, schur_parlett_over_smbp_exp_rot,
        adj_speedup_apply,
        adj_speedup_build_apply, adj_speedup_rot,
        adj_speedup_uncached_rot, dense_over_schur_adj,
        schur_over_smbp_adj_rot, schur_over_smbp_adj_uncached_rot,
    ]
    println(io, join(fmt_csv.(values), ","))
end

function main()
    blas_threads = configure_blas_threads()
    k_values = parse_int_list(get(ENV, "K_VALUES", nothing), DEFAULT_K_VALUES)
    dt_values = parse_float_list(get(ENV, "DT_VALUES", nothing), DEFAULT_DT_VALUES)
    warmup = env_int("WARMUP", 3)
    base_reps = env_int("REPS", 10)
    batch = env_int("INNER_REPS", 50)
    min_sample_ms = env_float("MIN_SAMPLE_MS", 50.0)
    max_batch = env_int("MAX_INNER_REPS", 10_000_000)
    seed = env_int("SEED", 1)
    out_path = get(ENV, "OUT", joinpath(@__DIR__, "..", "results", "schur_vs_smbp_exp_adjoint_results.csv"))

    @info "configured BLAS" blas_threads

    rng = MersenneTwister(seed)
    rows = []

    for (case_index, K) in enumerate(k_values)
        inst = make_instance(rng, K)
        ldu_inst = make_ldu_instance_like(rng, inst)
        dt = dt_values[1 + mod(case_index - 1, length(dt_values))]
        reps = reps_for_K(K, base_reps)
        @info "running case" K dt reps batch min_sample_ms max_batch
        row = run_case(
            inst, ldu_inst, dt;
            warmup = warmup,
            reps = reps,
            batch = batch,
            min_sample_ms = min_sample_ms,
            max_batch = max_batch,
            blas_threads = blas_threads)
        push!(rows, row)
        @printf("K=%d dt=%.5g exp_err=%.3e adj_err=%.3e dense_exp=%.3fms smbp_exp=%.3fms dense_adj=%.3fms smbp_adj=%.3fms\n",
                row.K, row.dt, row.exp_relerr, row.adj_relerr,
                row.dense_exp_ms, row.smbp_exp_with_rotation_ms,
                row.dense_adj_ms, row.smbp_adj_with_rotation_ms)
    end

    open(out_path, "w") do io
        write_header(io)
        for row in rows
            write_row(io, row)
        end
    end
    println("wrote ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
