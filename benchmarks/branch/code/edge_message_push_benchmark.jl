#!/usr/bin/env julia

# Child-to-parent OU edge message benchmark.
#
# This isolates the canonical Gaussian operation used after the child/subtree
# message has already been computed:
#
#   m_parent(x) = ∫ p(y | x) m_child(y) dy
#
# In canonical form, for transition blocks (x = parent, y = child),
#
#   A    = J_child + J_yy
#   J_p  = J_xx - J_xy A^{-1} J_yx
#   h_p  = h_x  - J_xy A^{-1} (h_child + h_y)
#
# The dense path stores all quantities in the original trait basis.  The SMBP
# path stores the OU transition in the 2x2 block basis, where transition
# precision blocks are block diagonal.  The "with rotation" timing includes
# rotating the child message into that basis and rotating the parent message
# back out.  The full-edge timings also include a known-Schur path where
# A = Z*T*Z' is precomputed, the child message is rotated into the Schur basis,
# exp(-dt*T) is evaluated there, and the parent message is rotated back out.

using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "schur_vs_smbp_exp_adjoint.jl"))

const EDGE_DEFAULT_K_VALUES = [4, 8, 16, 32, 64]
const EDGE_DEFAULT_DT_VALUES = [0.025, 0.041, 0.067, 0.113, 0.181]
const LOG_TWO_PI = log(2.0 * pi)

mutable struct EdgeTransition
    Jxx::Matrix{Float64}
    Jxy::Matrix{Float64}
    Jyx::Matrix{Float64}
    Jyy::Matrix{Float64}
    hx::Vector{Float64}
    hy::Vector{Float64}
    lognormalizer::Float64
end

function EdgeTransition(K::Int)
    return EdgeTransition(
        zeros(K, K), zeros(K, K), zeros(K, K), zeros(K, K),
        zeros(K), zeros(K), 0.0,
    )
end

struct EdgeInstance
    K::Int
    dt::Float64
    R::Matrix{Float64}
    Rinv::Matrix{Float64}
    smbp::SMBPInstance
    childJ_block::Matrix{Float64}
    childh_block::Vector{Float64}
    childJ_dense::Matrix{Float64}
    childh_dense::Vector{Float64}
    transition_block::EdgeTransition
    transition_dense::EdgeTransition
end

mutable struct EdgePushWorkspace
    A::Matrix{Float64}
    solve_matrix::Matrix{Float64}
    correction::Matrix{Float64}
    tmpJ::Matrix{Float64}
    outJ::Matrix{Float64}
    solve_vector::Vector{Float64}
    outh::Vector{Float64}
end

EdgePushWorkspace(K::Int) = EdgePushWorkspace(
    zeros(K, K), zeros(K, K), zeros(K, K), zeros(K, K), zeros(K, K),
    zeros(K), zeros(K),
)

mutable struct EdgeRotationWorkspace
    push::EdgePushWorkspace
    tmpJ1::Matrix{Float64}
    childJ_block::Matrix{Float64}
    parentJ_dense::Matrix{Float64}
    childh_block::Vector{Float64}
    parenth_dense::Vector{Float64}
end

EdgeRotationWorkspace(K::Int) = EdgeRotationWorkspace(
    EdgePushWorkspace(K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    zeros(K),
    zeros(K),
)

mutable struct EdgeFullWorkspace
    transition::EdgeTransition
    push::EdgePushWorkspace
    F::Matrix{Float64}
    Q::Matrix{Float64}
    P::Matrix{Float64}
    tmp::Matrix{Float64}
    eye::Matrix{Float64}
end

EdgeFullWorkspace(K::Int) = EdgeFullWorkspace(
    EdgeTransition(K),
    EdgePushWorkspace(K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    Matrix(I, K, K),
)

mutable struct EdgeFullRotationWorkspace
    full::EdgeFullWorkspace
    rot::EdgeRotationWorkspace
end

EdgeFullRotationWorkspace(K::Int) = EdgeFullRotationWorkspace(
    EdgeFullWorkspace(K),
    EdgeRotationWorkspace(K),
)

mutable struct EdgeFullKnownSchurWorkspace
    Z::Matrix{Float64}
    T::Matrix{Float64}
    scaledT::Matrix{Float64}
    FT::Matrix{Float64}
    full::EdgeFullWorkspace
    rot::EdgeRotationWorkspace
end

function EdgeFullKnownSchurWorkspace(A::Matrix{Float64})
    factor = schur(A)
    K = size(A, 1)
    return EdgeFullKnownSchurWorkspace(
        Matrix(factor.Z),
        Matrix(factor.T),
        zeros(K, K),
        zeros(K, K),
        EdgeFullWorkspace(K),
        EdgeRotationWorkspace(K),
    )
end

function make_edge_instance(rng::AbstractRNG, K::Int, dt::Float64)
    iseven(K) || error("K must be even; got K=$K")
    block_count = div(K, 2)
    diag = Vector{Float64}(undef, block_count)
    upper = Vector{Float64}(undef, block_count)
    lower = Vector{Float64}(undef, block_count)
    D = zeros(K, K)

    for b in 1:block_count
        i = 2b - 1
        a = 0.80 + 1.20 * rand(rng)
        u = 0.25 + 0.55 * rand(rng)
        l = -(0.25 + 0.55 * rand(rng))

        diag[b] = a
        upper[b] = u
        lower[b] = l

        D[i, i] = a
        D[i + 1, i + 1] = a
        D[i, i + 1] = u
        D[i + 1, i] = l
    end

    R = random_orthogonal(rng, K)
    Rinv = Matrix(R')
    A = R * D * Rinv
    smbp = SMBPInstance(K, R, Rinv, D, diag, upper, lower, A, randn(rng, K, K))

    G = randn(rng, K, K)
    childJ_block = Symmetric((G' * G) / K + 0.35I) |> Matrix
    childh_block = randn(rng, K)
    childJ_dense = R * childJ_block * R'
    childh_dense = R * childh_block

    transition_block = build_block_transition(smbp, dt)
    transition_dense = rotate_transition(transition_block, R)

    return EdgeInstance(
        K, dt, R, Rinv, smbp,
        childJ_block, childh_block,
        childJ_dense, childh_dense,
        transition_block, transition_dense,
    )
end

function make_ldu_edge_instance_like(rng::AbstractRNG, inst::EdgeInstance)
    ldu_smbp = make_ldu_instance_like(rng, inst.smbp)
    childJ_dense = ldu_smbp.Rinv' * inst.childJ_block * ldu_smbp.Rinv
    childh_dense = ldu_smbp.Rinv' * inst.childh_block
    transition_dense = rotate_transition_inverse(inst.transition_block, ldu_smbp.Rinv)
    return EdgeInstance(
        inst.K,
        inst.dt,
        ldu_smbp.R,
        ldu_smbp.Rinv,
        ldu_smbp,
        copy(inst.childJ_block),
        copy(inst.childh_block),
        childJ_dense,
        childh_dense,
        inst.transition_block,
        transition_dense,
    )
end

function build_block_transition(inst::SMBPInstance, dt::Float64)
    K = inst.K
    E = zeros(K, K)
    smbp_block_exp!(E, inst, dt)

    Q = Matrix(I, K, K) - E * E'
    Q = Symmetric(0.5 .* (Q .+ Q')) |> Matrix
    cholQ = cholesky(Symmetric(Q))
    P = inv(cholQ)

    Jyy = P
    Jyx = -P * E
    Jxy = Jyx'
    Jxx = E' * P * E
    hx = zeros(K)
    hy = zeros(K)
    lognormalizer = 0.5 * (K * LOG_TWO_PI + logdet(cholQ))
    return EdgeTransition(Jxx, Jxy, Jyx, Jyy, hx, hy, lognormalizer)
end

function zero_transition_blocks!(transition::EdgeTransition)
    fill!(transition.Jxx, 0.0)
    fill!(transition.Jxy, 0.0)
    fill!(transition.Jyx, 0.0)
    fill!(transition.Jyy, 0.0)
    fill!(transition.hx, 0.0)
    fill!(transition.hy, 0.0)
    transition.lognormalizer = 0.0
    return transition
end

function transpose_into!(out::Matrix{Float64}, input::Matrix{Float64})
    K = size(input, 1)
    @inbounds for j in 1:K
        for i in 1:K
            out[i, j] = input[j, i]
        end
    end
    return out
end

function fill_transition_from_moments!(transition::EdgeTransition,
                                       F::Matrix{Float64},
                                       Q::Matrix{Float64},
                                       P::Matrix{Float64},
                                       tmp::Matrix{Float64},
                                       eye::Matrix{Float64})
    K = size(F, 1)
    mul!(Q, F, F')
    @. Q = -Q + eye
    symmetrize!(Q)

    factor = cholesky!(Symmetric(Q, :L), check = false)
    copyto!(P, eye)
    ldiv!(factor, P)
    symmetrize!(P)

    copyto!(transition.Jyy, P)
    mul!(tmp, P, F)
    copyto!(transition.Jyx, tmp)
    @. transition.Jyx = -transition.Jyx
    transpose_into!(transition.Jxy, transition.Jyx)
    mul!(transition.Jxx, F', tmp)
    symmetrize!(transition.Jxx)
    fill!(transition.hx, 0.0)
    fill!(transition.hy, 0.0)
    transition.lognormalizer = 0.5 * (K * LOG_TWO_PI + logdet(factor))
    return transition
end

function fill_dense_transition!(work::EdgeFullWorkspace, inst::EdgeInstance)
    work.F .= exp(-inst.dt * inst.smbp.A)
    return fill_transition_from_moments!(
        work.transition,
        work.F,
        work.Q,
        work.P,
        work.tmp,
        work.eye,
    )
end

function fill_block_transition_specialized!(work::EdgeFullWorkspace, inst::EdgeInstance)
    K = inst.K
    zero_transition_blocks!(work.transition)
    smbp_block_exp!(work.F, inst.smbp, inst.dt)

    logdetQ = 0.0
    @inbounds for b in 1:div(K, 2)
        i = 2b - 1
        e00 = work.F[i, i]
        e01 = work.F[i, i + 1]
        e10 = work.F[i + 1, i]
        e11 = work.F[i + 1, i + 1]

        q00 = 1.0 - (e00 * e00 + e01 * e01)
        q01 = -(e00 * e10 + e01 * e11)
        q11 = 1.0 - (e10 * e10 + e11 * e11)
        detQ = q00 * q11 - q01 * q01
        if detQ <= 0.0
            error("Non-SPD block transition covariance in full edge benchmark")
        end
        logdetQ += log(detQ)

        p00 = q11 / detQ
        p01 = -q01 / detQ
        p11 = q00 / detQ

        pe00 = p00 * e00 + p01 * e10
        pe01 = p00 * e01 + p01 * e11
        pe10 = p01 * e00 + p11 * e10
        pe11 = p01 * e01 + p11 * e11

        transition = work.transition
        transition.Jyy[i, i] = p00
        transition.Jyy[i, i + 1] = p01
        transition.Jyy[i + 1, i] = p01
        transition.Jyy[i + 1, i + 1] = p11

        transition.Jyx[i, i] = -pe00
        transition.Jyx[i, i + 1] = -pe01
        transition.Jyx[i + 1, i] = -pe10
        transition.Jyx[i + 1, i + 1] = -pe11
        transition.Jxy[i, i] = -pe00
        transition.Jxy[i + 1, i] = -pe01
        transition.Jxy[i, i + 1] = -pe10
        transition.Jxy[i + 1, i + 1] = -pe11

        transition.Jxx[i, i] = e00 * pe00 + e10 * pe10
        transition.Jxx[i, i + 1] = e00 * pe01 + e10 * pe11
        transition.Jxx[i + 1, i] = e01 * pe00 + e11 * pe10
        transition.Jxx[i + 1, i + 1] = e01 * pe01 + e11 * pe11
    end
    work.transition.lognormalizer = 0.5 * (K * LOG_TWO_PI + logdetQ)
    return work.transition
end

function rotate_transition(t::EdgeTransition, R::Matrix{Float64})
    return EdgeTransition(
        R * t.Jxx * R',
        R * t.Jxy * R',
        R * t.Jyx * R',
        R * t.Jyy * R',
        R * t.hx,
        R * t.hy,
        t.lognormalizer,
    )
end

function rotate_transition_inverse(t::EdgeTransition, Rinv::Matrix{Float64})
    return EdgeTransition(
        Rinv' * t.Jxx * Rinv,
        Rinv' * t.Jxy * Rinv,
        Rinv' * t.Jyx * Rinv,
        Rinv' * t.Jyy * Rinv,
        Rinv' * t.hx,
        Rinv' * t.hy,
        t.lognormalizer,
    )
end

function symmetrize!(A::Matrix{Float64})
    K = size(A, 1)
    @inbounds for j in 1:K
        for i in (j + 1):K
            v = 0.5 * (A[i, j] + A[j, i])
            A[i, j] = v
            A[j, i] = v
        end
    end
    return A
end

function block_left_mul!(out::Matrix{Float64}, block_left::Matrix{Float64}, right::Matrix{Float64})
    K = size(out, 1)
    fill!(out, 0.0)
    @inbounds for b in 1:div(K, 2)
        i = 2b - 1
        a00 = block_left[i, i]
        a01 = block_left[i, i + 1]
        a10 = block_left[i + 1, i]
        a11 = block_left[i + 1, i + 1]
        for j in 1:K
            r0 = right[i, j]
            r1 = right[i + 1, j]
            out[i, j] = a00 * r0 + a01 * r1
            out[i + 1, j] = a10 * r0 + a11 * r1
        end
    end
    return out
end

function block_left_mul_vec!(out::Vector{Float64}, block_left::Matrix{Float64}, right::Vector{Float64})
    K = length(out)
    fill!(out, 0.0)
    @inbounds for b in 1:div(K, 2)
        i = 2b - 1
        r0 = right[i]
        r1 = right[i + 1]
        out[i] = block_left[i, i] * r0 + block_left[i, i + 1] * r1
        out[i + 1] = block_left[i + 1, i] * r0 + block_left[i + 1, i + 1] * r1
    end
    return out
end

function add_blockdiag!(out::Matrix{Float64}, blockdiag::Matrix{Float64})
    K = size(out, 1)
    @inbounds for b in 1:div(K, 2)
        i = 2b - 1
        out[i, i] += blockdiag[i, i]
        out[i, i + 1] += blockdiag[i, i + 1]
        out[i + 1, i] += blockdiag[i + 1, i]
        out[i + 1, i + 1] += blockdiag[i + 1, i + 1]
    end
    return out
end

function information_quadratic(childh::Vector{Float64},
                               transitionh::Vector{Float64},
                               solved::Vector{Float64})
    total = 0.0
    @inbounds for i in eachindex(solved)
        total += (childh[i] + transitionh[i]) * solved[i]
    end
    return total
end

function push_backward_dense!(work::EdgePushWorkspace,
                              childJ::Matrix{Float64},
                              childh::Vector{Float64},
                              transition::EdgeTransition)
    K = size(childJ, 1)
    copyto!(work.A, childJ)
    @. work.A = work.A + transition.Jyy
    symmetrize!(work.A)
    factor = cholesky!(Symmetric(work.A, :L), check = false)

    copyto!(work.solve_matrix, transition.Jyx)
    ldiv!(factor, work.solve_matrix)
    mul!(work.correction, transition.Jxy, work.solve_matrix)
    @. work.outJ = transition.Jxx - work.correction
    symmetrize!(work.outJ)

    @. work.solve_vector = childh + transition.hy
    ldiv!(factor, work.solve_vector)
    mul!(work.outh, transition.Jxy, work.solve_vector)
    @. work.outh = transition.hx - work.outh

    eliminated = 0.5 * (K * LOG_TWO_PI - logdet(factor) +
                        information_quadratic(childh, transition.hy, work.solve_vector))
    return transition.lognormalizer - eliminated + work.outJ[1, 1] + work.outh[1]
end

function push_backward_block!(work::EdgePushWorkspace,
                              childJ::Matrix{Float64},
                              childh::Vector{Float64},
                              transition::EdgeTransition)
    K = size(childJ, 1)
    copyto!(work.A, childJ)
    add_blockdiag!(work.A, transition.Jyy)
    symmetrize!(work.A)
    factor = cholesky!(Symmetric(work.A, :L), check = false)

    copyto!(work.solve_matrix, transition.Jyx)
    ldiv!(factor, work.solve_matrix)
    block_left_mul!(work.correction, transition.Jxy, work.solve_matrix)
    @. work.outJ = -work.correction
    add_blockdiag!(work.outJ, transition.Jxx)
    symmetrize!(work.outJ)

    @. work.solve_vector = childh + transition.hy
    ldiv!(factor, work.solve_vector)
    block_left_mul_vec!(work.outh, transition.Jxy, work.solve_vector)
    @. work.outh = transition.hx - work.outh

    eliminated = 0.5 * (K * LOG_TWO_PI - logdet(factor) +
                        information_quadratic(childh, transition.hy, work.solve_vector))
    return transition.lognormalizer - eliminated + work.outJ[1, 1] + work.outh[1]
end

function rotate_state_to_block!(J_block::Matrix{Float64},
                                h_block::Vector{Float64},
                                tmp::Matrix{Float64},
                                R::Matrix{Float64},
                                J_dense::Matrix{Float64},
                                h_dense::Vector{Float64})
    mul!(tmp, R', J_dense)
    mul!(J_block, tmp, R)
    symmetrize!(J_block)
    mul!(h_block, R', h_dense)
    return nothing
end

function rotate_state_to_dense!(J_dense::Matrix{Float64},
                                h_dense::Vector{Float64},
                                tmp::Matrix{Float64},
                                R::Matrix{Float64},
                                J_block::Matrix{Float64},
                                h_block::Vector{Float64})
    mul!(tmp, R, J_block)
    mul!(J_dense, tmp, R')
    symmetrize!(J_dense)
    mul!(h_dense, R, h_block)
    return nothing
end

function rotate_state_to_dense_inverse!(J_dense::Matrix{Float64},
                                        h_dense::Vector{Float64},
                                        tmp::Matrix{Float64},
                                        Rinv::Matrix{Float64},
                                        J_block::Matrix{Float64},
                                        h_block::Vector{Float64})
    mul!(tmp, Rinv', J_block)
    mul!(J_dense, tmp, Rinv)
    symmetrize!(J_dense)
    mul!(h_dense, Rinv', h_block)
    return nothing
end

function push_backward_block_with_rotation!(work::EdgeRotationWorkspace, inst::EdgeInstance)
    rotate_state_to_block!(
        work.childJ_block,
        work.childh_block,
        work.tmpJ1,
        inst.R,
        inst.childJ_dense,
        inst.childh_dense,
    )
    score = push_backward_block!(
        work.push,
        work.childJ_block,
        work.childh_block,
        inst.transition_block,
    )
    rotate_state_to_dense!(
        work.parentJ_dense,
        work.parenth_dense,
        work.tmpJ1,
        inst.R,
        work.push.outJ,
        work.push.outh,
    )
    return score + work.parentJ_dense[1, 1] + work.parenth_dense[1]
end

function push_backward_block_with_rotation_inverse!(work::EdgeRotationWorkspace, inst::EdgeInstance)
    rotate_state_to_block!(
        work.childJ_block,
        work.childh_block,
        work.tmpJ1,
        inst.R,
        inst.childJ_dense,
        inst.childh_dense,
    )
    score = push_backward_block!(
        work.push,
        work.childJ_block,
        work.childh_block,
        inst.transition_block,
    )
    rotate_state_to_dense_inverse!(
        work.parentJ_dense,
        work.parenth_dense,
        work.tmpJ1,
        inst.Rinv,
        work.push.outJ,
        work.push.outh,
    )
    return score + work.parentJ_dense[1, 1] + work.parenth_dense[1]
end

function full_edge_dense!(work::EdgeFullWorkspace, inst::EdgeInstance)
    fill_dense_transition!(work, inst)
    return push_backward_dense!(
        work.push,
        inst.childJ_dense,
        inst.childh_dense,
        work.transition,
    )
end

function full_edge_known_schur!(work::EdgeFullKnownSchurWorkspace, inst::EdgeInstance)
    full = work.full
    rot = work.rot
    rotate_state_to_block!(
        rot.childJ_block,
        rot.childh_block,
        rot.tmpJ1,
        work.Z,
        inst.childJ_dense,
        inst.childh_dense,
    )
    @. work.scaledT = -inst.dt * work.T
    work.FT .= exp(work.scaledT)
    fill_transition_from_moments!(
        full.transition,
        work.FT,
        full.Q,
        full.P,
        full.tmp,
        full.eye,
    )
    score = push_backward_dense!(
        full.push,
        rot.childJ_block,
        rot.childh_block,
        full.transition,
    )
    rotate_state_to_dense!(
        rot.parentJ_dense,
        rot.parenth_dense,
        rot.tmpJ1,
        work.Z,
        full.push.outJ,
        full.push.outh,
    )
    return score + rot.parentJ_dense[1, 1] + rot.parenth_dense[1]
end

function full_edge_block!(work::EdgeFullWorkspace, inst::EdgeInstance)
    fill_block_transition_specialized!(work, inst)
    return push_backward_block!(
        work.push,
        inst.childJ_block,
        inst.childh_block,
        work.transition,
    )
end

function full_edge_block_with_rotation!(work::EdgeFullRotationWorkspace, inst::EdgeInstance)
    rotate_state_to_block!(
        work.rot.childJ_block,
        work.rot.childh_block,
        work.rot.tmpJ1,
        inst.R,
        inst.childJ_dense,
        inst.childh_dense,
    )
    fill_block_transition_specialized!(work.full, inst)
    score = push_backward_block!(
        work.full.push,
        work.rot.childJ_block,
        work.rot.childh_block,
        work.full.transition,
    )
    rotate_state_to_dense!(
        work.rot.parentJ_dense,
        work.rot.parenth_dense,
        work.rot.tmpJ1,
        inst.R,
        work.full.push.outJ,
        work.full.push.outh,
    )
    return score + work.rot.parentJ_dense[1, 1] + work.rot.parenth_dense[1]
end

function full_edge_block_with_rotation_inverse!(work::EdgeFullRotationWorkspace, inst::EdgeInstance)
    rotate_state_to_block!(
        work.rot.childJ_block,
        work.rot.childh_block,
        work.rot.tmpJ1,
        inst.R,
        inst.childJ_dense,
        inst.childh_dense,
    )
    fill_block_transition_specialized!(work.full, inst)
    score = push_backward_block!(
        work.full.push,
        work.rot.childJ_block,
        work.rot.childh_block,
        work.full.transition,
    )
    rotate_state_to_dense_inverse!(
        work.rot.parentJ_dense,
        work.rot.parenth_dense,
        work.rot.tmpJ1,
        inst.Rinv,
        work.full.push.outJ,
        work.full.push.outh,
    )
    return score + work.rot.parentJ_dense[1, 1] + work.rot.parenth_dense[1]
end

function matrix_vector_relative_error(x::Vector{Float64}, y::Vector{Float64})
    return norm(x - y) / max(norm(x), norm(y), eps(Float64))
end

function run_edge_case(inst::EdgeInstance, ldu_inst::EdgeInstance; warmup::Int, reps::Int, batch::Int)
    K = inst.K
    dense_work = EdgePushWorkspace(K)
    block_work = EdgePushWorkspace(K)
    rot_work = EdgeRotationWorkspace(K)
    ldu_dense_work = EdgePushWorkspace(K)
    ldu_rot_work = EdgeRotationWorkspace(K)
    dense_full_work = EdgeFullWorkspace(K)
    schur_full_work = EdgeFullKnownSchurWorkspace(inst.smbp.A)
    block_full_work = EdgeFullWorkspace(K)
    rot_full_work = EdgeFullRotationWorkspace(K)
    ldu_rot_full_work = EdgeFullRotationWorkspace(K)

    push_backward_dense!(dense_work, inst.childJ_dense, inst.childh_dense, inst.transition_dense)
    push_backward_block!(block_work, inst.childJ_block, inst.childh_block, inst.transition_block)
    push_backward_block_with_rotation!(rot_work, inst)
    push_backward_dense!(ldu_dense_work, ldu_inst.childJ_dense, ldu_inst.childh_dense, ldu_inst.transition_dense)
    push_backward_block_with_rotation_inverse!(ldu_rot_work, ldu_inst)
    full_edge_dense!(dense_full_work, inst)
    full_edge_known_schur!(schur_full_work, inst)
    full_edge_block!(block_full_work, inst)
    full_edge_block_with_rotation!(rot_full_work, inst)
    full_edge_block_with_rotation_inverse!(ldu_rot_full_work, ldu_inst)

    J_block_to_dense = inst.R * block_work.outJ * inst.R'
    h_block_to_dense = inst.R * block_work.outh
    J_relerr = relative_error(dense_work.outJ, J_block_to_dense)
    h_relerr = matrix_vector_relative_error(dense_work.outh, h_block_to_dense)
    ldu_J_relerr = relative_error(ldu_dense_work.outJ, ldu_rot_work.parentJ_dense)
    ldu_h_relerr = matrix_vector_relative_error(ldu_dense_work.outh, ldu_rot_work.parenth_dense)
    ldu_full_J_relerr = relative_error(ldu_dense_work.outJ, ldu_rot_full_work.rot.parentJ_dense)
    ldu_full_h_relerr = matrix_vector_relative_error(ldu_dense_work.outh,
                                                     ldu_rot_full_work.rot.parenth_dense)
    schur_full_J_relerr = relative_error(dense_full_work.push.outJ, schur_full_work.rot.parentJ_dense)
    schur_full_h_relerr = matrix_vector_relative_error(dense_full_work.push.outh,
                                                       schur_full_work.rot.parenth_dense)
    full_J_block_to_dense = inst.R * block_full_work.push.outJ * inst.R'
    full_h_block_to_dense = inst.R * block_full_work.push.outh
    full_J_relerr = relative_error(dense_full_work.push.outJ, full_J_block_to_dense)
    full_h_relerr = matrix_vector_relative_error(dense_full_work.push.outh, full_h_block_to_dense)

    dense_stats = median_time_ms(
        () -> push_backward_dense!(dense_work, inst.childJ_dense, inst.childh_dense, inst.transition_dense);
        warmup = warmup, reps = reps, batch = batch)

    block_stats = median_time_ms(
        () -> push_backward_block!(block_work, inst.childJ_block, inst.childh_block, inst.transition_block);
        warmup = warmup, reps = reps, batch = batch)

    rot_stats = median_time_ms(
        () -> push_backward_block_with_rotation!(rot_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    ldu_rot_stats = median_time_ms(
        () -> push_backward_block_with_rotation_inverse!(ldu_rot_work, ldu_inst);
        warmup = warmup, reps = reps, batch = batch)

    dense_full_stats = median_time_ms(
        () -> full_edge_dense!(dense_full_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    schur_full_stats = median_time_ms(
        () -> full_edge_known_schur!(schur_full_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    block_full_stats = median_time_ms(
        () -> full_edge_block!(block_full_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    rot_full_stats = median_time_ms(
        () -> full_edge_block_with_rotation!(rot_full_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    ldu_rot_full_stats = median_time_ms(
        () -> full_edge_block_with_rotation_inverse!(ldu_rot_full_work, ldu_inst);
        warmup = warmup, reps = reps, batch = batch)

    return (
        K = K,
        dt = inst.dt,
        blocks = div(K, 2),
        reps = reps,
        batch = batch,
        J_relerr = J_relerr,
        h_relerr = h_relerr,
        ldu_J_relerr = ldu_J_relerr,
        ldu_h_relerr = ldu_h_relerr,
        schur_full_J_relerr = schur_full_J_relerr,
        schur_full_h_relerr = schur_full_h_relerr,
        full_J_relerr = full_J_relerr,
        full_h_relerr = full_h_relerr,
        ldu_full_J_relerr = ldu_full_J_relerr,
        ldu_full_h_relerr = ldu_full_h_relerr,
        dense_push_ms = dense_stats.median_ms,
        smbp_block_push_ms = block_stats.median_ms,
        smbp_push_with_rotation_ms = rot_stats.median_ms,
        smbp_ldu_push_with_rotation_ms = ldu_rot_stats.median_ms,
        dense_full_edge_ms = dense_full_stats.median_ms,
        schur_known_basis_full_edge_ms = schur_full_stats.median_ms,
        smbp_block_full_edge_ms = block_full_stats.median_ms,
        smbp_full_edge_with_rotation_ms = rot_full_stats.median_ms,
        smbp_ldu_full_edge_with_rotation_ms = ldu_rot_full_stats.median_ms,
    )
end

function edge_header()
    return "K,dt,blocks,reps,batch,J_relerr,h_relerr,ldu_J_relerr,ldu_h_relerr," *
           "schur_full_J_relerr,schur_full_h_relerr," *
           "full_J_relerr,full_h_relerr,ldu_full_J_relerr,ldu_full_h_relerr," *
           "dense_push_ms,smbp_block_push_ms,smbp_push_with_rotation_ms,smbp_ldu_push_with_rotation_ms," *
           "dense_over_smbp_block,dense_over_smbp_with_rotation,dense_over_smbp_ldu_with_rotation," *
           "dense_full_edge_ms,schur_known_basis_full_edge_ms," *
           "smbp_block_full_edge_ms,smbp_full_edge_with_rotation_ms,smbp_ldu_full_edge_with_rotation_ms," *
           "dense_full_over_schur_known_full,dense_full_over_smbp_block_full," *
           "dense_full_over_smbp_full_with_rotation,dense_full_over_smbp_ldu_full_with_rotation," *
           "schur_known_full_over_smbp_block_full,schur_known_full_over_smbp_full_with_rotation," *
           "schur_known_full_over_smbp_ldu_full_with_rotation"
end

function write_edge_row(io, row)
    values = Any[
        row.K, row.dt, row.blocks, row.reps, row.batch,
        row.J_relerr, row.h_relerr, row.ldu_J_relerr, row.ldu_h_relerr,
        row.schur_full_J_relerr, row.schur_full_h_relerr,
        row.full_J_relerr, row.full_h_relerr, row.ldu_full_J_relerr, row.ldu_full_h_relerr,
        row.dense_push_ms, row.smbp_block_push_ms, row.smbp_push_with_rotation_ms, row.smbp_ldu_push_with_rotation_ms,
        row.dense_push_ms / row.smbp_block_push_ms,
        row.dense_push_ms / row.smbp_push_with_rotation_ms,
        row.dense_push_ms / row.smbp_ldu_push_with_rotation_ms,
        row.dense_full_edge_ms, row.schur_known_basis_full_edge_ms,
        row.smbp_block_full_edge_ms, row.smbp_full_edge_with_rotation_ms, row.smbp_ldu_full_edge_with_rotation_ms,
        row.dense_full_edge_ms / row.schur_known_basis_full_edge_ms,
        row.dense_full_edge_ms / row.smbp_block_full_edge_ms,
        row.dense_full_edge_ms / row.smbp_full_edge_with_rotation_ms,
        row.dense_full_edge_ms / row.smbp_ldu_full_edge_with_rotation_ms,
        row.schur_known_basis_full_edge_ms / row.smbp_block_full_edge_ms,
        row.schur_known_basis_full_edge_ms / row.smbp_full_edge_with_rotation_ms,
        row.schur_known_basis_full_edge_ms / row.smbp_ldu_full_edge_with_rotation_ms,
    ]
    println(io, join(fmt_csv.(values), ","))
end

function main_edge()
    k_values = parse_int_list(get(ENV, "K_VALUES", nothing), EDGE_DEFAULT_K_VALUES)
    dt_values = parse_float_list(get(ENV, "DT_VALUES", nothing), EDGE_DEFAULT_DT_VALUES)
    warmup = env_int("WARMUP", 3)
    base_reps = env_int("REPS", 10)
    batch = env_int("INNER_REPS", 50)
    seed = env_int("SEED", 11)
    out_path = get(ENV, "OUT", joinpath(@__DIR__, "..", "results", "edge_message_push_results.csv"))

    rng = MersenneTwister(seed)
    rows = []
    for (case_index, K) in enumerate(k_values)
        dt = dt_values[1 + mod(case_index - 1, length(dt_values))]
        inst = make_edge_instance(rng, K, dt)
        ldu_inst = make_ldu_edge_instance_like(rng, inst)
        reps = reps_for_K(K, base_reps)
        @info "running edge case" K dt reps batch
        row = run_edge_case(inst, ldu_inst; warmup = warmup, reps = reps, batch = batch)
        push!(rows, row)
        @printf("K=%d dt=%.5g Jerr=%.3e herr=%.3e push_dense=%.4fms push_smbp=%.4fms full_dense=%.4fms full_schur=%.4fms full_smbp=%.4fms full_smbp_rot=%.4fms\n",
                row.K, row.dt, row.J_relerr, row.h_relerr,
                row.dense_push_ms, row.smbp_block_push_ms,
                row.dense_full_edge_ms, row.schur_known_basis_full_edge_ms,
                row.smbp_block_full_edge_ms,
                row.smbp_full_edge_with_rotation_ms)
    end

    open(out_path, "w") do io
        println(io, edge_header())
        for row in rows
            write_edge_row(io, row)
        end
    end
    println("wrote ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_edge()
end
