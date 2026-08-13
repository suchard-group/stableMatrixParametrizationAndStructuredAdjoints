#!/usr/bin/env julia

# Full branch likelihood/message benchmark with stationary variance solved inside
# the timed branch construction.
#
# For each branch this benchmark computes
#
#   A * Sigma + Sigma * A' = V
#   F   = exp(-dt * A)
#   Q_t = Sigma - F * Sigma * F'
#
# then builds the canonical Gaussian transition y | x ~ N(Fx, Q_t) and pushes a
# precomputed child canonical Gaussian message to the parent.
#
# Three paths are compared:
#
#   1. dense: Julia lyap(A, -V), dense exp(-dt*A), dense transition + push;
#   2. known Schur: precomputed A = Z*T*Z', timed Schur-basis Lyapunov,
#      exp(-dt*T), rotations, dense transition + push;
#   3. SMBP with rotation: rotate V and child message into the 2x2 block basis,
#      solve the block Lyapunov system, compute block exp, dense transition +
#      push in block basis, and rotate the parent message back.

using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "edge_message_push_benchmark.jl"))
include(joinpath(@__DIR__, "smbp_lyapunov_plan.jl"))

const BRANCH_LYAP_DEFAULT_K_VALUES = [4, 8, 16, 32, 64]
const BRANCH_LYAP_DEFAULT_DT_VALUES = [0.025, 0.041, 0.067, 0.113, 0.181]

struct BranchLyapunovInstance
    edge::EdgeInstance
    lyap_plan::SMBPLyapunovEqualDiagonalPlan
    V_block::Matrix{Float64}
    V_dense::Matrix{Float64}
end

mutable struct BranchLyapunovWorkspace
    transition::EdgeTransition
    push::EdgePushWorkspace
    Sigma::Matrix{Float64}
    F::Matrix{Float64}
    Q::Matrix{Float64}
    P::Matrix{Float64}
    tmp::Matrix{Float64}
    tmp2::Matrix{Float64}
    eye::Matrix{Float64}
end

BranchLyapunovWorkspace(K::Int) = BranchLyapunovWorkspace(
    EdgeTransition(K),
    EdgePushWorkspace(K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    Matrix(I, K, K),
)

mutable struct KnownSchurBranchLyapunovWorkspace
    Z::Matrix{Float64}
    T::Matrix{Float64}
    W::Matrix{Float64}
    FT::Matrix{Float64}
    branch::BranchLyapunovWorkspace
end

function KnownSchurBranchLyapunovWorkspace(A::Matrix{Float64})
    factor = schur(A)
    K = size(A, 1)
    return KnownSchurBranchLyapunovWorkspace(
        Matrix(factor.Z),
        Matrix(factor.T),
        zeros(K, K),
        zeros(K, K),
        BranchLyapunovWorkspace(K),
    )
end

mutable struct SMBPBranchLyapunovWorkspace
    branch::BranchLyapunovWorkspace
    rotation::EdgeRotationWorkspace
    V_block::Matrix{Float64}
    Sigma_block::Matrix{Float64}
    F_block::Matrix{Float64}
    tmp::Matrix{Float64}
end

SMBPBranchLyapunovWorkspace(K::Int) = SMBPBranchLyapunovWorkspace(
    BranchLyapunovWorkspace(K),
    EdgeRotationWorkspace(K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
    zeros(K, K),
)

function make_branch_lyapunov_instance(rng::AbstractRNG, K::Int, dt::Float64)
    edge = make_edge_instance(rng, K, dt)
    G = randn(rng, K, K)
    V_block = Symmetric((G' * G) / K + 0.50I) |> Matrix
    V_dense = edge.R * V_block * edge.R'
    lyap_plan = SMBPLyapunovEqualDiagonalPlan(edge.smbp.diag, edge.smbp.upper, edge.smbp.lower)
    return BranchLyapunovInstance(edge, lyap_plan, V_block, V_dense)
end

function make_ldu_branch_lyapunov_instance_like(rng::AbstractRNG,
                                                inst::BranchLyapunovInstance)
    edge = make_ldu_edge_instance_like(rng, inst.edge)
    V_dense = edge.R * inst.V_block * edge.R'
    lyap_plan = SMBPLyapunovEqualDiagonalPlan(edge.smbp.diag, edge.smbp.upper, edge.smbp.lower)
    return BranchLyapunovInstance(edge, lyap_plan, copy(inst.V_block), V_dense)
end

function fill_transition_from_stationary!(transition::EdgeTransition,
                                          F::Matrix{Float64},
                                          Sigma::Matrix{Float64},
                                          Q::Matrix{Float64},
                                          P::Matrix{Float64},
                                          tmp::Matrix{Float64},
                                          eye::Matrix{Float64})
    K = size(F, 1)
    mul!(tmp, F, Sigma)
    mul!(Q, tmp, F')
    @. Q = Sigma - Q
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

function full_branch_dense_with_lyapunov!(work::BranchLyapunovWorkspace,
                                          inst::BranchLyapunovInstance)
    edge = inst.edge
    work.Sigma .= lyap(edge.smbp.A, -inst.V_dense)
    symmetrize!(work.Sigma)
    work.F .= exp(-edge.dt * edge.smbp.A)
    fill_transition_from_stationary!(
        work.transition,
        work.F,
        work.Sigma,
        work.Q,
        work.P,
        work.tmp,
        work.eye,
    )
    return push_backward_dense!(
        work.push,
        edge.childJ_dense,
        edge.childh_dense,
        work.transition,
    )
end

function full_branch_known_schur_with_lyapunov!(work::KnownSchurBranchLyapunovWorkspace,
                                                inst::BranchLyapunovInstance)
    edge = inst.edge
    branch = work.branch

    mul!(branch.tmp, work.Z', inst.V_dense)
    mul!(work.W, branch.tmp, work.Z)
    _, scale = LAPACK.trsyl!('N', 'T', work.T, work.T, work.W, 1)
    rmul!(work.W, inv(scale))
    mul!(branch.tmp, work.Z, work.W)
    mul!(branch.Sigma, branch.tmp, work.Z')
    symmetrize!(branch.Sigma)

    work.FT .= exp(-edge.dt * work.T)
    mul!(branch.tmp, work.Z, work.FT)
    mul!(branch.F, branch.tmp, work.Z')

    fill_transition_from_stationary!(
        branch.transition,
        branch.F,
        branch.Sigma,
        branch.Q,
        branch.P,
        branch.tmp,
        branch.eye,
    )
    return push_backward_dense!(
        branch.push,
        edge.childJ_dense,
        edge.childh_dense,
        branch.transition,
    )
end

function full_branch_smbp_with_lyapunov_rotation!(work::SMBPBranchLyapunovWorkspace,
                                                  inst::BranchLyapunovInstance)
    edge = inst.edge
    rot = work.rotation
    branch = work.branch

    mul!(work.tmp, edge.R', inst.V_dense)
    mul!(work.V_block, work.tmp, edge.R)
    symmetrize!(work.V_block)
    apply_smbp_lyapunov_symmetric!(work.Sigma_block, inst.lyap_plan, work.V_block)
    smbp_block_exp!(work.F_block, edge.smbp, edge.dt)

    rotate_state_to_block!(
        rot.childJ_block,
        rot.childh_block,
        rot.tmpJ1,
        edge.R,
        edge.childJ_dense,
        edge.childh_dense,
    )

    fill_transition_from_stationary!(
        branch.transition,
        work.F_block,
        work.Sigma_block,
        branch.Q,
        branch.P,
        branch.tmp,
        branch.eye,
    )
    score = push_backward_dense!(
        branch.push,
        rot.childJ_block,
        rot.childh_block,
        branch.transition,
    )
    rotate_state_to_dense!(
        rot.parentJ_dense,
        rot.parenth_dense,
        rot.tmpJ1,
        edge.R,
        branch.push.outJ,
        branch.push.outh,
    )
    return score + rot.parentJ_dense[1, 1] + rot.parenth_dense[1]
end

function full_branch_smbp_with_lyapunov_rotation_inverse!(work::SMBPBranchLyapunovWorkspace,
                                                          inst::BranchLyapunovInstance)
    edge = inst.edge
    rot = work.rotation
    branch = work.branch

    mul!(work.tmp, edge.Rinv, inst.V_dense)
    mul!(work.V_block, work.tmp, edge.Rinv')
    symmetrize!(work.V_block)
    apply_smbp_lyapunov_symmetric!(work.Sigma_block, inst.lyap_plan, work.V_block)
    smbp_block_exp!(work.F_block, edge.smbp, edge.dt)

    rotate_state_to_block!(
        rot.childJ_block,
        rot.childh_block,
        rot.tmpJ1,
        edge.R,
        edge.childJ_dense,
        edge.childh_dense,
    )

    fill_transition_from_stationary!(
        branch.transition,
        work.F_block,
        work.Sigma_block,
        branch.Q,
        branch.P,
        branch.tmp,
        branch.eye,
    )
    score = push_backward_dense!(
        branch.push,
        rot.childJ_block,
        rot.childh_block,
        branch.transition,
    )
    rotate_state_to_dense_inverse!(
        rot.parentJ_dense,
        rot.parenth_dense,
        rot.tmpJ1,
        edge.Rinv,
        branch.push.outJ,
        branch.push.outh,
    )
    return score + rot.parentJ_dense[1, 1] + rot.parenth_dense[1]
end

function run_branch_lyapunov_case(inst::BranchLyapunovInstance,
                                  ldu_inst::BranchLyapunovInstance;
                                  warmup::Int, reps::Int, batch::Int)
    K = inst.edge.K
    dense_work = BranchLyapunovWorkspace(K)
    schur_work = KnownSchurBranchLyapunovWorkspace(inst.edge.smbp.A)
    smbp_work = SMBPBranchLyapunovWorkspace(K)
    ldu_dense_work = BranchLyapunovWorkspace(K)
    ldu_smbp_work = SMBPBranchLyapunovWorkspace(K)

    full_branch_dense_with_lyapunov!(dense_work, inst)
    full_branch_known_schur_with_lyapunov!(schur_work, inst)
    full_branch_smbp_with_lyapunov_rotation!(smbp_work, inst)
    full_branch_dense_with_lyapunov!(ldu_dense_work, ldu_inst)
    full_branch_smbp_with_lyapunov_rotation_inverse!(ldu_smbp_work, ldu_inst)

    schur_J_relerr = relative_error(dense_work.push.outJ, schur_work.branch.push.outJ)
    schur_h_relerr = matrix_vector_relative_error(dense_work.push.outh, schur_work.branch.push.outh)
    smbp_J_relerr = relative_error(dense_work.push.outJ, smbp_work.rotation.parentJ_dense)
    smbp_h_relerr = matrix_vector_relative_error(dense_work.push.outh, smbp_work.rotation.parenth_dense)
    smbp_ldu_J_relerr = relative_error(ldu_dense_work.push.outJ, ldu_smbp_work.rotation.parentJ_dense)
    smbp_ldu_h_relerr = matrix_vector_relative_error(ldu_dense_work.push.outh,
                                                     ldu_smbp_work.rotation.parenth_dense)

    dense_stats = median_time_ms(
        () -> full_branch_dense_with_lyapunov!(dense_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    schur_stats = median_time_ms(
        () -> full_branch_known_schur_with_lyapunov!(schur_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    smbp_stats = median_time_ms(
        () -> full_branch_smbp_with_lyapunov_rotation!(smbp_work, inst);
        warmup = warmup, reps = reps, batch = batch)

    smbp_ldu_stats = median_time_ms(
        () -> full_branch_smbp_with_lyapunov_rotation_inverse!(ldu_smbp_work, ldu_inst);
        warmup = warmup, reps = reps, batch = batch)

    return (
        K = K,
        dt = inst.edge.dt,
        blocks = div(K, 2),
        reps = reps,
        batch = batch,
        schur_J_relerr = schur_J_relerr,
        schur_h_relerr = schur_h_relerr,
        smbp_J_relerr = smbp_J_relerr,
        smbp_h_relerr = smbp_h_relerr,
        smbp_ldu_J_relerr = smbp_ldu_J_relerr,
        smbp_ldu_h_relerr = smbp_ldu_h_relerr,
        dense_branch_lyap_ms = dense_stats.median_ms,
        schur_branch_lyap_ms = schur_stats.median_ms,
        smbp_branch_lyap_rotation_ms = smbp_stats.median_ms,
        smbp_branch_lyap_ldu_rotation_ms = smbp_ldu_stats.median_ms,
    )
end

function branch_lyapunov_header()
    return "K,dt,blocks,reps,batch,schur_J_relerr,schur_h_relerr," *
           "smbp_J_relerr,smbp_h_relerr,smbp_ldu_J_relerr,smbp_ldu_h_relerr," *
           "dense_branch_lyap_ms,schur_branch_lyap_ms," *
           "smbp_branch_lyap_rotation_ms,smbp_branch_lyap_ldu_rotation_ms," *
           "dense_over_schur,dense_over_smbp_rotation,dense_over_smbp_ldu_rotation," *
           "schur_over_smbp_rotation,schur_over_smbp_ldu_rotation"
end

function write_branch_lyapunov_row(io, row)
    values = Any[
        row.K, row.dt, row.blocks, row.reps, row.batch,
        row.schur_J_relerr, row.schur_h_relerr,
        row.smbp_J_relerr, row.smbp_h_relerr,
        row.smbp_ldu_J_relerr, row.smbp_ldu_h_relerr,
        row.dense_branch_lyap_ms,
        row.schur_branch_lyap_ms,
        row.smbp_branch_lyap_rotation_ms,
        row.smbp_branch_lyap_ldu_rotation_ms,
        row.dense_branch_lyap_ms / row.schur_branch_lyap_ms,
        row.dense_branch_lyap_ms / row.smbp_branch_lyap_rotation_ms,
        row.dense_branch_lyap_ms / row.smbp_branch_lyap_ldu_rotation_ms,
        row.schur_branch_lyap_ms / row.smbp_branch_lyap_rotation_ms,
        row.schur_branch_lyap_ms / row.smbp_branch_lyap_ldu_rotation_ms,
    ]
    println(io, join(fmt_csv.(values), ","))
end

function main_branch_lyapunov()
    k_values = parse_int_list(get(ENV, "K_VALUES", nothing), BRANCH_LYAP_DEFAULT_K_VALUES)
    dt_values = parse_float_list(get(ENV, "DT_VALUES", nothing), BRANCH_LYAP_DEFAULT_DT_VALUES)
    warmup = env_int("WARMUP", 3)
    base_reps = env_int("REPS", 10)
    batch = env_int("INNER_REPS", 50)
    seed = env_int("SEED", 23)
    out_path = get(ENV, "OUT", joinpath(@__DIR__, "..", "results", "branch_likelihood_with_lyapunov_results.csv"))

    rng = MersenneTwister(seed)
    rows = []
    for (case_index, K) in enumerate(k_values)
        dt = dt_values[1 + mod(case_index - 1, length(dt_values))]
        inst = make_branch_lyapunov_instance(rng, K, dt)
        ldu_inst = make_ldu_branch_lyapunov_instance_like(rng, inst)
        reps = reps_for_K(K, base_reps)
        @info "running branch Lyapunov case" K dt reps batch
        row = run_branch_lyapunov_case(inst, ldu_inst; warmup = warmup, reps = reps, batch = batch)
        push!(rows, row)
        @printf("K=%d dt=%.5g err_smbp_J=%.3e dense=%.4fms schur=%.4fms smbp_rot=%.4fms\n",
                row.K, row.dt, row.smbp_J_relerr,
                row.dense_branch_lyap_ms,
                row.schur_branch_lyap_ms,
                row.smbp_branch_lyap_rotation_ms)
    end

    open(out_path, "w") do io
        println(io, branch_lyapunov_header())
        for row in rows
            write_branch_lyapunov_row(io, row)
        end
    end
    println("wrote ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_branch_lyapunov()
end
