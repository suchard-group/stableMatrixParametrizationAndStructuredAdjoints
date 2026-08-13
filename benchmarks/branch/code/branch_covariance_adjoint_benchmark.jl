#!/usr/bin/env julia

# Branch covariance adjoint benchmark.
#
# This isolates the reverse-mode counterpart of the full edge message: given
# a random symmetric seed Pbar on the precision contributed by the
# innovation/branch covariance, P = Q^{-1} with Q = I - F*F' (F = exp(-dt*A)),
# it computes the cotangent of the drift matrix A. The forward construction
# (build F from A and dt, form Q and P) matches the full edge message
# benchmark exactly.
#
# The seed lives on P rather than Q because P (= Jyy in the edge-message
# construction) is the quantity with an exact change-of-basis congruence for
# any invertible R; Q = I - F*F' does not, because the bare identity term
# does not commute with a non-orthogonal change of basis (R*R' <> I unless R
# is orthogonal). Q and P are therefore (re)computed natively inside each
# basis from that basis's own F, and only the precision-like seed Pbar is
# ever carried across bases.
#
# Following the Lyapunov-adjoint convention used elsewhere in this file set,
# the known-Schur and SSBP timings rotate the seed into the cached basis and
# solve there, but exclude the final change of basis ("pullback") back to
# dense coordinates. The SSBP path uses a single generic invertible rotation
# R; its inverse is cached once outside the timed region (never re-derived
# inside it, since the pullback that would need it is excluded from timing).
#
# Correctness checks: the known-Schur basis is always orthogonal (a Schur
# factor), so its pulled-back result can be checked directly against the
# dense reference. The same orthogonal check is used to validate the SMBP
# kernel itself, by pulling back through the edge instance's own orthogonal
# rotation. The separately drawn generic invertible rotation used for the
# *timed* SMBP kernel is not re-validated against an independently
# regenerated dense drift matrix: because Q = I - F*F' is not congruence-
# exact under a non-orthogonal change of basis, no such comparison is
# meaningful (the full edge message benchmark sidesteps the same issue by
# reusing, rather than re-deriving, the block-native transition for its own
# generic-invertible-rotation checks).

using LinearAlgebra
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "edge_message_push_benchmark.jl"))

const BRANCH_COV_ADJ_DEFAULT_K_VALUES = EDGE_DEFAULT_K_VALUES
const BRANCH_COV_ADJ_DEFAULT_DT_VALUES = EDGE_DEFAULT_DT_VALUES

struct BranchCovAdjointInstance
    edge::EdgeInstance
    Pbar::Matrix{Float64}
    R::Matrix{Float64}
    Rinv::Matrix{Float64}
end

function make_branch_cov_adjoint_instance(rng::AbstractRNG, K::Int, dt::Float64)
    edge = make_edge_instance(rng, K, dt)
    M = randn(rng, K, K)
    Pbar = Matrix(Symmetric(0.5 .* (M .+ M')))
    R, Rinv = random_ldu_rotation_pair(rng, K)
    return BranchCovAdjointInstance(edge, Pbar, R, Rinv)
end

# --- shared local step: Pbar -> Qbar -> Fbar, given this basis's own F -----

function precision_seed_to_exp_seed(P::Matrix{Float64}, Pbar::Matrix{Float64}, F::Matrix{Float64})
    Qbar = -(P * Pbar * P)
    return -(Qbar .+ Qbar') * F
end

function dense_branch_covariance(F::Matrix{Float64})
    K = size(F, 1)
    Q = Matrix(I, K, K) .- F * F'
    Q = 0.5 .* (Q .+ Q')
    P = inv(Q)
    return Q, 0.5 .* (P .+ P')
end

# --- dense (full; no basis change to exclude) -------------------------------

function dense_branch_cov_adjoint(A::Matrix{Float64}, Pbar::Matrix{Float64}, dt::Float64)
    F = exp(-dt * A)
    _, P = dense_branch_covariance(F)
    Fbar = precision_seed_to_exp_seed(P, Pbar, F)
    return dense_exp_adjoint(A, Fbar, dt)
end

# --- known Schur (basis is always orthogonal) -------------------------------

mutable struct SchurBranchCovAdjointWorkspace
    Z::Matrix{Float64}
    T::Matrix{Float64}
    scaledT::Matrix{Float64}
    Pbar_schur::Matrix{Float64}
    tmp::Matrix{Float64}
end

function SchurBranchCovAdjointWorkspace(A::Matrix{Float64})
    factor = schur(A)
    K = size(A, 1)
    return SchurBranchCovAdjointWorkspace(
        Matrix(factor.Z), Matrix(factor.T), zeros(K, K), zeros(K, K), zeros(K, K),
    )
end

function schur_branch_cov_adjoint_basis!(work::SchurBranchCovAdjointWorkspace,
                                         Pbar_dense::Matrix{Float64}, dt::Float64)
    mul!(work.tmp, work.Z', Pbar_dense)
    mul!(work.Pbar_schur, work.tmp, work.Z)
    @. work.scaledT = -dt * work.T
    F_schur = exp(UpperHessenberg(work.scaledT))
    _, P_schur = dense_branch_covariance(F_schur)
    Fbar_schur = precision_seed_to_exp_seed(P_schur, work.Pbar_schur, F_schur)
    return -dt * dense_frechet_exp(Matrix(work.scaledT'), Fbar_schur)
end

function schur_branch_cov_adjoint_with_pullback(work::SchurBranchCovAdjointWorkspace,
                                                Pbar_dense::Matrix{Float64}, dt::Float64)
    Tbar = schur_branch_cov_adjoint_basis!(work, Pbar_dense, dt)
    return work.Z * Tbar * work.Z'
end

# --- SSBP / SMBP -------------------------------------------------------------

mutable struct SMBPBranchCovAdjointWorkspace
    plan::SMBPFrechetAdjointPlan
    Pbar_block::Matrix{Float64}
    F_block::Matrix{Float64}
    P_block::Matrix{Float64}
    Fbar_block::Matrix{Float64}
    Dbar::Matrix{Float64}
    tmp::Matrix{Float64}
end

SMBPBranchCovAdjointWorkspace(K::Int) = SMBPBranchCovAdjointWorkspace(
    SMBPFrechetAdjointPlan(K), zeros(K, K), zeros(K, K), zeros(K, K), zeros(K, K), zeros(K, K), zeros(K, K),
)

# The Frechet-adjoint coefficient plan depends only on D and dt, not on the
# seed Pbar; like the Schur factorization Z,T cached for the known-Schur
# baseline, it is assumed already available and is excluded from the timed
# kernel below (matching the established convention for the plotted "SSBP"
# series in the matrix-exponential-adjoint panel, which times apply_plan!
# alone and treats evaluate_plan! as precomputed).
function SMBPBranchCovAdjointWorkspace(smbp::SMBPInstance, dt::Float64)
    work = SMBPBranchCovAdjointWorkspace(smbp.K)
    evaluate_plan!(work.plan, smbp, dt)
    return work
end

function fill_block_diagonal_precision!(P_block::Matrix{Float64}, F_block::Matrix{Float64})
    K = size(F_block, 1)
    fill!(P_block, 0.0)
    @inbounds for b in 1:div(K, 2)
        i = 2b - 1
        e00 = F_block[i, i]
        e01 = F_block[i, i + 1]
        e10 = F_block[i + 1, i]
        e11 = F_block[i + 1, i + 1]

        q00 = 1.0 - (e00 * e00 + e01 * e01)
        q01 = -(e00 * e10 + e01 * e11)
        q11 = 1.0 - (e10 * e10 + e11 * e11)
        detQ = q00 * q11 - q01 * q01
        detQ <= 0.0 && error("Non-SPD block branch covariance in branch covariance adjoint benchmark")

        P_block[i, i] = q11 / detQ
        P_block[i, i + 1] = -q01 / detQ
        P_block[i + 1, i] = -q01 / detQ
        P_block[i + 1, i + 1] = q00 / detQ
    end
    return P_block
end

# R, Rinv are passed explicitly: inst.R/inst.Rinv on SMBPInstance is the
# edge instance's own (orthogonal) rotation, which is reused only for the
# orthogonal correctness check; the timed path uses a separately drawn
# generic invertible rotation instead.
function smbp_branch_cov_adjoint_basis!(work::SMBPBranchCovAdjointWorkspace,
                                        smbp::SMBPInstance, R::Matrix{Float64},
                                        Pbar_dense::Matrix{Float64}, dt::Float64)
    mul!(work.tmp, R', Pbar_dense)
    mul!(work.Pbar_block, work.tmp, R)
    smbp_block_exp!(work.F_block, smbp, dt)
    fill_block_diagonal_precision!(work.P_block, work.F_block)
    Fbar_block = precision_seed_to_exp_seed(work.P_block, work.Pbar_block, work.F_block)
    copyto!(work.Fbar_block, Fbar_block)
    apply_plan!(work.Dbar, work.Fbar_block, work.plan)
    return work.Dbar
end

function smbp_branch_cov_adjoint_with_pullback(work::SMBPBranchCovAdjointWorkspace,
                                               smbp::SMBPInstance, R::Matrix{Float64}, Rinv::Matrix{Float64},
                                               Pbar_dense::Matrix{Float64}, dt::Float64)
    Dbar = smbp_branch_cov_adjoint_basis!(work, smbp, R, Pbar_dense, dt)
    tmp = zeros(size(Dbar))
    Abar = zeros(size(Dbar))
    rotate_downstream_adjoint_inverse!(Abar, tmp, R, Rinv, Dbar)
    return Abar
end

# --- case driver -------------------------------------------------------------

function run_branch_cov_adjoint_case(inst::BranchCovAdjointInstance;
                                     warmup::Int, reps::Int, batch::Int)
    K = inst.edge.K
    dt = inst.edge.dt
    A = inst.edge.smbp.A
    smbp = inst.edge.smbp

    schur_work = SchurBranchCovAdjointWorkspace(A)
    smbp_check_work = SMBPBranchCovAdjointWorkspace(smbp, dt)
    smbp_work = SMBPBranchCovAdjointWorkspace(smbp, dt)

    Abar_dense = dense_branch_cov_adjoint(A, inst.Pbar, dt)
    Abar_schur = schur_branch_cov_adjoint_with_pullback(schur_work, inst.Pbar, dt)
    schur_relerr = relative_error(Abar_dense, Abar_schur)

    # Orthogonal sanity check on the SMBP kernel: the edge instance's own
    # rotation is orthogonal, so the pullback is directly comparable to the
    # dense reference (see module docstring for why the generic invertible
    # rotation used below for timing cannot be checked the same way).
    Abar_smbp_check = smbp_branch_cov_adjoint_with_pullback(
        smbp_check_work, smbp, smbp.R, smbp.Rinv, inst.Pbar, dt)
    smbp_relerr = relative_error(Abar_dense, Abar_smbp_check)

    dense_stats = median_time_ms(
        () -> begin
            Abar = dense_branch_cov_adjoint(A, inst.Pbar, dt)
            return Abar[1, 1]
        end;
        warmup = warmup, reps = reps, batch = batch)

    schur_basis_stats = median_time_ms(
        () -> begin
            Tbar = schur_branch_cov_adjoint_basis!(schur_work, inst.Pbar, dt)
            return Tbar[1, 1]
        end;
        warmup = warmup, reps = reps, batch = batch)

    smbp_basis_stats = median_time_ms(
        () -> begin
            Dbar = smbp_branch_cov_adjoint_basis!(smbp_work, smbp, inst.R, inst.Pbar, dt)
            return Dbar[1, 1]
        end;
        warmup = warmup, reps = reps, batch = batch)

    return (
        K = K,
        dt = dt,
        blocks = div(K, 2),
        reps = reps,
        batch = batch,
        schur_relerr = schur_relerr,
        smbp_relerr = smbp_relerr,
        dense_branch_cov_adj_ms = dense_stats.median_ms,
        schur_basis_branch_cov_adj_ms = schur_basis_stats.median_ms,
        smbp_basis_branch_cov_adj_ms = smbp_basis_stats.median_ms,
    )
end

function branch_cov_adjoint_header()
    return "K,dt,blocks,reps,batch,schur_relerr,smbp_relerr," *
           "dense_branch_cov_adj_ms,schur_basis_branch_cov_adj_ms,smbp_basis_branch_cov_adj_ms," *
           "dense_over_schur_basis,dense_over_smbp_basis,schur_basis_over_smbp_basis"
end

function write_branch_cov_adjoint_row(io, row)
    values = Any[
        row.K, row.dt, row.blocks, row.reps, row.batch,
        row.schur_relerr, row.smbp_relerr,
        row.dense_branch_cov_adj_ms, row.schur_basis_branch_cov_adj_ms, row.smbp_basis_branch_cov_adj_ms,
        row.dense_branch_cov_adj_ms / row.schur_basis_branch_cov_adj_ms,
        row.dense_branch_cov_adj_ms / row.smbp_basis_branch_cov_adj_ms,
        row.schur_basis_branch_cov_adj_ms / row.smbp_basis_branch_cov_adj_ms,
    ]
    println(io, join(fmt_csv.(values), ","))
end

function main_branch_cov_adjoint()
    k_values = parse_int_list(get(ENV, "K_VALUES", nothing), BRANCH_COV_ADJ_DEFAULT_K_VALUES)
    dt_values = parse_float_list(get(ENV, "DT_VALUES", nothing), BRANCH_COV_ADJ_DEFAULT_DT_VALUES)
    warmup = env_int("WARMUP", 3)
    base_reps = env_int("REPS", 10)
    batch = env_int("INNER_REPS", 50)
    seed = env_int("SEED", 17)
    out_path = get(ENV, "OUT", joinpath(@__DIR__, "..", "results", "branch_covariance_adjoint_results.csv"))

    rng = MersenneTwister(seed)
    rows = []
    for (case_index, K) in enumerate(k_values)
        dt = dt_values[1 + mod(case_index - 1, length(dt_values))]
        inst = make_branch_cov_adjoint_instance(rng, K, dt)
        reps = reps_for_K(K, base_reps)
        @info "running branch covariance adjoint case" K dt reps batch
        row = run_branch_cov_adjoint_case(inst; warmup = warmup, reps = reps, batch = batch)
        push!(rows, row)
        @printf("K=%d dt=%.5g schur_err=%.3e smbp_err=%.3e dense=%.4fms schur_basis=%.4fms smbp_basis=%.4fms\n",
                row.K, row.dt, row.schur_relerr, row.smbp_relerr,
                row.dense_branch_cov_adj_ms, row.schur_basis_branch_cov_adj_ms, row.smbp_basis_branch_cov_adj_ms)
    end

    open(out_path, "w") do io
        println(io, branch_cov_adjoint_header())
        for row in rows
            write_branch_cov_adjoint_row(io, row)
        end
    end
    println("wrote ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_branch_cov_adjoint()
end
