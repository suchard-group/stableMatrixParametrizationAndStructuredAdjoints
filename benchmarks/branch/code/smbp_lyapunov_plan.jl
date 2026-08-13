# Cached equal-diagonal 2x2 block Lyapunov solver.
#
# Solves D*X + X*D' = H for symmetric H when D is block diagonal with
# equal-diagonal 2x2 blocks.  The plan precomputes the linear maps for all
# diagonal and upper-triangular block pairs; application only reads the
# symmetric half and mirrors the solution.

struct SMBPLyapunovEqualDiagonalPlan
    K::Int
    block_count::Int
    diag_coeffs::Vector{Float64}
    offdiag_coeffs::Vector{Float64}
end

function _solve_equal_diag_22(ai::Float64, ui::Float64, vi::Float64,
                              aj::Float64, uj::Float64, vj::Float64,
                              h00::Float64, h01::Float64,
                              h10::Float64, h11::Float64)
    s = ai + aj
    ri2 = ui * vi
    rj2 = uj * vj

    a = s * s + rj2 - ri2
    b = 2.0 * s * uj
    c = 2.0 * s * vj
    det = a * a - b * c
    scale = max(abs(a), abs(b), abs(c))
    (scale > 0.0 && abs(det) >= eps(Float64) * scale * scale) ||
        error("Singular block Lyapunov pair")

    invdet = 1.0 / det

    q0 = s * h00 + uj * h01 - ui * h10
    q1 = vj * h00 + s * h01 - ui * h11
    r0 = s * h10 + uj * h11 - vi * h00
    r1 = vj * h10 + s * h11 - vi * h01

    y00 = (a * q0 - b * q1) * invdet
    y01 = (a * q1 - c * q0) * invdet
    y10 = (a * r0 - b * r1) * invdet
    y11 = (a * r1 - c * r0) * invdet
    return y00, y01, y10, y11
end

function _solve3!(A::NTuple{9,Float64}, b::NTuple{3,Float64})
    a11, a12, a13,
    a21, a22, a23,
    a31, a32, a33 = A
    b1, b2, b3 = b

    det =
        a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
    scale = max(
        abs(a11), abs(a12), abs(a13),
        abs(a21), abs(a22), abs(a23),
        abs(a31), abs(a32), abs(a33),
    )
    (scale > 0.0 && abs(det) >= eps(Float64) * scale * scale * scale) ||
        error("Singular diagonal block Lyapunov system")
    invdet = 1.0 / det

    x1 =
        b1 * (a22 * a33 - a23 * a32) -
        a12 * (b2 * a33 - a23 * b3) +
        a13 * (b2 * a32 - a22 * b3)
    x2 =
        a11 * (b2 * a33 - a23 * b3) -
        b1 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * b3 - b2 * a31)
    x3 =
        a11 * (a22 * b3 - b2 * a32) -
        a12 * (a21 * b3 - b2 * a31) +
        b1 * (a21 * a32 - a22 * a31)
    return x1 * invdet, x2 * invdet, x3 * invdet
end

function SMBPLyapunovEqualDiagonalPlan(diag::Vector{Float64},
                                       upper::Vector{Float64},
                                       lower::Vector{Float64})
    block_count = length(diag)
    length(upper) == block_count || error("upper length mismatch")
    length(lower) == block_count || error("lower length mismatch")
    K = 2 * block_count
    diag_coeffs = zeros(9 * block_count)
    offdiag_coeffs = zeros(16 * block_count * block_count)

    for bi in 1:block_count
        ai = diag[bi]
        ui = upper[bi]
        vi = lower[bi]
        # Symmetric diagonal block solve:
        # [2a 2u 0; v 2a u; 0 2v 2a] * [y00,y01,y11] = [h00,h01,h11].
        system = (
            2.0 * ai, 2.0 * ui, 0.0,
            vi, 2.0 * ai, ui,
            0.0, 2.0 * vi, 2.0 * ai,
        )
        base = 9 * (bi - 1)
        for col in 1:3
            rhs = col == 1 ? (1.0, 0.0, 0.0) :
                  col == 2 ? (0.0, 1.0, 0.0) :
                             (0.0, 0.0, 1.0)
            y00, y01, y11 = _solve3!(system, rhs)
            diag_coeffs[base + 3 * (col - 1) + 1] = y00
            diag_coeffs[base + 3 * (col - 1) + 2] = y01
            diag_coeffs[base + 3 * (col - 1) + 3] = y11
        end

        for bj in (bi + 1):block_count
            aj = diag[bj]
            uj = upper[bj]
            vj = lower[bj]
            base4 = 16 * ((bi - 1) * block_count + (bj - 1))
            for col in 1:4
                h00 = col == 1 ? 1.0 : 0.0
                h01 = col == 2 ? 1.0 : 0.0
                h10 = col == 3 ? 1.0 : 0.0
                h11 = col == 4 ? 1.0 : 0.0
                y00, y01, y10, y11 = _solve_equal_diag_22(
                    ai, ui, vi, aj, uj, vj, h00, h01, h10, h11)
                offdiag_coeffs[base4 + 4 * (col - 1) + 1] = y00
                offdiag_coeffs[base4 + 4 * (col - 1) + 2] = y01
                offdiag_coeffs[base4 + 4 * (col - 1) + 3] = y10
                offdiag_coeffs[base4 + 4 * (col - 1) + 4] = y11
            end
        end
    end

    return SMBPLyapunovEqualDiagonalPlan(K, block_count, diag_coeffs, offdiag_coeffs)
end

function apply_smbp_lyapunov_symmetric!(out::Matrix{Float64},
                                        plan::SMBPLyapunovEqualDiagonalPlan,
                                        H::Matrix{Float64})
    K = plan.K
    block_count = plan.block_count
    @inbounds for bi in 1:block_count
        i = 2bi - 1
        h00 = H[i, i]
        h01 = H[i, i + 1]
        h11 = H[i + 1, i + 1]
        d = 9 * (bi - 1)
        y00 = plan.diag_coeffs[d + 1] * h00 + plan.diag_coeffs[d + 4] * h01 + plan.diag_coeffs[d + 7] * h11
        y01 = plan.diag_coeffs[d + 2] * h00 + plan.diag_coeffs[d + 5] * h01 + plan.diag_coeffs[d + 8] * h11
        y11 = plan.diag_coeffs[d + 3] * h00 + plan.diag_coeffs[d + 6] * h01 + plan.diag_coeffs[d + 9] * h11
        out[i, i] = y00
        out[i, i + 1] = y01
        out[i + 1, i] = y01
        out[i + 1, i + 1] = y11

        for bj in (bi + 1):block_count
            j = 2bj - 1
            h00 = H[i, j]
            h01 = H[i, j + 1]
            h10 = H[i + 1, j]
            h11 = H[i + 1, j + 1]
            c = 16 * ((bi - 1) * block_count + (bj - 1))
            y00 = plan.offdiag_coeffs[c + 1]  * h00 + plan.offdiag_coeffs[c + 5]  * h01 +
                  plan.offdiag_coeffs[c + 9]  * h10 + plan.offdiag_coeffs[c + 13] * h11
            y01 = plan.offdiag_coeffs[c + 2]  * h00 + plan.offdiag_coeffs[c + 6]  * h01 +
                  plan.offdiag_coeffs[c + 10] * h10 + plan.offdiag_coeffs[c + 14] * h11
            y10 = plan.offdiag_coeffs[c + 3]  * h00 + plan.offdiag_coeffs[c + 7]  * h01 +
                  plan.offdiag_coeffs[c + 11] * h10 + plan.offdiag_coeffs[c + 15] * h11
            y11 = plan.offdiag_coeffs[c + 4]  * h00 + plan.offdiag_coeffs[c + 8]  * h01 +
                  plan.offdiag_coeffs[c + 12] * h10 + plan.offdiag_coeffs[c + 16] * h11

            out[i, j] = y00
            out[i, j + 1] = y01
            out[i + 1, j] = y10
            out[i + 1, j + 1] = y11

            out[j, i] = y00
            out[j + 1, i] = y01
            out[j, i + 1] = y10
            out[j + 1, i + 1] = y11
        end
    end
    return out
end
