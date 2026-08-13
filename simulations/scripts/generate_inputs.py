#!/usr/bin/env python3
"""Generate the K=5 MAP simulation battery used in the paper.

The script writes BEAST XML files, truth matrices, simulated data, and a task
manifest for the three simulation panels:

* real/complex boundary;
* non-orthogonal shear;
* Jordan(4,1) coupling.

For each grid point it creates independent OU trajectories, then emits
orthogonal H-SSBP and globally normalized dense-R H-SSBP MAP fits with
multiple small-random starts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


DIM = 5
BLOCK_COUNT = DIM // 2
ANGLE_COUNT = DIM * (DIM - 1) // 2
TIME_COUNT = 800
WINDOW_LOW = 0.0
WINDOW_HIGH = 0.1
MEAN = np.zeros(DIM)
DIFFUSION = np.diag([0.16, 0.12, 0.14, 0.11, 0.13])
OBSERVATION = np.diag([0.012, 0.010, 0.011, 0.009, 0.0105])
BASE_SCALAR = np.array([0.72])
BASE_RHO = np.array([0.55, 0.35])
BASE_THETA = np.array([0.12, -0.09])
BASE_T = np.array([0.24, 0.16])
BASE_ANGLES = np.array([0.16, -0.12, 0.08, 0.05, -0.09, 0.11, -0.06, 0.07, -0.04, 0.10])
REAL_COMPLEX_RATIOS = (0.5, 0.75, 0.95, 1.0, 1.05, 1.25, 1.5)
SHEARS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
JORDAN_COUPLINGS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
SIM_SEEDS = {
    "real_complex_boundary": 2026074300,
    "nonorthogonal_shear": 2026080100,
    "jordan41_coupling": 2026080200,
}
FIT_SEED_BASE = 2026089500
START_SEED_BASE = 2026089000
SHEAR_TRUTH_SEED = 20260741
JORDAN_ROTATION_SEED = 20260742
START_SD = 0.05
SHRINKAGE_LAMBDA = 0.1
MAP_ITERATIONS = 5000
LOG_EVERY = 100
CHECKPOINT_EVERY = 10000
MIN_POSITIVE_T = 1.0e-6


@dataclass(frozen=True)
class Case:
    panel: str
    panel_key: str
    x_column: str
    x_value: float
    grid_tag: str
    replicate: int


@dataclass
class Truth:
    selection: np.ndarray
    drift: np.ndarray
    initial_cov: np.ndarray
    scalar: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    t: np.ndarray
    rotation: np.ndarray | None
    metadata: dict


@dataclass(frozen=True)
class Start:
    scalar: np.ndarray
    rho: np.ndarray
    theta: np.ndarray
    t: np.ndarray
    angles: np.ndarray | None
    raw_rotation: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("simulations/runs/final_battery"))
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=MAP_ITERATIONS)
    parser.add_argument("--start-sd", type=float, default=START_SD)
    parser.add_argument("--shrinkage-lambda", type=float, default=SHRINKAGE_LAMBDA)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{float(value):.12g}"


def fmt_vec(values: Iterable[float]) -> str:
    return " ".join(fmt(float(value)) for value in values)


def token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def pair_indices(dim: int):
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            yield i, j


def givens_rotation(dim: int, angles: np.ndarray) -> np.ndarray:
    out = np.eye(dim)
    for theta, (i, j) in zip(angles, pair_indices(dim)):
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        old_i = out[:, i].copy()
        old_j = out[:, j].copy()
        out[:, i] = old_i * c + old_j * s
        out[:, j] = -old_i * s + old_j * c
    return out


def deterministic_orthogonal(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    signs = np.where(np.diag(r) == 0.0, 1.0, np.sign(np.diag(r)))
    return q * signs


def block_diag(blocks: Iterable[np.ndarray]) -> np.ndarray:
    blocks = tuple(blocks)
    out = np.zeros((sum(block.shape[0] for block in blocks),) * 2)
    offset = 0
    for block in blocks:
        size = block.shape[0]
        out[offset : offset + size, offset : offset + size] = block
        offset += size
    return out


def hssbp_block_matrix(scalar: np.ndarray, rho_increments: np.ndarray,
                       theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros((DIM, DIM))
    out[0, 0] = float(scalar[0])
    for block, (rho, th, tt) in enumerate(zip(np.cumsum(rho_increments), theta, t)):
        start = 1 + 2 * block
        out[start : start + 2, start : start + 2] = [
            [rho * math.cos(th), rho * math.sin(th) - tt],
            [rho * math.sin(th) + tt, rho * math.cos(th)],
        ]
    return out


def matrix_sqrt_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def stable_initial_cov(selection: np.ndarray) -> np.ndarray:
    cov = solve_continuous_lyapunov(selection, DIFFUSION)
    return 0.5 * (cov + cov.T)


def baseline_selection(t_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    t_values = BASE_T if t_values is None else np.asarray(t_values, dtype=float)
    rotation = givens_rotation(DIM, BASE_ANGLES)
    block = hssbp_block_matrix(BASE_SCALAR, BASE_RHO, BASE_THETA, t_values)
    return rotation @ block @ rotation.T, rotation


def shear_truth(shear: float) -> Truth:
    rng = np.random.default_rng(SHEAR_TRUTH_SEED)
    signs = np.where(rng.integers(0, 2, size=BLOCK_COUNT) == 0, -1.0, 1.0)
    rates = np.linspace(0.5, 1.5, BLOCK_COUNT + 1)
    drift_blocks = []
    canonical_blocks = []
    bases = []
    scalar_rate = rates[-1]
    for rate, sign in zip(rates[:BLOCK_COUNT], signs):
        a = float(rate)
        b = 2.0 * a
        k = sign * float(shear) * a
        drift_blocks.append(np.array([[-a, k], [0.0, -b]]))
        m = 0.5 * (a + b)
        d = 0.5 * (b - a)
        canonical_blocks.append(np.array([[-m, d], [d, -m]]))
        h = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
        v = np.array([[1.0, k / (a - b)], [0.0, 1.0]])
        bases.append(v @ h)
    drift0 = block_diag((*drift_blocks, np.array([[-scalar_rate]])))
    canonical = block_diag((*canonical_blocks, np.array([[-scalar_rate]])))
    basis0 = block_diag((*bases, np.array([[1.0]])))
    dense_rotation = deterministic_orthogonal(SHEAR_TRUTH_SEED, DIM)
    drift = dense_rotation @ drift0 @ dense_rotation.T
    basis = dense_rotation @ basis0
    order = [DIM - 1, 0, 1, 2, 3]
    start_rotation = basis[:, order]
    effective_rho = np.array([math.sqrt(0.5 * (rate**2 + (2.0 * rate) ** 2)) for rate in rates[:BLOCK_COUNT]])
    rho = np.empty(BLOCK_COUNT)
    rho[0] = effective_rho[0]
    rho[1:] = np.diff(effective_rho)
    theta = np.array([math.atan2(rate - 2.0 * rate, rate + 2.0 * rate) for rate in rates[:BLOCK_COUNT]])
    selection = -drift
    sym = 0.5 * (drift + drift.T)
    metadata = {
        "construction": "diagonalizable nonnormal drift from sheared real blocks",
        "shear_multiplier": float(shear),
        "shear_signs": [float(value) for value in signs],
        "basis_condition_number": float(np.linalg.cond(start_rotation)),
        "orthogonal_rmse_lower_bound": float(np.linalg.norm(np.maximum(np.linalg.eigvalsh(sym), 0.0)) / DIM),
        "reconstruction_relative_error": float(np.linalg.norm(drift - basis @ canonical @ np.linalg.inv(basis)) / np.linalg.norm(drift)),
    }
    return Truth(
        selection=selection,
        drift=drift,
        initial_cov=stable_initial_cov(selection),
        scalar=np.array([scalar_rate]),
        rho=rho,
        theta=theta,
        t=np.full(BLOCK_COUNT, MIN_POSITIVE_T),
        rotation=start_rotation,
        metadata=metadata,
    )


def jordan_truth(coupling: float) -> Truth:
    block4 = -np.eye(4)
    for index in range(3):
        block4[index, index + 1] = float(coupling)
    jordan = block_diag((block4, np.array([[-2.0]])))
    rotation = deterministic_orthogonal(JORDAN_ROTATION_SEED, DIM)
    drift = rotation @ jordan @ rotation.T
    selection = -drift
    sym = 0.5 * (drift + drift.T)
    metadata = {
        "construction": "orthogonal similarity of a 4x4 Jordan block and one scalar block",
        "jordan_coupling": float(coupling),
        "block_sizes": [4, 1],
        "orthogonal_densifier_seed": JORDAN_ROTATION_SEED,
        "symmetric_part_max_eigenvalue": float(np.max(np.linalg.eigvalsh(sym))),
    }
    return Truth(
        selection=selection,
        drift=drift,
        initial_cov=stable_initial_cov(selection),
        scalar=BASE_SCALAR.copy(),
        rho=BASE_RHO.copy(),
        theta=0.5 * BASE_THETA,
        t=0.75 * BASE_T,
        rotation=None,
        metadata=metadata,
    )


def truth_for_case(case: Case) -> Truth:
    if case.panel_key == "real_complex_boundary":
        t_values = BASE_T.copy()
        boundary = abs(float(np.cumsum(BASE_RHO)[0]) * math.sin(float(BASE_THETA[0])))
        t_values[0] = boundary * float(case.x_value)
        selection, rotation = baseline_selection(t_values)
        disc = (np.cumsum(BASE_RHO)[0] * math.sin(BASE_THETA[0])) ** 2 - t_values[0] ** 2
        metadata = {
            "construction": "orthogonal H-SSBP with one block moved across the real/complex boundary",
            "omega_ratio": float(case.x_value),
            "omega_star": float(boundary),
            "omega": float(t_values[0]),
            "block_discriminant": float(disc),
        }
        return Truth(
            selection=selection,
            drift=-selection,
            initial_cov=stable_initial_cov(selection),
            scalar=BASE_SCALAR.copy(),
            rho=BASE_RHO.copy(),
            theta=BASE_THETA.copy(),
            t=t_values,
            rotation=rotation,
            metadata=metadata,
        )
    if case.panel_key == "nonorthogonal_shear":
        return shear_truth(case.x_value)
    if case.panel_key == "jordan41_coupling":
        return jordan_truth(case.x_value)
    raise ValueError(case.panel_key)


def default_start() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alternating = 1.0 + 0.05 * np.where(np.arange(BLOCK_COUNT) % 2 == 0, 1.0, -1.0)
    return (
        BASE_SCALAR * 0.95 + 0.03,
        BASE_RHO * alternating,
        BASE_THETA * 0.5,
        BASE_T * 0.75,
    )


def start_for(case: Case, truth: Truth, fit_family: str, start_seed: int,
              start_sd: float) -> Start:
    rng = np.random.default_rng(start_seed)
    if case.panel_key == "real_complex_boundary":
        scalar, rho, theta, t = truth.scalar, truth.rho, truth.theta, truth.t
    elif case.panel_key == "nonorthogonal_shear" and fit_family == "generic dense-R H-SSBP":
        scalar, rho, theta, t = truth.scalar, truth.rho, truth.theta, truth.t
    else:
        scalar, rho, theta, t = default_start()
    if fit_family == "orthogonal H-SSBP":
        angles = rng.normal(0.0, start_sd, size=ANGLE_COUNT)
        return Start(scalar.copy(), rho.copy(), theta.copy(), t.copy(), angles, None)
    raw_rotation = np.eye(DIM) + rng.normal(0.0, start_sd, size=(DIM, DIM))
    return Start(scalar.copy(), rho.copy(), theta.copy(), t.copy(), None, raw_rotation)


def simulate_observations(truth: Truth, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = matrix_sqrt_spd(truth.initial_cov) @ rng.normal(size=DIM)
    obs_sqrt = matrix_sqrt_spd(OBSERVATION)
    times = np.zeros(TIME_COUNT)
    observations = np.empty((TIME_COUNT, DIM))
    for time_index in range(TIME_COUNT):
        observations[time_index] = state + obs_sqrt @ rng.normal(size=DIM)
        if time_index < TIME_COUNT - 1:
            dt = max(float(rng.uniform(WINDOW_LOW, WINDOW_HIGH)), 1.0e-12)
            times[time_index + 1] = times[time_index] + dt
            transition = expm(-truth.selection * dt)
            innovation_cov = truth.initial_cov - transition @ truth.initial_cov @ transition.T
            innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
            state = transition @ state + matrix_sqrt_spd(innovation_cov) @ rng.normal(size=DIM)
    return observations, times


def parameter_xml(parameter_id: str, values: np.ndarray, lower=None, upper=None) -> str:
    pieces = [f'<parameter id="{parameter_id}" value="{fmt_vec(values)}"']
    if lower is not None:
        pieces.append(f'lower="{fmt_vec(np.asarray(lower, dtype=float))}"')
    if upper is not None:
        pieces.append(f'upper="{fmt_vec(np.asarray(upper, dtype=float))}"')
    return "    " + " ".join(pieces) + "/>"


def matrix_parameter_xml(matrix_id: str, matrix: np.ndarray, indent: str = "    ") -> str:
    lines = [f'{indent}<matrixParameter id="{matrix_id}">']
    for col in range(matrix.shape[1]):
        lines.append(f'{indent}    <parameter value="{fmt_vec(matrix[:, col])}"/>')
    lines.append(f"{indent}</matrixParameter>")
    return "\n".join(lines)


def observations_xml(observations: np.ndarray) -> str:
    lines = ['    <matrixParameter id="series01.observations">']
    for row in observations:
        lines.append(f'        <parameter value="{fmt_vec(row)}"/>')
    lines.append("    </matrixParameter>")
    return "\n".join(lines)


def rotation_xml(fit_family: str, start: Start) -> str:
    if fit_family == "orthogonal H-SSBP":
        return """
    <givensRotationMatrixParameter id="ou.A.rotation" dimension="5">
        <angles><parameter idref="ou.A.angles"/></angles>
    </givensRotationMatrixParameter>
"""
    return f"""
    <normalizedMatrixParameter id="ou.A.rotation" normalization="global">
{matrix_parameter_xml("ou.A.rotation.raw", start.raw_rotation, indent="        ")}
    </normalizedMatrixParameter>
"""


def drift_xml(fit_family: str) -> str:
    if fit_family == "orthogonal H-SSBP":
        return """
    <orthogonalBlockDiagonalPolarStableMatrixParameter id="ou.A" blockRhoOrdering="ascending">
        <orthogonalRotationMatrix><givensRotationMatrixParameter idref="ou.A.rotation"/></orthogonalRotationMatrix>
        <scalarBlock><parameter idref="ou.A.scalar"/></scalarBlock>
        <blockRho><parameter idref="ou.A.rho"/></blockRho>
        <blockTheta><parameter idref="ou.A.theta"/></blockTheta>
        <blockT><parameter idref="ou.A.t"/></blockT>
    </orthogonalBlockDiagonalPolarStableMatrixParameter>
"""
    return """
    <blockDiagonalPolarStableMatrixParameter id="ou.A" blockRhoOrdering="ascending">
        <rotationMatrix><normalizedMatrixParameter idref="ou.A.rotation"/></rotationMatrix>
        <scalarBlock><parameter idref="ou.A.scalar"/></scalarBlock>
        <blockRho><parameter idref="ou.A.rho"/></blockRho>
        <blockTheta><parameter idref="ou.A.theta"/></blockTheta>
        <blockT><parameter idref="ou.A.t"/></blockT>
    </blockDiagonalPolarStableMatrixParameter>
"""


def fit_specific_xml(fit_family: str, shrinkage_lambda: float) -> dict[str, str]:
    if fit_family == "orthogonal H-SSBP":
        return {
            "chart": "orthogonalBlock",
            "drift_ref": "orthogonalBlockDiagonalPolarStableMatrixParameter",
            "extra_gradient": """    <timeSeriesGradient id="grad.angles">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="ou.A.angles"/></parameter>
    </timeSeriesGradient>""",
            "extra_prior": '    <normalPrior id="ou.A.angles.prior" mean="0.0" stdev="0.25"><parameter idref="ou.A.angles"/></normalPrior>',
            "extra_likelihood_gradient_ref": '        <timeSeriesGradient idref="grad.angles"/>',
            "extra_prior_gradient": '        <gradient><distributionLikelihood idref="ou.A.angles.prior"/><parameter idref="ou.A.angles"/></gradient>',
            "extra_parameter": '        <parameter idref="ou.A.angles"/>',
            "extra_transform": """                <composedTransform>
                    <inner><affineTransform location="0.0" scale="1.5707963267948966" dim="10"/></inner>
                    <outer><transform type="fisherZ" dim="10"/></outer>
                </composedTransform>""",
            "extra_prior_ref": '            <normalPrior idref="ou.A.angles.prior"/>',
            "extra_log": '            <parameter idref="ou.A.angles"/>',
        }
    return {
        "chart": "dense",
        "drift_ref": "blockDiagonalPolarStableMatrixParameter",
        "extra_gradient": """    <timeSeriesGradient id="grad.rotation">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><matrixParameter idref="ou.A.rotation.raw"/></parameter>
    </timeSeriesGradient>""",
        "extra_prior": f"""    <invertibleMatrixShrinkagePrior id="ou.A.rotation.shrinkagePrior">
        <matrix><normalizedMatrixParameter idref="ou.A.rotation"/></matrix>
        <lambda>
            <parameter id="ou.A.rotation.shrinkagePrior.lambda" value="{fmt(shrinkage_lambda)}" lower="0.0"/>
        </lambda>
    </invertibleMatrixShrinkagePrior>""",
        "extra_likelihood_gradient_ref": '        <timeSeriesGradient idref="grad.rotation"/>',
        "extra_prior_gradient": '        <invertibleMatrixShrinkagePrior idref="ou.A.rotation.shrinkagePrior"/>',
        "extra_parameter": '        <matrixParameter idref="ou.A.rotation.raw"/>',
        "extra_transform": '                <transform type="none" dim="25"/>',
        "extra_prior_ref": '            <invertibleMatrixShrinkagePrior idref="ou.A.rotation.shrinkagePrior"/>',
        "extra_log": '            <matrixParameter idref="ou.A.rotation"/>',
    }


def xml_document(case: Case, truth: Truth, start: Start, fit_family: str,
                 label: str, observations: np.ndarray, times: np.ndarray,
                 iterations: int, shrinkage_lambda: float) -> str:
    info = fit_specific_xml(fit_family, shrinkage_lambda)
    lower_corr = np.full(ANGLE_COUNT, -0.999)
    upper_corr = np.full(ANGLE_COUNT, 0.999)
    theta_lower = np.full(BLOCK_COUNT, -math.pi / 4.0)
    theta_upper = np.full(BLOCK_COUNT, math.pi / 4.0)
    angle_lower = np.full(ANGLE_COUNT, -math.pi / 2.0)
    angle_upper = np.full(ANGLE_COUNT, math.pi / 2.0)
    diffusion_start = np.diag(DIFFUSION) * (1.0 + 0.05 * np.cos(np.arange(DIM)))
    angle_parameter = ""
    if fit_family == "orthogonal H-SSBP":
        angle_parameter = "\n" + parameter_xml("ou.A.angles", start.angles, lower=angle_lower, upper=angle_upper)
    return f"""<?xml version="1.0" standalone="yes"?>
<beast version="1.10">

    <!--
        K=5 final simulation battery: {case.panel}, {case.grid_tag}, replicate {case.replicate:03d}.
        The manuscript Hurwitz drift is -ou.A; BEAST uses ou.A as selection strength.
    -->

{parameter_xml("diffusion.diagonal", diffusion_start, lower=np.zeros(DIM))}
{parameter_xml("diffusion.offDiagonal", np.zeros(ANGLE_COUNT), lower=lower_corr, upper=upper_corr)}

    <compoundSymmetricMatrix id="diffusionMatrix" asCorrelation="true" isCholesky="true">
        <diagonal><parameter idref="diffusion.diagonal"/></diagonal>
        <offDiagonal><parameter idref="diffusion.offDiagonal"/></offDiagonal>
    </compoundSymmetricMatrix>

{parameter_xml("ou.A.scalar", start.scalar, lower=np.zeros(1))}
{parameter_xml("ou.A.rho", start.rho, lower=np.zeros(BLOCK_COUNT))}
{parameter_xml("ou.A.theta", start.theta, lower=theta_lower, upper=theta_upper)}
{parameter_xml("ou.A.t", start.t, lower=np.zeros(BLOCK_COUNT))}{angle_parameter}
{parameter_xml("ou.mu", MEAN)}
{rotation_xml(fit_family, start)}
{drift_xml(fit_family)}
{matrix_parameter_xml("ou.P0", truth.initial_cov)}

{matrix_parameter_xml("obs.H", np.eye(DIM))}

{matrix_parameter_xml("obs.R", OBSERVATION)}

    <ouProcessModel id="ou.process" stateDimension="5" selectionChart="{info["chart"]}">
        <driftMatrix><{info["drift_ref"]} idref="ou.A"/></driftMatrix>
        <diffusionMatrix><compoundSymmetricMatrix idref="diffusionMatrix"/></diffusionMatrix>
        <stationaryMean><parameter idref="ou.mu"/></stationaryMean>
        <initialCovariance><matrixParameter idref="ou.P0"/></initialCovariance>
    </ouProcessModel>

    <irregularTimeGrid id="series01.grid" times="{fmt_vec(times)}"/>

{observations_xml(observations)}
    <gaussianObservationModel id="series01.observationModel" observationDimension="5">
        <designMatrix><matrixParameter idref="obs.H"/></designMatrix>
        <noiseCovariance><matrixParameter idref="obs.R"/></noiseCovariance>
        <observations><matrixParameter idref="series01.observations"/></observations>
    </gaussianObservationModel>
    <timeSeriesModel id="series01.model">
        <latentProcess><ouProcessModel idref="ou.process"/></latentProcess>
        <observationModel><gaussianObservationModel idref="series01.observationModel"/></observationModel>
        <timeGrid><irregularTimeGrid idref="series01.grid"/></timeGrid>
    </timeSeriesModel>
    <timeSeriesLikelihood id="series01.likelihood"
                          forwardMode="canonical"
                          smootherMode="canonical"
                          gradientMode="canonicalAnalytical">
        <model><timeSeriesModel idref="series01.model"/></model>
    </timeSeriesLikelihood>

    <parallelTimeSeriesLikelihood id="parallel.series" threads="1">
        <timeSeriesLikelihood idref="series01.likelihood"/>
    </parallelTimeSeriesLikelihood>

    <timeSeriesGradient id="grad.diffusion.diagonal">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="diffusion.diagonal"/></parameter>
    </timeSeriesGradient>
    <timeSeriesGradient id="grad.diffusion.offDiagonal">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="diffusion.offDiagonal"/></parameter>
    </timeSeriesGradient>
    <timeSeriesGradient id="grad.scalar">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="ou.A.scalar"/></parameter>
    </timeSeriesGradient>
    <timeSeriesGradient id="grad.rho">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="ou.A.rho"/></parameter>
    </timeSeriesGradient>
    <timeSeriesGradient id="grad.theta">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="ou.A.theta"/></parameter>
    </timeSeriesGradient>
    <timeSeriesGradient id="grad.t">
        <likelihood><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
        <parameter><parameter idref="ou.A.t"/></parameter>
    </timeSeriesGradient>
{info["extra_gradient"]}

    <gammaPrior id="diffusion.diagonal.prior" shape="0.5" scale="0.5"><parameter idref="diffusion.diagonal"/></gammaPrior>
    <LKJCorrelationPrior id="diffusion.offDiagonal.prior" shapeParameter="1.0" dimension="5">
        <data><parameter idref="diffusion.offDiagonal"/></data>
    </LKJCorrelationPrior>
    <logNormalPrior id="ou.A.scalar.prior" mu="0" sigma="1"><parameter idref="ou.A.scalar"/></logNormalPrior>
    <logNormalPrior id="ou.A.rho.prior" mu="0" sigma="1"><parameter idref="ou.A.rho"/></logNormalPrior>
    <uniformPrior id="ou.A.theta.prior" lower="-0.7853981633974483" upper="0.7853981633974483"><parameter idref="ou.A.theta"/></uniformPrior>
    <logNormalPrior id="ou.A.t.prior" mu="0" sigma="1"><parameter idref="ou.A.t"/></logNormalPrior>
{info["extra_prior"]}

    <compoundGradient id="likelihoodGradient">
        <timeSeriesGradient idref="grad.diffusion.diagonal"/>
        <timeSeriesGradient idref="grad.diffusion.offDiagonal"/>
        <timeSeriesGradient idref="grad.scalar"/>
        <timeSeriesGradient idref="grad.rho"/>
        <timeSeriesGradient idref="grad.theta"/>
        <timeSeriesGradient idref="grad.t"/>
{info["extra_likelihood_gradient_ref"]}
    </compoundGradient>

    <compoundGradient id="priorGradient">
        <gradient><distributionLikelihood idref="diffusion.diagonal.prior"/><parameter idref="diffusion.diagonal"/></gradient>
        <gradient><LKJCorrelationPrior idref="diffusion.offDiagonal.prior"/></gradient>
        <gradient><distributionLikelihood idref="ou.A.scalar.prior"/><parameter idref="ou.A.scalar"/></gradient>
        <gradient><distributionLikelihood idref="ou.A.rho.prior"/><parameter idref="ou.A.rho"/></gradient>
        <gradient><distributionLikelihood idref="ou.A.theta.prior"/><parameter idref="ou.A.theta"/></gradient>
        <gradient><distributionLikelihood idref="ou.A.t.prior"/><parameter idref="ou.A.t"/></gradient>
{info["extra_prior_gradient"]}
    </compoundGradient>

    <jointGradient id="posteriorGradient">
        <compoundGradient idref="likelihoodGradient"/>
        <compoundGradient idref="priorGradient"/>
    </jointGradient>

    <compoundParameter id="inference.parameter">
        <parameter idref="diffusion.diagonal"/>
        <parameter idref="diffusion.offDiagonal"/>
        <parameter idref="ou.A.scalar"/>
        <parameter idref="ou.A.rho"/>
        <parameter idref="ou.A.theta"/>
        <parameter idref="ou.A.t"/>
{info["extra_parameter"]}
    </compoundParameter>

    <operators id="operators" optimizationSchedule="log">
        <hamiltonianMonteCarloOperator weight="1" nSteps="8" stepSize="0.004" mode="vanilla"
                                       drawVariance="1.0" autoOptimize="true"
                                       gradientCheckTolerance="1e-3" gradientCheckCount="10">
            <jointGradient idref="posteriorGradient"/>
            <compoundParameter idref="inference.parameter"/>
            <multivariateCompoundTransform id="compound.transform">
                <transform type="log" dim="5"/>
                <LKJTransform dimension="5"/>
                <transform type="log" dim="1"/>
                <transform type="log" dim="2"/>
                <composedTransform>
                    <inner><affineTransform location="0.0" scale="0.7853981633974483" dim="2"/></inner>
                    <outer><transform type="fisherZ" dim="2"/></outer>
                </composedTransform>
                <transform type="log" dim="2"/>
{info["extra_transform"]}
            </multivariateCompoundTransform>
        </hamiltonianMonteCarloOperator>
    </operators>

    <maximizeWrtParameter id="jointMAP"
                          startAtCurrentState="true"
                          printToScreen="true"
                          nIterations="{iterations}"
                          includeJacobian="false" lbfgsEpsilon="1e-30" lbfgsPast="0" lbfgsDelta="1e-30" lbfgsMaxLineSearch="200">
        <jointGradient idref="posteriorGradient"/>
        <multivariateCompoundTransform idref="compound.transform"/>
    </maximizeWrtParameter>

    <posterior id="posterior">
        <prior id="prior">
            <gammaPrior idref="diffusion.diagonal.prior"/>
            <LKJCorrelationPrior idref="diffusion.offDiagonal.prior"/>
            <logNormalPrior idref="ou.A.scalar.prior"/>
            <logNormalPrior idref="ou.A.rho.prior"/>
            <uniformPrior idref="ou.A.theta.prior"/>
            <logNormalPrior idref="ou.A.t.prior"/>
{info["extra_prior_ref"]}
        </prior>
        <likelihood id="likelihood"><parallelTimeSeriesLikelihood idref="parallel.series"/></likelihood>
    </posterior>

    <mcmc id="mcmc" chainLength="0" autoOptimize="true" fullEvaluation="0"
          operatorAnalysis="{label}.ops">
        <posterior idref="posterior"/>
        <operators idref="operators"/>
        <log id="screenLog" logEvery="1000">
            <column label="posterior" dp="4" width="12"><posterior idref="posterior"/></column>
            <column label="likelihood" dp="4" width="12"><likelihood idref="likelihood"/></column>
            <column label="scalar" sf="6" width="12"><parameter idref="ou.A.scalar"/></column>
            <column label="rho" sf="6" width="16"><parameter idref="ou.A.rho"/></column>
        </log>
        <log id="fileLog" logEvery="{LOG_EVERY}" fileName="{label}.log" overwrite="true">
            <posterior idref="posterior"/>
            <likelihood idref="likelihood"/>
            <prior idref="prior"/>
            <matrixParameter idref="diffusionMatrix"/>
            <parameter idref="diffusion.diagonal"/>
            <parameter idref="diffusion.offDiagonal"/>
            <parameter idref="ou.A.scalar"/>
            <parameter idref="ou.A.rho"/>
            <parameter idref="ou.A.theta"/>
            <parameter idref="ou.A.t"/>
{info["extra_log"]}
            <parameter idref="ou.mu"/>
            <parameter idref="ou.A"/>
        </log>
        <logCheckpoint id="checkpointFileLog" checkpointEvery="{CHECKPOINT_EVERY}"
                       checkpointFinal="0"
                       fileName="{label}.chkpt"
                       overwrite="true"/>
    </mcmc>

    <report>K=5 final simulation battery: {case.panel}, {case.grid_tag}, replicate {case.replicate:03d}.</report>

</beast>
"""


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "col", "value"])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                writer.writerow([row + 1, col + 1, fmt(matrix[row, col])])


def write_observations(path: Path, observations: np.ndarray, times: np.ndarray, case: Case, seed: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel_key", "grid_tag", "replicate", "simulation_seed", "time_index", "time", "dimension", "value"])
        for time_index, time in enumerate(times, start=1):
            for dim_index in range(DIM):
                writer.writerow([case.panel_key, case.grid_tag, case.replicate, seed, time_index, fmt(time), dim_index + 1, fmt(observations[time_index - 1, dim_index])])


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "task_id", "xml_template", "panel", "panel_key", "x_column", "x_value",
        "grid_tag", "replicate", "fit_family", "start_index", "simulation_seed",
        "start_seed", "beast_seed", "optimizer_iterations", "shrinkage_lambda",
        "truth_matrix_csv", "metadata_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_cases(replicates: int) -> list[Case]:
    cases = []
    for replicate in range(1, replicates + 1):
        for ratio in REAL_COMPLEX_RATIOS:
            cases.append(Case("Real/complex boundary", "real_complex_boundary", "t_ratio_to_boundary", ratio, f"ratio{token(ratio)}", replicate))
        for shear in SHEARS:
            cases.append(Case("Non-orthogonal shear", "nonorthogonal_shear", "shear_multiplier", shear, f"shear{token(shear)}", replicate))
        for coupling in JORDAN_COUPLINGS:
            cases.append(Case("Jordan(4,1) coupling", "jordan41_coupling", "jordan_coupling", coupling, f"coupling{token(coupling)}", replicate))
    return cases


def label(case: Case, fit_family: str, start_index: int, task_id: int) -> str:
    fit_slug = "orthogonal" if fit_family == "orthogonal H-SSBP" else "denseR"
    return f"{case.panel_key}_{case.grid_tag}_rep{case.replicate:03d}_{fit_slug}_start{start_index:02d}_task{task_id:03d}"


def prepare(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.out_dir} exists; pass --force to replace it")
        shutil.rmtree(args.out_dir)
    inputs = args.out_dir / "inputs"
    inputs.mkdir(parents=True)
    (args.out_dir / "output").mkdir()
    truth_dir = inputs / "truth_matrices"
    metadata_dir = inputs / "metadata"
    obs_dir = inputs / "observations"
    truth_dir.mkdir()
    metadata_dir.mkdir()
    obs_dir.mkdir()

    manifest = []
    task_id = 0
    for case_id, case in enumerate(build_cases(args.replicates), start=1):
        truth = truth_for_case(case)
        sim_seed = SIM_SEEDS[case.panel_key] + case.replicate
        observations, times = simulate_observations(truth, sim_seed)
        truth_rel = f"truth_matrices/true_selection_matrix_case{case_id:03d}.csv"
        meta_rel = f"metadata/case{case_id:03d}_{case.panel_key}_{case.grid_tag}_rep{case.replicate:03d}.json"
        obs_rel = f"observations/{case.panel_key}_{case.grid_tag}_rep{case.replicate:03d}.csv"
        write_matrix(inputs / truth_rel, truth.selection)
        write_observations(inputs / obs_rel, observations, times, case, sim_seed)
        (inputs / meta_rel).write_text(json.dumps({
            "panel": case.panel,
            "panel_key": case.panel_key,
            "x_column": case.x_column,
            "x_value": case.x_value,
            "grid_tag": case.grid_tag,
            "replicate": case.replicate,
            "simulation_seed": sim_seed,
            "dimension": DIM,
            "time_count": TIME_COUNT,
            "series_count": 1,
            "diffusion_diagonal": [float(value) for value in np.diag(DIFFUSION)],
            "observation_diagonal": [float(value) for value in np.diag(OBSERVATION)],
            **truth.metadata,
        }, indent=2, sort_keys=True) + "\n")
        for fit_family in ("orthogonal H-SSBP", "generic dense-R H-SSBP"):
            for start_index in range(1, args.starts + 1):
                task_id += 1
                start_seed = START_SEED_BASE + case_id * 100 + (0 if fit_family == "orthogonal H-SSBP" else 50) + start_index
                start = start_for(case, truth, fit_family, start_seed, args.start_sd)
                run_label = label(case, fit_family, start_index, task_id)
                xml_name = f"input_{task_id:04d}.xml"
                (inputs / xml_name).write_text(xml_document(
                    case, truth, start, fit_family, run_label, observations, times,
                    args.iterations, args.shrinkage_lambda
                ))
                manifest.append({
                    "task_id": f"{task_id:04d}",
                    "xml_template": xml_name,
                    "panel": case.panel,
                    "panel_key": case.panel_key,
                    "x_column": case.x_column,
                    "x_value": fmt(case.x_value),
                    "grid_tag": case.grid_tag,
                    "replicate": f"{case.replicate:03d}",
                    "fit_family": fit_family,
                    "start_index": f"{start_index:02d}",
                    "simulation_seed": sim_seed,
                    "start_seed": start_seed,
                    "beast_seed": FIT_SEED_BASE + task_id,
                    "optimizer_iterations": args.iterations,
                    "shrinkage_lambda": fmt(args.shrinkage_lambda) if fit_family != "orthogonal H-SSBP" else "",
                    "truth_matrix_csv": truth_rel,
                    "metadata_json": meta_rel,
                })
        if case_id % 25 == 0:
            print(f"prepared {case_id} data cases, {task_id} XML tasks")

    write_manifest(inputs / "task_manifest.tsv", manifest)
    (args.out_dir / "run_metadata.json").write_text(json.dumps({
        "description": "K=5 final MAP simulation battery",
        "replicates_per_grid_point": args.replicates,
        "starts_per_data_set_and_fit": args.starts,
        "optimizer_iterations": args.iterations,
        "relative_logposterior_stabilization_rule": "last-step relative improvement <= 1e-6",
        "fit_families": ["orthogonal H-SSBP", "generic dense-R H-SSBP"],
        "dense_R_normalization": "global",
        "dense_R_shrinkage_lambda": args.shrinkage_lambda,
        "total_tasks": task_id,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out_dir}")
    print(f"tasks={task_id}")


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
