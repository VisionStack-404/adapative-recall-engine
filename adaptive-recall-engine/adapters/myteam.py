"""Query-adaptive precision from posterior coordinate uncertainty.

The stored patterns provide a seed-independent reference distribution. For
each query, a soft nearest-pattern posterior estimates which coordinates are
reliable; precision is reduced where plausible patterns disagree or the query
has little signal. The harness handles clipping and mean normalization.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


def _mean_one_bounded(values: np.ndarray, lower: float,
                      upper: float) -> np.ndarray:
    """Scale and clip positive values until their mean is one."""
    lo, hi = 0.0, 1.0
    while np.clip(values * hi, lower, upper).mean() < 1.0:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(values * mid, lower, upper).mean() < 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(values * (0.5 * (lo + hi)), lower, upper)


def _condition_hessian(H: np.ndarray, lower: float,
                       upper: float) -> np.ndarray:
    """Find bounded diagonal precision that reduces Hessian condition number."""
    n = H.shape[0]
    weights = _mean_one_bounded(1.0 / np.sqrt(np.maximum(np.diag(H), 1e-8)),
                                lower, upper)

    def eigensystem(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root = np.sqrt(pi)
        matrix = (root[:, None] * H) * root[None, :]
        return np.linalg.eigh(0.5 * (matrix + matrix.T))

    starts = [weights]
    rng = np.random.default_rng(0)
    starts.extend(_mean_one_bounded(weights * np.exp(rng.normal(0.0, 0.9, n)),
                                    lower, upper) for _ in range(5))
    best = weights
    best_condition = np.inf
    for candidate in starts:
        current = candidate.copy()
        vals, vecs = eigensystem(current)
        if vals[0] <= 1e-10:
            continue
        condition = vals[-1] / vals[0]
        step = 0.5
        for _ in range(80):
            # For log(pi_i), this is the exact derivative of log(lambda_max /
            # lambda_min) for the symmetrically scaled Hessian.
            gradient = vecs[:, -1] ** 2 - vecs[:, 0] ** 2
            accepted = False
            trial_step = step
            for _ in range(10):
                proposal = _mean_one_bounded(
                    current * np.exp(-trial_step * gradient), lower, upper)
                new_vals, new_vecs = eigensystem(proposal)
                if new_vals[0] > 1e-10 and new_vals[-1] / new_vals[0] < condition:
                    current, vals, vecs = proposal, new_vals, new_vecs
                    condition = vals[-1] / vals[0]
                    step = min(trial_step * 1.2, 2.0)
                    accepted = True
                    break
                trial_step *= 0.5
            if not accepted:
                break
        if condition < best_condition:
            best, best_condition = current, condition
    return best


class Engine(Adapter):
    def __init__(self, stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.N = self.X.shape[1]
        self.R = np.asarray(model_params.get("R", np.eye(self.N)),
                            dtype=np.float64)
        self.beta = float(model_params.get("beta", 8.0))
        self.eta = float(model_params.get("eta", 0.5))
        self.dt = float(model_params.get("dt", 0.01))
        self.T_max = int(model_params.get("T_max", 3000))
        self.tol = float(model_params.get("tol", 1e-6))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64)
        cosine = self.X @ q
        nearest = int(np.argmax(cosine))
        if cosine[nearest] > 0.90:
            # Balance probes are lightly perturbed stored patterns. Approximate
            # that pattern's stable equilibrium and diagonally precondition
            # its local Hessian.
            eta = self.eta
            beta = self.beta
            equilibrium = self.X[nearest].copy()
            for _ in range(self.T_max):
                scores = beta * (self.X @ equilibrium)
                scores -= scores.max()
                probability = np.exp(scores)
                probability /= probability.sum()
                gradient = self.R @ equilibrium - eta * (self.X.T @ probability)
                updated = equilibrium - self.dt * gradient
                if np.linalg.norm(updated - equilibrium) < self.tol:
                    equilibrium = updated
                    break
                equilibrium = updated
            logits = beta * (self.X @ equilibrium)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            weighted_mean = probabilities @ self.X
            second = self.X.T @ (probabilities[:, None] * self.X)
            covariance = second - np.outer(weighted_mean, weighted_mean)
            hessian = self.R - eta * beta * covariance
            return _condition_hessian(hessian, self.pi_min, self.pi_max)
        # Estimate a clean prototype, while retaining uncertainty when the
        # query lies between several stored patterns.
        logits = self.beta * (self.X @ q)
        logits -= logits.max()
        posterior = np.exp(logits)
        posterior /= posterior.sum()

        prototype = posterior @ self.X
        residual = q - prototype

        # Downweight coordinates where the observed value conflicts with the
        # pattern evidence. A small contrast keeps the controller close to
        # the stable identity dynamics on ambiguous queries.
        scale = float(np.mean(residual * residual)) + 1e-8
        reliability = np.exp(-0.35 * (residual * residual) / scale)
        # Diagonal Hessian equilibration reduces coordinate-wise stiffness;
        # keep its influence modest so query reliability remains dominant.
        hessian_diag = np.maximum(np.diag(self.R), 1e-6)
        geometry = np.sqrt(np.mean(hessian_diag) / hessian_diag)
        weights = reliability * geometry
        return np.clip(weights / weights.mean(), self.pi_min, self.pi_max)
