"""
tabular_polygraph.fidelity.hif.sentinel
------------------------------------
Logical Sentinel Ensemble (LSE) — the categorical manifold auditor of HIF.

Discovers high-dependency "hub" columns and trains per-hub Random Forest
classifiers that learn categorical manifold laws.  Synthetic rows whose
manifold confidence falls below the learned floor are flagged as logical
ruptures.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

LSE_MIN_SAMPLES_LEAF = 5

__all__ = [
    "LSE_MIN_SAMPLES_LEAF",
    "ManifoldEncoder",
    "LogicalSentinelEnsemble",
]


class ManifoldEncoder:
    """Stateful Categorical-to-Ordinal projection with feature mapping."""

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: list[str] = []
        self.feature_map: dict[str, list[str]] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit encoder on reference manifold and build feature map."""
        if df.empty:
            return
        self.encoder.fit(df)
        self.feature_names = list(self.encoder.get_feature_names_out())

        self.feature_map = {}
        for original_col in df.columns:
            self.feature_map[original_col] = [
                name
                for name in self.feature_names
                if name.startswith(f"{original_col}_")
            ]
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project data into the reference subspace."""
        if not self.is_fitted:
            return pd.DataFrame(index=df.index)
        encoded = self.encoder.transform(df)
        return pd.DataFrame(encoded, columns=self.feature_names, index=df.index)


class LogicalSentinelEnsemble:
    """
    Neuro-Symbolic Integrity Oracle (LSE).
    Learns 'Manifold Laws' using Random Forest Sentinels on high-dependency hubs.
    """

    def __init__(
        self,
        top_n_hubs: int = 5,
        max_depth: int = 12,
        random_state: int = 42,
        confidence_percentile: float = 5.0,
    ) -> None:
        self.top_n_hubs = top_n_hubs
        self.max_depth = max_depth
        self.random_state = random_state
        self.confidence_percentile = confidence_percentile
        self.sentinels: dict[str, RandomForestClassifier] = {}
        self.hubs: list[str] = []
        self.confidence_floors: dict[str, float] = {}
        self.encoder = ManifoldEncoder()
        self.is_trained: bool = False

    def _discover_hubs(
        self,
        df: pd.DataFrame,
        x_encoded: pd.DataFrame,
        potential_hubs: list[str] | None = None,
    ) -> list[str]:
        """Discover 'Manifold Hubs' using predictive synergy (captures higher-order interactions)."""
        cols = potential_hubs if potential_hubs is not None else df.columns
        scores = {}

        for hub_col in cols:
            other_cols = [c for c in df.columns if c != hub_col]
            hub_features = []
            for col in other_cols:
                hub_features.extend(self.encoder.feature_map.get(col, []))

            if not hub_features:
                continue

            X = x_encoded[hub_features]
            y = df[hub_col].astype(str)

            n_unique = len(y.unique())
            if n_unique < 2:
                continue

            if n_unique > 50 and n_unique > len(y) * 0.15:
                continue

            clf = RandomForestClassifier(
                n_estimators=25,
                max_depth=8,
                min_samples_leaf=LSE_MIN_SAMPLES_LEAF,
                random_state=self.random_state,
                max_features="sqrt",
            )
            n_unique = len(y.unique())
            n_splits = min(5, n_unique)
            if n_splits < 2:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module="sklearn.model_selection._split",
                    )
                    cv_scores = cross_val_score(
                        clf, X, y, cv=n_splits, scoring="accuracy"
                    )
                scores[hub_col] = float(cv_scores.mean())
            except (ValueError, TypeError):
                continue

        sorted_hubs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [h[0] for h in sorted_hubs[: self.top_n_hubs]]

    def fit(
        self,
        df: pd.DataFrame,
        hif_epochs: int = 10,
        verbose: bool = True,
        x_precomputed: pd.DataFrame | None = None,
        potential_hubs: list[str] | None = None,
    ):
        """Train Sentinels using stateful manifold projection.

        Parameters
        ----------
        df : pd.DataFrame
            Binned real data (categorical hub candidates).
        hif_epochs : int
            Multiplier for the number of Random Forest trees per sentinel:
            ``n_trees = max(10, hif_epochs * 10)``.  This is NOT training
            epochs — RF classifiers are non-iterative.
        verbose : bool
            Print progress messages.
        x_precomputed : pd.DataFrame or None
            Pre-encoded categorical manifold (OneHot).
        potential_hubs : list[str] or None
            Subset of columns to consider as hub candidates.
        """
        if x_precomputed is not None:
            x_encoded = x_precomputed
            self.encoder.feature_names = x_encoded.columns.tolist()
            self.encoder.is_fitted = True
            for original_col in df.columns:
                self.encoder.feature_map[original_col] = [
                    name
                    for name in self.encoder.feature_names
                    if name.startswith(f"{original_col}_")
                ]
        else:
            self.encoder.fit(df)
            x_encoded = self.encoder.transform(df)

        if verbose:
            logger.debug("Discovering Synergistic Manifold Hubs...")
        self.hubs = self._discover_hubs(df, x_encoded, potential_hubs=potential_hubs)

        if verbose:
            logger.debug(f"Hubs Selected: {self.hubs}")

        for hub_col in self.hubs:
            other_cols = [c for c in df.columns if c != hub_col]
            hub_features = []
            for col in other_cols:
                hub_features.extend(self.encoder.feature_map.get(col, []))

            X = x_encoded[hub_features]
            y = df[hub_col]

            if verbose:
                logger.debug(
                    f"Training Sentinel for Hub '{hub_col}' ({X.shape[1]} features)..."
                )

            n_trees = max(10, hif_epochs * 10)
            clf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=self.max_depth,
                min_samples_leaf=LSE_MIN_SAMPLES_LEAF,
                random_state=self.random_state,
                max_features="log2",
                oob_score=True,
            )
            clf.fit(X, y)
            self.sentinels[hub_col] = clf
            if verbose:
                logger.debug(f"Training Sentinel for Hub '{hub_col}' complete.")

            if hasattr(clf, "oob_decision_function_"):
                oob_probs = clf.oob_decision_function_
            else:
                oob_probs = clf.predict_proba(X)

            classes = clf.classes_
            y_str = y.astype(str).values
            probs_true = np.zeros(len(y))
            for idx, cls in enumerate(classes):
                mask = y_str == str(cls)
                if mask.any():
                    probs_true[mask] = oob_probs[mask, idx]

            self.confidence_floors[hub_col] = max(
                float(np.nanpercentile(probs_true, self.confidence_percentile)), 0.01
            )

        self.is_trained = True

    def audit(
        self, df: pd.DataFrame, x_precomputed: pd.DataFrame | None = None
    ) -> tuple[float, np.ndarray, dict[str, Any]]:
        """Audit synthetic rows for 'Logical Ruptures' using reference manifold."""
        if not self.is_trained:
            raise ValueError("LogicalSentinelEnsemble must be fitted before audit().")

        row_penalties = np.zeros(len(df))
        traces = []

        if x_precomputed is not None:
            x_encoded = x_precomputed
        else:
            x_encoded = self.encoder.transform(df)

        for hub_col in self.hubs:
            clf = self.sentinels[hub_col]
            X = x_encoded.reindex(columns=clf.feature_names_in_, fill_value=0)

            probs = clf.predict_proba(X)
            classes = clf.classes_
            observed_values = df[hub_col].astype(str).values

            probs_observed = np.zeros(len(df))
            for class_idx, class_val in enumerate(classes):
                mask = observed_values == str(class_val)
                if mask.any():
                    probs_observed[mask] = probs[mask, class_idx]

            floor = self.confidence_floors[hub_col]

            if floor > 1e-6:
                penalty = np.clip((floor - probs_observed) / floor, 0, 1)
            else:
                penalty = np.zeros(len(df))

            row_penalties = 1.0 - (1.0 - row_penalties) * (1.0 - penalty)

            ruptures = penalty > 0.5
            if ruptures.any():
                traces.append(
                    {
                        "column": hub_col,
                        "violations": int(ruptures.sum()),
                        "mean_prob": float(probs_observed[ruptures].mean()),
                    }
                )

        hif_score_val = 1.0 - row_penalties.mean()
        avg_floor = (
            float(np.mean(list(self.confidence_floors.values())))
            if self.confidence_floors
            else 1.0
        )
        return (
            float(hif_score_val),
            row_penalties,
            {
                "traces": traces,
                "confidence_floors": self.confidence_floors,
                "avg_floor": avg_floor,
            },
        )
