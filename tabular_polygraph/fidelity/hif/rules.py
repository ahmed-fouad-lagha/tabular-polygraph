"""
tabular_polygraph.fidelity.hif.rules
---------------------------------
FP-Growth-style implication rule mining and violation scoring for HIF.

Mines high-confidence antecedent -> consequent rules from real data and
checks which synthetic rows violate them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from .binning import (
    RULE_QUANTIZATION_BINS,
    apply_binning,
    canonicalize_code_columns,
    fit_binning,
)

logger = logging.getLogger(__name__)

MAX_RULE_CANDIDATES = 10000
ANTE_JOIN = " & "

__all__ = [
    "MAX_RULE_CANDIDATES",
    "mine_implication_rules",
    "rule_violation_score",
]


def mine_implication_rules(
    real: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.95,
    min_support: float = 0.005,
    max_rules: int = 25,
    min_lift: float = 1.0,
    max_antecedents: int = 2,
    random_state: int | None = None,
    pre_binned: bool = False,
) -> list[dict[str, Any]]:
    n_rows = len(real)
    if n_rows == 0:
        return []
    min_support_count = max(1, int(np.ceil(min_support * n_rows)))
    rules: list[dict[str, Any]] = []
    cat = pd.DataFrame(index=real.index)
    for col in columns:
        col_data = real[col]
        if pre_binned:
            cat[col] = col_data.astype(str)
            continue
        if pd.api.types.is_object_dtype(col_data) or isinstance(
            col_data.dtype, pd.CategoricalDtype
        ):
            cat[col] = col_data.astype(str)
            continue

        n_unique = col_data.nunique()
        if pd.api.types.is_numeric_dtype(col_data) and n_unique > 50:
            try:
                quantized = pd.qcut(
                    col_data, RULE_QUANTIZATION_BINS, labels=None, duplicates="drop"
                )
                cat[col] = quantized.astype(str)
            except Exception:
                cat[col] = col_data.astype(str)
        else:
            cat[col] = col_data.astype(str)
    frequent_items: dict[tuple[str, str], int] = {}
    for col in columns:
        counts = cat[col].value_counts()
        for val, count in counts.items():
            if count >= min_support_count:
                frequent_items[(col, str(val))] = int(count)
    item_masks: dict[tuple[str, str], np.ndarray] = {}
    for col, val in frequent_items.keys():
        item_masks[(col, val)] = cat[col].values == val
    frequent_sets_by_size: dict[int, list[tuple[tuple[str, str], ...]]] = {
        1: [(item,) for item in frequent_items.keys()]
    }
    support_counts: dict[tuple[tuple[str, str], ...], int] = {
        (item,): count for item, count in frequent_items.items()
    }
    max_k = max_antecedents + 1
    for k in range(2, max_k + 1):
        prev_frequent = frequent_sets_by_size.get(k - 1, [])
        if not prev_frequent:
            break
        candidates_set = set()
        prefix_map: dict[tuple[tuple[str, str], ...], list[tuple[str, str]]] = {}
        for itemset in prev_frequent:
            prefix = itemset[:-1]
            last_item = itemset[-1]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(last_item)

        for prefix, items in prefix_map.items():
            feature_groups: dict[str, list[tuple[str, str]]] = {}
            for item in items:
                feat = item[0]
                if feat not in feature_groups:
                    feature_groups[feat] = []
                feature_groups[feat].append(item)

            feat_list = list(feature_groups.keys())
            for i in range(len(feat_list)):
                for j in range(i + 1, len(feat_list)):
                    for item_a in feature_groups[feat_list[i]]:
                        for item_b in feature_groups[feat_list[j]]:
                            cand = tuple(sorted(list(prefix) + [item_a, item_b]))
                            candidates_set.add(cand)
        candidates = sorted(candidates_set)
        if not candidates:
            break

        if len(candidates) > MAX_RULE_CANDIDATES:
            rng = check_random_state(random_state)
            indices = rng.choice(len(candidates), MAX_RULE_CANDIDATES, replace=False)
            candidates = [candidates[i] for i in sorted(indices)]

        current_frequent = []
        for cand in candidates:
            mask = item_masks[cand[0]]
            for i in range(1, len(cand)):
                mask = mask & item_masks[cand[i]]

            count = int(mask.sum())
            if count >= min_support_count:
                support_counts[cand] = count
                current_frequent.append(cand)

                for i in range(len(cand)):
                    consequent_item = cand[i]
                    antecedent_items = tuple(cand[:i] + cand[i + 1 :])

                    ant_count = support_counts.get(antecedent_items)
                    if ant_count is None or ant_count == 0:
                        continue

                    confidence = count / ant_count
                    if confidence >= min_confidence:
                        consequent_support = support_counts[(consequent_item,)] / n_rows
                        lift = confidence / consequent_support
                        if lift >= min_lift:
                            antecedents = [
                                {"feature": f, "value": v} for f, v in antecedent_items
                            ]
                            rules.append(
                                {
                                    "antecedents": antecedents,
                                    "antecedent_repr": ANTE_JOIN.join(
                                        f"{a['feature']}={a['value']}"
                                        for a in antecedents
                                    ),
                                    "consequent_feature": consequent_item[0],
                                    "consequent_value": consequent_item[1],
                                    "support": round(count / n_rows, 4),
                                    "confidence": round(confidence, 4),
                                    "lift": round(lift, 4),
                                    "support_count": count,
                                    "antecedent_count": support_counts[
                                        antecedent_items
                                    ],
                                    "antecedent_feature": antecedents[0]["feature"]
                                    if len(antecedents) == 1
                                    else None,
                                }
                            )
        if not current_frequent:
            break
        frequent_sets_by_size[k] = current_frequent
    rules.sort(
        key=lambda x: (
            x["confidence"],
            x["lift"],
            x["support"],
            x["antecedent_repr"],
            x["consequent_feature"],
            x["consequent_value"],
        ),
        reverse=True,
    )
    return rules[:max_rules]


def rule_violation_score(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    min_confidence: float = 0.95,
    min_support: float = 0.005,
    max_rules: int = 25,
    min_lift: float = 1.0,
    max_antecedents: int = 2,
    max_violation_examples: int = 20,
    random_state: int | None = None,
    pre_binned: bool = False,
) -> dict[str, Any]:
    rule_diagnostics: list[dict[str, Any]] = []
    violation_examples: list[dict[str, Any]] = []

    if not columns or max_rules < 1:
        return {
            "rule_violation_rate": 0.0,
            "total_rule_hits": 0,
            "num_rules_mined": 0,
            "num_rows_with_violations": 0,
            "rows_evaluated": len(synthetic),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    if pre_binned:
        real_f, syn_f = real, synthetic
    else:
        bin_edges = fit_binning(real, columns)
        real_f = apply_binning(real, columns, bin_edges)
        syn_f = apply_binning(synthetic, columns, bin_edges)
        real_f, syn_f = canonicalize_code_columns(real_f, syn_f, columns)
    rules = mine_implication_rules(
        real_f,
        columns=columns,
        min_confidence=min_confidence,
        min_support=min_support,
        max_rules=max_rules,
        min_lift=min_lift,
        max_antecedents=max_antecedents,
        random_state=random_state,
        pre_binned=pre_binned,
    )

    if not rules:
        return {
            "rule_violation_rate": 0.0,
            "total_rule_hits": 0,
            "num_rules_mined": 0,
            "num_rows_with_violations": 0,
            "rows_evaluated": len(syn_f),
            "top_violated_rules": [],
            "violation_examples": [],
        }

    row_violation_mask = np.zeros(len(syn_f), dtype=bool)
    total_violations = 0

    for rule in rules:
        ants = rule["antecedents"]
        ant_mask = pd.Series(True, index=syn_f.index)
        for ant in ants:
            ant_mask &= syn_f[ant["feature"]].astype(str).eq(str(ant["value"]))

        if not ant_mask.any():
            continue

        violates = ant_mask & (
            ~syn_f[rule["consequent_feature"]].astype(str).eq(rule["consequent_value"])
        )
        row_violation_mask |= violates.to_numpy()
        v_count = int(violates.sum())
        total_violations += v_count
        if v_count > 0:
            rule_diagnostics.append({**rule, "violation_count": v_count})
            for ridx in syn_f.index[violates][:3]:
                if len(violation_examples) >= max_violation_examples:
                    break
                violation_examples.append(
                    {
                        "row_index": str(ridx),
                        "antecedent": rule["antecedent_repr"],
                        "expected": f"{rule['consequent_feature']}={rule['consequent_value']}",
                        "actual": f"{rule['consequent_feature']}={syn_f.loc[ridx, rule['consequent_feature']]}",
                    }
                )

    rule_diagnostics.sort(key=lambda d: d["violation_count"], reverse=True)
    return {
        "rule_violation_rate": round(row_violation_mask.sum() / len(syn_f), 4),
        "total_rule_hits": int(total_violations),
        "num_rules_mined": int(len(rules)),
        "num_rows_with_violations": int(row_violation_mask.sum()),
        "rows_evaluated": int(len(syn_f)),
        "row_violation_mask": row_violation_mask.astype(float),
        "top_violated_rules": rule_diagnostics[:10],
        "violation_examples": violation_examples,
    }
