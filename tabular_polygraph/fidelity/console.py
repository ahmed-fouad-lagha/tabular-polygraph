from __future__ import annotations

from typing import Any

PLACEHOLDER = "\u2014"


class ConsoleFormatter:
    def __init__(self, width: int = 60):
        self.width = width

    def format(self, report: dict[str, Any]) -> str:
        lines = []
        s = report.get("summary", {})

        lines.append("=" * self.width)
        lines.append("  FIDELITY REPORT")
        lines.append("=" * self.width)
        lines.append(f"  Dataset type    : {report.get('dataset_type', PLACEHOLDER)}")
        lines.append(
            f"  Rows (real/syn) : {s.get('rows_real', '?')} / {s.get('rows_synthetic', '?')}"
        )
        lines.append("")
        lines.append("-" * self.width)
        lines.append(
            f"  Moment matching : {s.get('moment_matching_score') or 0:>6.2f}%"
        )
        lines.append(f"  KS distribution : {s.get('ks_score') or 0:>6.2f}%")
        lines.append(f"  Joint distance  : {s.get('joint_score') or 0:>6.2f}%")

        ap = s.get("alpha_precision")
        br = s.get("beta_recall")
        au = s.get("authenticity")
        if ap is not None:
            lines.append(f"  Alpha-precision : {ap:>6.3f}")
            lines.append(f"  Beta-recall     : {br:>6.3f}")
            lines.append(f"  Authenticity    : {au:>6.3f}")
        l_score = s.get("logic_score")
        if l_score is not None:
            lines.append(f"  HIF (structural): {l_score:>6.2f}%")
        lines.append("-" * self.width)
        lines.append("")

        sf = report.get("stylized_facts", {})
        lines.append("  Stylized facts:")
        if sf.get("applicable", True):
            lines.append(f"    Mean score  : {sf.get('mean_score', PLACEHOLDER)}%")
            lines.append(f"    Columns     : {sf.get('columns_tested', PLACEHOLDER)}")
        else:
            lines.append(f"    {sf.get('note', 'Not evaluated.')}")
        lines.append("")

        mm_cols = report.get("moment_matching", {}).get("column_scores", {})
        if mm_cols:
            lines.append("")
            lines.append("  Per-column moment matching scores:")
            for col, sc in mm_cols.items():
                bar = "\u2588" * int(sc / 5) + "\u2591" * (20 - int(sc / 5))
                lines.append(f"    {col:<26} {bar}  {sc}%")

        ks_cols = report.get("distribution_fit", {}).get("column_scores", {})
        if ks_cols:
            lines.append("")
            lines.append("  Per-column KS distribution scores:")
            for col, sc in ks_cols.items():
                bar = "\u2588" * int(sc / 5) + "\u2591" * (20 - int(sc / 5))
                lines.append(f"    {col:<26} {bar}  {sc}%")

        ds = report.get("downstream")
        if ds:
            lines.append("")
            lines.append("  Downstream (TSTR):")
            if ds.get("status") == "skipped":
                lines.append(f"    Skipped: {ds.get('reason')}")
            elif ds.get("tstr_score") is not None:
                lines.append(f"    Target  : {ds.get('target_col')}")
                lines.append(
                    f"    Metric  : {ds.get('metric')} | TSTR {ds.get('tstr_score')} | TRR {ds.get('trr_score')}"
                )
                lines.append(f"    Ratio   : {ds.get('ratio')}")

        lg = report.get("logical", {})
        lines.append("")
        lines.append("  Logical:")
        if lg.get("error"):
            lines.append(f"    Status         : ERROR ({lg['error']})")
        else:
            lines.append(
                f"    Unified violation rate : {lg.get('hif_violation_rate_pct', PLACEHOLDER)}%"
            )
            lines.append(
                f"    NIC (Continuous) rate  : {lg.get('nic_violation_rate_pct', PLACEHOLDER)}%"
            )
            lines.append(
                f"    Rule violation rate    : {lg.get('rule_violation_rate_pct', PLACEHOLDER)}%"
            )
            lines.append(
                f"    Mean penalty           : {lg.get('mean_penalty_pct', PLACEHOLDER)}%"
            )
            lines.append(
                f"    Noise floor threshold  : {lg.get('violation_threshold', PLACEHOLDER)}"
            )
            lines.append(
                f"    Violations found       : {lg.get('num_hif_violations', PLACEHOLDER)} (rules mined: {lg.get('num_rules_mined', PLACEHOLDER)})"
            )

            top_rules = lg.get("top_violated_rules", [])
            if top_rules:
                lines.append("    Top violated rules:")
                for rule in top_rules[:5]:
                    ant = rule.get("antecedent_repr") or (
                        f"{rule.get('antecedent_feature')}={rule.get('antecedent_value')}"
                        if rule.get("antecedent_feature") is not None
                        else "(unknown antecedent)"
                    )
                    lines.append(
                        f"      IF {ant} "
                        f"THEN {rule.get('consequent_feature')}={rule.get('consequent_value')} "
                        f"| conf={rule.get('confidence')} lift={rule.get('lift', PLACEHOLDER)} "
                        f"support={rule.get('support')} violations={rule.get('violation_count')}"
                    )

            examples = lg.get("violation_examples", [])
            if examples:
                lines.append("    Example violating rows:")
                for ex in examples[:5]:
                    lines.append(
                        f"      row={ex.get('row_index')} | {ex.get('antecedent')} | "
                        f"expected {ex.get('expected')} | actual {ex.get('actual')}"
                    )

        lines.append("")
        lines.append(f"  Computed in {s.get('elapsed_seconds', '?')}s")
        lines.append("=" * self.width)

        # Privacy section
        pv = report.get("privacy")
        if pv:
            lines.append("")
            lines.append("=" * self.width)
            lines.append("  PRIVACY (TAMIS Audit)")
            lines.append("=" * self.width)
            v = pv.get("verdict", {})
            overall = v.get("overall_risk", "—").upper()
            icon = "[OK]" if overall in ("VERY_LOW", "LOW") else "[FAIL]"
            lines.append(f"  {icon} Overall risk: {overall}")

            ec = pv.get("exact_copies", {})
            lines.append(
                f"  Exact copies      : {ec.get('count', '—')}  [{ec.get('risk_level', '—')}]"
            )

            mi = pv.get("membership_inference", {})
            lines.append(
                f"  Membership inf.   : AUC={mi.get('attack_auc', '—')}  [{mi.get('risk_level', '—')}]"
            )
            lines.append(f"    {mi.get('interpretation', '')}")

            so = pv.get("singling_out", {})
            lines.append(
                f"  Singling-out      : rate={so.get('singling_out_rate', '—')}  [{so.get('risk_level', '—')}]"
            )

            lk = pv.get("linkability", {})
            lines.append(
                f"  Linkability       : rate={lk.get('linkability_rate', '—')}  [{lk.get('risk_level', '—')}]"
            )
            lines.append(
                f"    lift={lk.get('lift_over_baseline_pct', '—')}% over baseline"
            )

            lines.append("")
            lines.append(f"  Recommendation: {v.get('recommendation', '—')}")
            lines.append(f"  Elapsed: {v.get('elapsed_seconds', '—')}s")
            lines.append("=" * self.width)

        return "\n".join(lines)
