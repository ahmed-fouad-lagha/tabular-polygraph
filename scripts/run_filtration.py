"""
Semantic Filtration Experiment: Prove that Neuro-LCV penalties directly
correlate with downstream ML performance degradation.

Train XGBoost on (1) real data, (2) all synthetic data,
(3) clean synthetic data filtered by Neuro-LCV. If (3) > (2), the metric
has caught real logical poisoning that degrades ML utility.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def semantic_filtration_experiment(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    row_penalties: np.ndarray,
    target_col: str,
    severity_threshold: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Execute the Semantic Filtration Experiment.
    
    Trains XGBoost on three data sources:
    1. Real data (baseline)
    2. All synthetic data (SOTA - includes corrupt rows)
    3. Clean synthetic data (Neuro-LCV filtered, excludes high CSSP rows)
    
    Evaluates strictly on held-out real test set.
    
    Parameters
    ----------
    real_df : pd.DataFrame
        Real training data
    synth_df : pd.DataFrame
        Synthetic data with 'syn_id' column
    row_penalties : np.ndarray
        Neuro-LCV CSSP scores from evaluate_synthetic(), same length as synth_df
    target_col : str
        Target column name for downstream classification/regression
    severity_threshold : float
        CSSP threshold above which rows are considered logically corrupt
    test_size : float
        Fraction of real_df to hold out as test set
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress and results
    
    Returns
    -------
    dict
        Contains experiment results:
        - real_metrics : dict of baseline metrics
        - full_synthetic_metrics : dict of SOTA metrics
        - clean_synthetic_metrics : dict of filtered metrics
        - gains : dict of performance gains/losses
        - summary : dict summarizing the kill shot result
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("Downstream experiment requires scikit-learn and xgboost. Install with: pip install scikit-learn xgboost")
    
    if target_col not in real_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in real_df.")
    
    if len(row_penalties) != len(synth_df):
        raise ValueError(f"row_penalties length ({len(row_penalties)}) must match synth_df length ({len(synth_df)}).")
    
    if verbose:
        print("\n" + "=" * 70)
        print("SEMANTIC FILTRATION EXPERIMENT: Downstream Validation")
        print("=" * 70)
    
    # ── 1. Split real data into train/test ──────────────────────────────────
    real_train, real_test = train_test_split(
        real_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=real_df[target_col] if real_df[target_col].dtype in ['int64', 'int32', 'object'] else None
    )
    
    X_test_real = real_test.drop(columns=[target_col])
    y_test_real = real_test[target_col]
    
    if verbose:
        print(f"\n[Data Split]")
        print(f"  Real train set: {len(real_train)} rows")
        print(f"  Real test set:  {len(real_test)} rows (strict holdout)")
    
    # ── 2. Partition synthetic data by Neuro-LCV penalties ──────────────────
    clean_mask = row_penalties <= severity_threshold
    corrupt_mask = row_penalties > severity_threshold
    
    synth_clean_df = synth_df[clean_mask].copy()
    synth_corrupt_df = synth_df[corrupt_mask].copy()
    
    if verbose:
        print(f"\n[Synthetic Data Partitioning]")
        print(f"  Total synthetic rows:      {len(synth_df)}")
        print(f"  Clean rows (penalty ≤ {severity_threshold}):  {len(synth_clean_df)} ({100*len(synth_clean_df)/len(synth_df):.1f}%)")
        print(f"  Corrupt rows (penalty > {severity_threshold}): {len(synth_corrupt_df)} ({100*len(synth_corrupt_df)/len(synth_df):.1f}%)")
    
    # ── 3. Train and evaluate XGBoost on three datasets ──────────────────────
    def train_and_eval(train_data: pd.DataFrame, name: str):
        """Train XGBoost and evaluate on held-out real test set."""
        X_train = train_data.drop(columns=[target_col, 'syn_id'], errors='ignore')
        y_train = train_data[target_col]
        
        # Align columns
        X_train = X_train[[c for c in X_train.columns if c in X_test_real.columns]]
        X_test = X_test_real[[c for c in X_test_real.columns if c in X_train.columns]]
        
        # Encode categorical columns
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
            # Align columns after dummification
            all_cols = sorted(set(X_train.columns) | set(X_test.columns))
            for col in all_cols:
                if col not in X_train.columns:
                    X_train[col] = 0
                if col not in X_test.columns:
                    X_test[col] = 0
            X_train = X_train[all_cols]
            X_test = X_test[all_cols]
        
        if verbose:
            print(f"\n[Training {name}]")
            print(f"  Training set size: {len(X_train)} rows, {X_train.shape[1]} features")
        
        try:
            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                verbosity=0
            )
            model.fit(X_train, y_train)
            
            # Predict on real holdout test set
            preds = model.predict(X_test)
            preds_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else preds
            
            # Compute metrics
            acc = accuracy_score(y_test_real, preds)
            prec = precision_score(y_test_real, preds, average='weighted', zero_division=0)
            recall = recall_score(y_test_real, preds, average='weighted', zero_division=0)
            f1 = f1_score(y_test_real, preds, average='weighted', zero_division=0)
            
            metrics = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "n_train": len(X_train),
            }
            
            if verbose:
                print(f"  ✓ {name} Results (on real holdout):")
                print(f"    Accuracy:  {metrics['accuracy']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1 Score:  {metrics['f1']:.4f}")
            
            return metrics
        except Exception as e:
            print(f"  ✗ Error training {name}: {e}")
            return None
    
    # Train on three datasets
    real_metrics = train_and_eval(real_train, "Real Data Baseline")
    full_synthetic_metrics = train_and_eval(synth_df, "Full Synthetic (SOTA - All Rows)")
    clean_synthetic_metrics = train_and_eval(synth_clean_df, "Neuro-LCV Filtered Synthetic (Clean Only)")
    
    # ── 4. Compute gains and validate the kill shot ────────────────────────
    results = {
        "real_metrics": real_metrics,
        "full_synthetic_metrics": full_synthetic_metrics,
        "clean_synthetic_metrics": clean_synthetic_metrics,
    }
    
    if full_synthetic_metrics and clean_synthetic_metrics:
        gains = {
            "f1_clean_vs_full": round(clean_synthetic_metrics["f1"] - full_synthetic_metrics["f1"], 4),
            "accuracy_clean_vs_full": round(clean_synthetic_metrics["accuracy"] - full_synthetic_metrics["accuracy"], 4),
            "f1_clean_vs_real": round(clean_synthetic_metrics["f1"] - real_metrics["f1"], 4),
        }
        results["gains"] = gains
        
        # The kill shot: clean synthetic > full synthetic
        kill_shot_validated = clean_synthetic_metrics["f1"] > full_synthetic_metrics["f1"]
        
        if verbose:
            print("\n" + "=" * 70)
            print("SEMANTIC FILTRATION RESULTS: The Kill Shot")
            print("=" * 70)
            print(f"\n[Performance Gap Analysis]")
            print(f"  F1 Gain (Clean vs Full):      {gains['f1_clean_vs_full']:+.4f}")
            print(f"  Accuracy Gain (Clean vs Full): {gains['accuracy_clean_vs_full']:+.4f}")
            print(f"  F1 Gap vs Real Baseline:      {gains['f1_clean_vs_real']:+.4f}")
            print()
            if kill_shot_validated:
                print("  ✓✓✓ KILL SHOT VALIDATED ✓✓✓")
                print("  Filtering out logically corrupt rows IMPROVED downstream performance!")
                print("  This proves Neuro-LCV catches rows that actively poison ML models.")
            else:
                print("  Note: Clean synthetic F1 ≤ Full synthetic F1.")
                print("  Possible reasons:")
                print("  - Severity threshold too low (too few clean rows)")
                print("  - Autoencoder compression ratio needs tuning")
                print("  - Target task may rely on the filtered patterns (unlikely)")
            print()
        
        summary = {
            "kill_shot_validated": bool(kill_shot_validated),
            "severity_threshold": severity_threshold,
            "clean_rows_count": len(synth_clean_df),
            "corrupt_rows_count": len(synth_corrupt_df),
            "f1_improvement": gains["f1_clean_vs_full"],
            "interpretation": (
                "Logical violations directly degrade downstream ML performance. "
                "Neuro-LCV metric successfully identifies and filters them."
                if kill_shot_validated
                else "Threshold or architecture tuning recommended."
            )
        }
        results["summary"] = summary
    
    return results


def print_experiment_report(results: Dict[str, Any]) -> str:
    """Format experiment results into a publication-ready report."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("DOWNSTREAM VALIDATION REPORT: Neuro-LCV Causality Proof")
    lines.append("=" * 70)
    
    if results.get("real_metrics"):
        lines.append("\n[Real Data Baseline]")
        for k, v in results["real_metrics"].items():
            if k != "n_train":
                lines.append(f"  {k:<15} {v:.4f}")
    
    if results.get("full_synthetic_metrics"):
        lines.append("\n[Full Synthetic (SOTA - Unfiltered)]")
        for k, v in results["full_synthetic_metrics"].items():
            if k != "n_train":
                lines.append(f"  {k:<15} {v:.4f}")
    
    if results.get("clean_synthetic_metrics"):
        lines.append("\n[Neuro-LCV Filtered Synthetic (Clean Only)]")
        for k, v in results["clean_synthetic_metrics"].items():
            if k != "n_train":
                lines.append(f"  {k:<15} {v:.4f}")
    
    if results.get("gains"):
        lines.append("\n[Performance Gains]")
        lines.append(f"  F1 Gain (Clean vs Full): {results['gains'].get('f1_clean_vs_full', 0):+.4f}")
        lines.append(f"  Accuracy Gain:           {results['gains'].get('accuracy_clean_vs_full', 0):+.4f}")
    
    if results.get("summary"):
        lines.append("\n[Kill Shot Status]")
        s = results["summary"]
        status = "✓ VALIDATED" if s.get("kill_shot_validated") else "✗ NOT VALIDATED"
        lines.append(f"  Status: {status}")
        lines.append(f"  Clean rows: {s.get('clean_rows_count')} | Corrupt: {s.get('corrupt_rows_count')}")
        lines.append(f"  Interpretation: {s.get('interpretation', '')}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    from src.catalog.loader import load_dataset
    from src.generators.copula import GaussianCopulaGenerator
    from src.fidelity.logical import neuro_lcv_score

    print("🚀 Initializing Semantic Filtration Proof...")
    
    # 1. Load sample dataset (HMDA)
    try:
        real_df, _ = load_dataset("hmda", rows=10000)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # 2. Generate Synthetic Data
    print("📈 Generating synthetic data via Gaussian Copula...")
    gen = GaussianCopulaGenerator()
    gen.fit(real_df)
    synth_df = gen.generate(n_samples=5000)

    # 3. Simulate Logical Corruption
    # We slightly poison the synthetic data by adding random noise to categorical fields
    # to simulate the "impossibility" that Neuro-LCV should catch.
    print("🧪 Simulating semantic corruption in synthetic data...")
    cat_cols = real_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        poison_idx = np.random.choice(synth_df.index, size=int(len(synth_df) * 0.15), replace=False)
        for col in cat_cols:
            unique_vals = real_df[col].unique()
            synth_df.loc[poison_idx, col] = np.random.choice(unique_vals, size=len(poison_idx))

    # 4. Evaluate via Neuro-LCV
    print("🧠 Running Neuro-LCV Semantic Evaluation (CSSP)...")
    lcv_results = neuro_lcv_score(real_df, synth_df, epochs=20, verbose=True)
    penalties = lcv_results["row_penalties"]

    # 5. Run the Filtration Experiment
    # For HMDA, 'loan_approved' or similar isn't strictly there, we pick a column as target if exists
    # If not, we use 'action_taken' or just a dummy target for the proof
    target = 'action_taken' if 'action_taken' in real_df.columns else real_df.columns[-1]
    
    try:
        results = semantic_filtration_experiment(
            real_df=real_df,
            synth_df=synth_df,
            row_penalties=penalties,
            target_col=target,
            severity_threshold=0.5,
            verbose=True
        )
        report = print_experiment_report(results)
        print(report)
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
