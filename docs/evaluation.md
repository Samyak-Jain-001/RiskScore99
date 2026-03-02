Evaluation Strategy
===================

This document describes how RiskScore99 evaluates its fraud detection model and how the risk score / decision thresholds are derived.

### Metrics

- **ROC-AUC (Area Under ROC Curve)**:
  - Measures the trade-off between true positive rate (TPR) and false positive rate (FPR) across thresholds.
  - Useful as a general discrimination metric, but can be overly optimistic for highly imbalanced datasets.

- **PR-AUC (Area Under Precision-Recall Curve)**:
  - More informative for imbalanced datasets like fraud detection.
  - Focuses on precision (fraction of predicted fraud that is truly fraud) and recall (fraction of total fraud caught).
  - High PR-AUC indicates good performance at identifying rare fraudulent transactions without too many false alarms.

- **Confusion Matrix and Threshold Metrics**:
  - At chosen probability thresholds, we compute TP, FP, TN, FN, as well as precision, recall, and F1.
  - This forms the basis for practical decision thresholds (APPROVE / CHALLENGE / REVIEW / BLOCK).

### Why PR-AUC Matters for Imbalanced Data

- Fraud datasets are typically dominated by legitimate transactions.
- A classifier can achieve high ROC-AUC and accuracy while still missing many fraud cases or overwhelming operations with false positives.
- PR-AUC focuses directly on the positive (fraud) class:
  - **Precision** penalizes false positives (customer friction and operational cost).
  - **Recall** penalizes false negatives (missed fraud losses).
- RiskScore99 reports both ROC-AUC and PR-AUC, but prioritizes PR-AUC for threshold selection.

### Threshold Trade-Offs

The model outputs a calibrated fraud probability \( p \in [0,1] \). We sweep probability thresholds from 0 to 1 and compute, for each threshold:

- True/false positives and negatives.
- Precision, recall, and F1.
- An approximate **expected cost**:
  - \( \text{cost\_FN} = \text{fraud\_loss\_multiplier} \times \text{avg\_TransactionAmt} \)
  - \( \text{cost\_FP} = \text{friction\_cost} \)
  - Total expected cost \( = FN \times \text{cost\_FN} + FP \times \text{cost\_FP} \)

This helps answer:

- Where is the “knee” of the PR curve?
- How much additional fraud is prevented by lowering thresholds vs. how much customer friction and review workload is introduced?

### Mapping Probability → Score → Decision Thresholds

1. **Calibrated Probability**:
   - The model is wrapped with `CalibratedClassifierCV` (sigmoid / Platt scaling) to output well-calibrated probabilities \( p \).

2. **Risk Score (0–99)**:
   - The risk score is derived deterministically:
     - \( \text{risk\_score} = \text{round}(99 \times p) \)
   - This yields a human-friendly 0–99 scale where:
     - 0 ≈ very low fraud risk
     - 99 ≈ highest risk

3. **Policy Thresholds (T1, T2, T3)**:
   - Configuration in `config.py` defines:
     - \( T1 \) – lower threshold (e.g., \( p \approx 0.2 \))
     - \( T2 \) – mid threshold (e.g., \( p \approx 0.5 \))
     - \( T3 \) – high threshold (e.g., \( p \approx 0.8 \))
   - These map to decisions via the `PolicyAgent`:
     - `score < T1` → **APPROVE**
     - `T1 ≤ score < T2` → **CHALLENGE**
     - `T2 ≤ score < T3` → **REVIEW**
     - `score ≥ T3` → **BLOCK**
   - Additional business rules adjust the decision for extreme cases (e.g., very high amount with high risk → BLOCK, or missing identity → REVIEW).

4. **From Evaluation to Policy**:
   - The threshold sweep from `scripts/evaluate_model.py` provides a table of performance and cost at various probability thresholds.
   - This table can be used to select \( T1, T2, T3 \) that balance:
     - Fraud prevented (high recall, low FN).
     - Customer friction and operational load (controlled FP and CHALLENGE/REVIEW volume).

### Outputs

The evaluation pipeline produces:

- **Core Metrics**:
  - ROC-AUC and PR-AUC logged to the model registry.
- **Threshold Sweep Table**:
  - Used by `/metrics` and documentation for reasoning about decision boundaries.
- **Plots**:
  - ROC curve (`docs/roc_curve.png`).
  - Precision-Recall curve (`docs/pr_curve.png`).

Together, these artifacts provide a clear story for model quality, trade-offs, and the relationship between probabilities, risk scores, and business decisions.

