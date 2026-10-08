# MTech abstract implementation

Source: user-supplied `m.tech abstract 1.pdf`, abstract on page 2. Future work is excluded.

## Requirements and evidence

| Abstract requirement | Implementation | Review evidence |
|---|---|---|
| Water measurements including pH, hardness, solids, sulfate, conductivity | Nine original measurements; no derived target leakage | `config.py`, `data/water_potability.csv` |
| RF, SVR, ANN and XGBoost comparisons | Same stratified partitions for all baselines | `model/train_model.py`, `reports/abstract/metrics.csv` |
| SHAP identifies important features | Training-only XGBoost regressor, Tree SHAP, normalized mean absolute contributions | `feature_importance` and `initialization_weights` in evaluation JSON |
| Feature importance initializes ANN | Feature-weighted Glorot initialization, 9-16-8-1 ANN | `wqi/neural.py`, initialization test |
| ANN combined with XGBoost | Validation-selected weighted averaging or logistic stacking | `selected_fusion`, `fusion_validation` in evaluation JSON |
| Binary potable/nonpotable classification | Probability and class at threshold 0.5 | `model/predict.py`, Streamlit prediction page |
| Fixed dataset and two classes | Original dataset; no multiclass or synthetic time data | Dataset profile and saved split row IDs |

## Explicit implementation choice

The abstract does not give a weight-matrix formula. The abstract mode uses
`W[i,j] = GlorotUniform(seed)[i,j] * sqrt(9 * normalized_SHAP_importance[i])`.
Larger importance scales the initial connection variance while keeping different signed connections
between hidden neurons. This is our documented implementation choice, not recovered author code
and not a guarantee of improved performance. Glorot's seeded implementation is documented at
https://keras.io/api/layers/initializers/ ; Tree SHAP at
https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html .

The earlier paper reproduction mode (`repeat`) remains independently available. It expands the
nine importance weights into identical columns. The two snapshots are never overwritten by
the default commands and the app labels the selected mode.

## Train and verify

```bash
python model/train_model.py --initialization feature_glorot --full
python model/verify_artifacts.py --output model/abstract --reports reports/abstract
python -m unittest discover -s tests -v
```

The full run includes baselines, ablations, feature perturbations, five split seeds, and final-hybrid
Kernel SHAP. Selection uses validation labels only; test labels are used for final reporting.
The baseline SVR output is clipped to [0,1] for comparison with binary probabilities; it is not a
calibrated classifier. The ANN and hybrid outputs are not established as calibrated either.

## Scope limits

Multiclass WQI levels, time-series modeling, transfer learning and domain adaptation are future
work in the abstract. They are not presented as completed features. This system predicts a
dataset's binary potability label, not a continuous physical WQI value or laboratory safety verdict.
Measured results are in `reports/abstract/`; no reference paper scores are inserted into results.
