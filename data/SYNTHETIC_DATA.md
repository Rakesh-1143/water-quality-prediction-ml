# Synthetic water-quality dataset (demonstration only)

`synthetic_water_quality.csv` is **generated data, not real measurements**. Do not present results on it as real-world accuracy.

- **Generator:** `src/synthetic.py` (`python model/make_synthetic_dataset.py`). Seed 7, 5000 rows, same 9 columns as the Kaggle dataset plus `Potability`.
- **Features:** sampled from normal distributions with the mean and spread of the real dataset, clipped to its observed ranges. No missing values.
- **Label rule:** a risk score (weighted, scaled exceedances of guideline values for pH distance from 7.5, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity). The 39% lowest-risk samples are potable.
- **Label noise:** 10% of labels are flipped at random (9.9% measured) to mimic measurement and labelling error. Noise sets the ceiling: no model can exceed 90.1% accuracy on this file. The noise level was fixed in advance and not tuned to reach a score.
- **Metadata:** `synthetic_water_quality.meta.json`.
