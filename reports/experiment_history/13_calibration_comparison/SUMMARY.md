# Krahasimi sigmoid dhe isotonic calibration

## 1. Qëllimi

Të ktheheshin decision scores të Linear SVM në probabilitete të përdorshme dhe të zgjidhej metoda me cilësinë më të mirë pa rrezik të panevojshëm overfitting-u.

## 2. Çfarë u provua

Sigmoid dhe isotonic u krahasuan me nested out-of-fold calibration dhe pesë folds group-safe. U matën klasifikimi, Brier score, log-loss, ECE, stabiliteti ndërmjet folds dhe gabimet me confidence të lartë. Test-i i brendshëm dhe external mbetën të mbyllur gjatë zgjedhjes.

## 3. Rezultatet kryesore

| Metoda | F1 weighted | Brier | Log-loss | ECE |
|---|---:|---:|---:|---:|
| Sigmoid | 0.9133 | 0.06534 | 0.21752 | 0.01486 |
| Isotonic | 0.9114 | 0.06588 | 0.24753 | 0.01379 |

Sigmoid ishte brenda tolerancës së calibration-it, kishte log-loss më të mirë dhe rrezik më të ulët overfitting-u. Fold-et e plota gjenden te [calibration_fold_metrics.csv](calibration_fold_metrics.csv).

## 4. Përfundimi

Sigmoid ishte zgjedhja më e sigurt dhe më e qëndrueshme për dataset-in me këtë madhësi. ECE pak më e ulët e isotonic nuk kompensoi log-loss më të dobët dhe fleksibilitetin më të madh.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U ngri sigmoid calibration. Probabilitetet OOF të kësaj metode u përdorën më pas për të zgjedhur zonën e pasigurisë dhe pragjet finale.

## 6. Artefaktet dhe script-et përkatëse

- Entrypoint: `archive/experiments/models/calibrate_linear_svm.py`.
- Zbatim: `archive/experiments/models/experiment_support/day16_analysis.py`.
- Rezultate: [metrics.json](metrics.json), [selection.json](selection.json), [calibration_fold_metrics.csv](calibration_fold_metrics.csv), [internal_predictions.csv](internal_predictions.csv), [external_predictions.csv](external_predictions.csv).
- Model kandidat: [artifacts/word_char_linear_svm_calibrated.joblib](artifacts/word_char_linear_svm_calibrated.joblib).
- Figura: [oof_calibration_comparison.png](figures/oof_calibration_comparison.png), [internal_calibration.png](figures/internal_calibration.png), [model_comparison.png](figures/model_comparison.png), [length_probability.png](figures/length_probability.png).
