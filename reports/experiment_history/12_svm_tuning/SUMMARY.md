# Tuning i Linear SVM

## 1. Qëllimi

Të zgjidhej vlera e parametrit `C` që ruan performancën e Linear SVM pa shtuar paqëndrueshmëri ose rrezik overfitting-u.

## 2. Çfarë u provua

Mbi të njëjtin Word+Character TF-IDF dhe pesë folds group-safe u krahasuan `C = 0.25, 0.5, 1.0, 2.0, 4.0`. Zgjedhja përdori mean F1, variancën ndërmjet folds, diferencën e recall-it real/fake, generalization gap dhe preferencën për `C` më të ulët kur rezultatet ishin afër.

## 3. Rezultatet kryesore

`C=1.0` kishte mean F1 weighted 0.9110, devijim standard 0.0021 dhe mean recall gap 0.0752. `C=4.0` dha vetëm rreth 0.0013 më shumë mean F1, por ishte më pak stabil. [length_bias_comparison.csv](length_bias_comparison.csv) ruan analizën e gjatësisë për kandidatët kryesorë.

## 4. Përfundimi

Përmirësimi minimal i `C=4.0` nuk justifikoi regularizim më të dobët. `C=1.0` ishte zgjedhja më e qëndrueshme dhe më konservatore.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U ngri Linear SVM me `C=1.0`. Meqë Linear SVM prodhon decision scores dhe jo probabilitete të kalibruara, faza pasuese krahasoi sigmoid me isotonic calibration.

## 6. Artefaktet dhe script-et përkatëse

- Script: `archive/experiments/models/tune_linear_svm.py`.
- Rezultate: [selection.json](selection.json), [length_bias_comparison.csv](length_bias_comparison.csv).
- Modele: [artifacts/linear_svm_c_1_0.joblib](artifacts/linear_svm_c_1_0.joblib), [artifacts/linear_svm_c_2_0.joblib](artifacts/linear_svm_c_2_0.joblib), [artifacts/linear_svm_c_4_0.joblib](artifacts/linear_svm_c_4_0.joblib).
- Figura: [cv_c_tuning.png](figures/cv_c_tuning.png), [internal_candidate_comparison.png](figures/internal_candidate_comparison.png), [length_performance.png](figures/length_performance.png), [external_confusion_matrix.png](figures/external_confusion_matrix.png).
