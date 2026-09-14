# Krahasimi i classifier-ave

## 1. Qëllimi

Të zgjidhej algoritmi klasifikues më i mirë mbi përfaqësimin tashmë të ngrirë Word+Character TF-IDF.

## 2. Çfarë u provua

U vlerësuan gjashtë konfigurime nga tre familje: Logistic Regression (`C=0.5/1.0`), Linear SVM (`C=0.5/1.0`) dhe Complement Naive Bayes (`alpha=0.5/1.0`). Të gjithë përdorën të njëjtat pesë folds group-safe vetëm mbi train.

## 3. Rezultatet kryesore

| Classifier-i më i mirë i familjes | Mean F1 weighted | Devijimi standard |
|---|---:|---:|
| Logistic Regression, C=1.0 | 0.8952 | 0.0041 |
| Linear SVM, C=1.0 | 0.9110 | 0.0021 |
| ComplementNB, alpha=0.5 | 0.8810 | 0.0106 |

[selection.json](selection.json) tregon se Linear SVM kishte F1 më të lartë dhe stabilitetin më të mirë. Test-i i brendshëm dhe external nuk u përdorën për zgjedhje.

## 4. Përfundimi

Linear SVM dha kompromisin më të mirë midis performancës, stabilitetit të folds dhe ekuilibrit të recall-it.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U zgjodh familja Linear SVM dhe eksperimenti pasues testoi forcën e regularizimit përmes parametrit `C`.

## 6. Artefaktet dhe script-et përkatëse

- Entrypoint: `archive/experiments/models/compare_classifiers.py`.
- Zbatim: `archive/experiments/models/experiment_support/day14_analysis.py`.
- Rezultat: [selection.json](selection.json).
- Modele: [artifacts/logistic_regression.joblib](artifacts/logistic_regression.joblib), [artifacts/linear_svm.joblib](artifacts/linear_svm.joblib), [artifacts/complement_nb.joblib](artifacts/complement_nb.joblib).
- Figura: [cv_comparison.png](figures/cv_comparison.png), [internal_comparison.png](figures/internal_comparison.png), [length_performance.png](figures/length_performance.png), [external_diagnostic.png](figures/external_diagnostic.png).
