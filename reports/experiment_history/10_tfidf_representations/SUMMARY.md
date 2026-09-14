# Krahasimi i përfaqësimeve TF-IDF

## 1. Qëllimi

Të përcaktohej nëse modeli duhet të përdorë fjalë, karaktere apo bashkimin e tyre për të përfaqësuar tekstin.

## 2. Çfarë u provua

Me të njëjtin Logistic Regression dhe protokoll evaluation u krahasuan Word TF-IDF, Character TF-IDF dhe Word+Character TF-IDF. Për Character u kontrolluan konfigurimet `char_wb (3,5)` dhe `(3,6)`; zgjedhjet u bënë vetëm nga të dhënat e brendshme.

## 3. Rezultatet kryesore

Word+Character arriti F1 weighted të brendshëm 0.9027 dhe lidhjen më të ulët me gjatësinë ndër kandidatët kryesorë. Character-only ishte më i mirë në disa raste të shkurtra. [selection.json](selection.json) verifikon zgjedhjen dhe faktin që external results nuk u përdorën.

## 4. Përfundimi

Bashkimi ruan kuptimin e fjalëve dhe shton sinjale ortografike, mbaresa, pjesë fjalësh dhe pikësim që Word TF-IDF nuk i kap plotësisht.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U ngri konfigurimi Word+Character me `char_wb (3,5)` dhe mbi këtë përfaqësim u krahasuan classifier-a të ndryshëm.

## 6. Artefaktet dhe script-et përkatëse

- Entrypoint: `archive/experiments/models/compare_tfidf_representations.py`.
- Zbatim: `archive/experiments/models/experiment_support/day13_analysis.py`.
- Rezultat: [selection.json](selection.json).
- Modele: [artifacts/word_tfidf_logreg_calibrated.joblib](artifacts/word_tfidf_logreg_calibrated.joblib), [artifacts/char_tfidf_logreg_calibrated.joblib](artifacts/char_tfidf_logreg_calibrated.joblib), [artifacts/word_char_tfidf_logreg_calibrated.joblib](artifacts/word_char_tfidf_logreg_calibrated.joblib).
- Figura: [internal_model_comparison.png](figures/internal_model_comparison.png), [cohort_accuracy.png](figures/cohort_accuracy.png), [length_bias.png](figures/length_bias.png), [stability.png](figures/stability.png), [external_diagnostic.png](figures/external_diagnostic.png).
