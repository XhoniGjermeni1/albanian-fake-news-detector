# Validimi dhe vlerësimi i benchmark-ut të jashtëm

## 1. Qëllimi

Të matej nëse modeli ruan performancën në lajme jashtë corpus-it të trajnimit dhe të kontrollohej që benchmark-u nuk ka leakage.

## 2. Çfarë u provua

Dataset-i me 40 raste të balancuara u kontrollua për skemë, etiketa, URL, data, Unicode, dublikata dhe ngjashmëri me train-in. Më pas kontrata e modelit u aplikua mbi 20 raste real dhe 20 fake dhe u krahasua me vlerësimin e brendshëm.

## 3. Rezultatet kryesore

Në fazën historike, baseline-i Word TF-IDF + Logistic Regression dha accuracy 0.4750, recall real 0.0500 dhe recall fake 0.9000. [external_evaluation.csv](external_evaluation.csv) ruan edhe rezultatin final 0.6000 për krahasim, ndërsa [external_predictions.csv](external_predictions.csv) përmban prediction-et e modelit final.

## 4. Përfundimi

Rënia e fortë dhe prirja e baseline-it drejt klasës fake treguan domain shift. Benchmark-u është i vlefshëm si pilot diagnostik, jo si burim për tuning ose zgjedhje modeli.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U analizua ndikimi i gjatësisë, stilit dhe domain-it për të kuptuar pse rezultati external ndryshonte kaq shumë nga test-i i brendshëm.

## 6. Artefaktet dhe script-et përkatëse

- Script-e: `archive/experiments/data/validate_external_dataset.py`, `archive/experiments/models/evaluate_external_dataset.py`.
- Dataset: `data/external/external_news.csv`.
- Model historik: `../06_initial_probability_calibration/artifacts/calibrated_tfidf_logreg.joblib`.
- Rezultate: `reports/final/external_evaluation.csv`, `reports/final/external_predictions.csv`.
- Figurë: [figures/external_confusion_matrix.png](figures/external_confusion_matrix.png).
