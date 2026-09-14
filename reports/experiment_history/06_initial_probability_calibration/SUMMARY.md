# Analiza fillestare e probabiliteteve dhe calibration-it

## 1. Qëllimi

Të kontrollohej nëse probabilitetet e baseline-it Logistic Regression mund të përdoren me kujdes në aplikacion dhe nëse nevojitej një zonë e veçantë pasigurie.

## 2. Çfarë u provua

Baseline-i u kalibrua me folds pa group leakage. U matën accuracy, F1, Brier score, log-loss, ECE dhe gabimet me confidence të lartë. U krahasuan tre variante pragjesh për vendime `likely_real`, `uncertain` dhe `likely_fake`.

## 3. Rezultatet kryesore

Sigmoid calibration në këtë fazë uli accuracy nga 0.8977 në 0.8826, por krijoi probabilitete të përdorshme për vendime me tre nivele. Zona 0.30–0.70 ishte varianti më konservator dhe dha cilësinë më të lartë për vendimet e forta. JSON/CSV-të historike të kësaj faze nuk ekzistojnë aktualisht; përfundimet verifikohen në raportin final.

## 4. Përfundimi

Pragu binar 0.50 fsheh pasigurinë. Calibration-i e bën score-in më të interpretueshëm, por nuk eliminon gabimet me confidence të lartë ose domain shift-in.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U ruajt ideja e zonës `uncertain` dhe u testua kontrata e plotë probabilitet → prag → vendim përpara përdorimit në benchmark-un e jashtëm.

## 6. Artefaktet dhe script-et përkatëse

- Script: `archive/experiments/models/analyze_model_quality.py`.
- Dataset-e: `data/interim/train.csv`, `data/interim/test.csv`.
- Modele: `../03_word_tfidf_logreg/artifacts/baseline_tfidf_logreg.joblib`, [artifacts/calibrated_tfidf_logreg.joblib](artifacts/calibrated_tfidf_logreg.joblib).
- Burim rezultatesh: `reports/final/FINAL_REPORT.md`.
- Figurë: [figures/probability_calibration.png](figures/probability_calibration.png).
- CSV/JSON historike: të padisponueshme.
