# Modelet linguistic-only dhe hybrid

## 1. Qëllimi

Të testohej nëse karakteristikat gjuhësore mund të zëvendësojnë ose të përmirësojnë Word TF-IDF.

## 2. Çfarë u provua

U krahasuan katër variante me Logistic Regression: vetëm TF-IDF, vetëm linguistic features, TF-IDF me të gjitha linguistic features dhe TF-IDF me linguistic features pa karakteristikat direkte të gjatësisë.

## 3. Rezultatet kryesore

| Varianti | Accuracy | F1 fake |
|---|---:|---:|
| TF-IDF | 0.8977 | 0.8919 |
| Linguistic-only | 0.8270 | 0.8237 |
| Hybrid | 0.8902 | 0.8863 |
| Hybrid pa length features | 0.8914 | 0.8883 |

Tabela historike CSV/JSON nuk ekziston aktualisht; vlerat verifikohen në `reports/final/FINAL_REPORT.md`. Figura origjinale e krahasimit është ruajtur.

## 4. Përfundimi

Linguistic features kanë sinjal klasifikues, por nuk janë aq të forta sa TF-IDF. As modeli hybrid dhe as varianti pa gjatësi nuk e përmirësuan baseline-in tekstual.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Linguistic features u hoqën nga input-i i modelit kandidat dhe u mbajtën vetëm për analizë dhe shpjegim. Më pas u analizua cilësia e probabiliteteve të baseline-it tekstual.

## 6. Artefaktet dhe script-et përkatëse

- Script: `archive/experiments/models/train_hybrid_model.py`.
- Dataset-e: `data/interim/train.csv`, `data/interim/test.csv`, `data/processed/linguistic_features.csv`.
- Modele: [artifacts/linguistic_features_logreg.joblib](artifacts/linguistic_features_logreg.joblib), [artifacts/hybrid_tfidf_linguistic_logreg.joblib](artifacts/hybrid_tfidf_linguistic_logreg.joblib), [artifacts/hybrid_tfidf_linguistic_no_length_logreg.joblib](artifacts/hybrid_tfidf_linguistic_no_length_logreg.joblib).
- Burim rezultatesh: `reports/final/FINAL_REPORT.md`.
- Figurë: [figures/model_comparison.png](figures/model_comparison.png).
