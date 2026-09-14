# Word TF-IDF dhe Logistic Regression

## 1. Qëllimi

Të ndërtohej baseline-i i parë që mëson lidhjen midis përmbajtjes së tekstit dhe etiketave real/fake.

## 2. Çfarë u provua

Teksti i normalizuar u kthye në vektorë me Word TF-IDF, n-grams `(1, 2)`, `min_df=2`, maksimumi 30,000 features dhe pa kthim në lowercase. Mbi këta vektorë u trajnua Logistic Regression me pesha të balancuara të klasave.

## 3. Rezultatet kryesore

Në evaluation-in zyrtar me 792 artikuj:

| Metrika | Rezultati |
|---|---:|
| Accuracy | 0.8838 |
| F1 weighted | 0.8838 |
| F1 fake | 0.8808 |
| Recall real | 0.9023 |
| Recall fake | 0.8651 |

Kopja e tabelës zyrtare është [model_comparison.csv](model_comparison.csv); ajo përmban baseline-in dhe modelin final për krahasim.

## 4. Përfundimi

Word TF-IDF prodhoi një baseline të fortë dhe të interpretueshëm, shumë mbi Dummy baseline. Kufizimi kryesor ishte recall më i ulët për fake dhe ndjeshmëria ndaj formave ortografike ose strukturave që word tokens nuk i kapin mirë.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Para zgjerimit të përfaqësimit tekstual, u analizuan sinjalet gjuhësore dhe u testua nëse ato mund ta plotësonin TF-IDF-in.

## 6. Artefaktet dhe script-et përkatëse

- Script trajnimi: `archive/experiments/models/train_model.py`.
- Helper historik i prediction-it: `archive/experiments/models/predict.py`.
- Dataset-e: `data/interim/train.csv`, `data/interim/test.csv`.
- Model: [artifacts/baseline_tfidf_logreg.joblib](artifacts/baseline_tfidf_logreg.joblib).
- Rezultate: `reports/final/model_comparison.csv`, `reports/final/metrics.json`.
- Figurë e ripërdorur: [figures/model_comparison.png](figures/model_comparison.png).
