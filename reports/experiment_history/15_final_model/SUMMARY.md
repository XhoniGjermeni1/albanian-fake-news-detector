# Vlerësimi dhe ngrirja e modelit final

## 1. Qëllimi

Të verifikohej kandidati i zgjedhur dhe të publikohej saktësisht i njëjti artefakt, pa ritrajnim ose ndryshim konfigurimi.

## 2. Çfarë u provua

U kontrolluan struktura Word/Character TF-IDF, Linear SVM `C=1.0`, sigmoid calibration, pesë folds, klasat, preprocessing-u NFC, pragjet 0.30/0.70, probabilitetet, prediction anchors, reload-i determinist dhe mungesa e group leakage. Kandidati u kopjua byte-for-byte si versioni final.

## 3. Rezultatet kryesore

| Metrika e brendshme | Rezultati |
|---|---:|
| Accuracy | 0.9116 |
| F1 weighted | 0.9116 |
| F1 fake | 0.9098 |
| Recall real | 0.9248 |
| Recall fake | 0.8982 |
| Brier score | 0.0658 |
| Log-loss | 0.2192 |
| Strong coverage | 0.9104 |
| Strong accuracy | 0.9431 |

Në benchmark-un external accuracy ishte 0.6000. Modeli final dhe kandidati kanë të njëjtin SHA-256: `52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`.

## 4. Përfundimi

Modeli final është determinist dhe i versionuar. Ai përmirëson baseline-in e brendshëm dhe balancën external, por vazhdon të ketë domain shift dhe nuk duhet paraqitur si fact-checker.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Modeli u ngri si `v1.0.0` dhe u lidh me kontratën aktive të prediction-it dhe Streamlit. Hapi i fundit ishte inspektimi i koeficientëve për interpretim global, pa ndryshuar modelin.

## 6. Artefaktet dhe script-et përkatëse

- Entrypoint finalizimi: `archive/experiments/models/finalize_model.py`.
- Verifikim: `archive/experiments/models/experiment_support/day17_analysis.py`, `src/models/model_contract.py`.
- Prediction aktiv: `src/models/predict_final.py`, `src/models/prediction_utils.py`.
- Model: `final_model/final_word_char_linear_svm_calibrated_v1.joblib`.
- Manifest: `final_model/final_model_v1_manifest.json`.
- Rezultate të kopjuara: [metrics.json](metrics.json), [model_comparison.csv](model_comparison.csv), [external_evaluation.csv](external_evaluation.csv), [external_predictions.csv](external_predictions.csv), [length_metrics.csv](length_metrics.csv), [demo_cases.csv](demo_cases.csv).
- Figura: [model_comparison.png](figures/model_comparison.png), [length_performance.png](figures/length_performance.png), [streamlit_desktop.png](figures/streamlit_desktop.png), [streamlit_mobile.png](figures/streamlit_mobile.png), [streamlit_final.png](figures/streamlit_final.png).
