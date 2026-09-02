# Historical Experiments

Kjo dosje është indeksi i eksperimenteve që çuan te modeli final. Skriptet
mbeten në path-et e tyre origjinale sepse testet, komandat e raporteve dhe
importet historike i referojnë drejtpërdrejt. Ato nuk importohen nga runtime-i
final i Streamlit.

## Harta e eksperimenteve

| Faza | Skripti kryesor | Qëllimi |
|---|---|---|
| Dita 2 | `src/models/train_model.py` | Word TF-IDF + Logistic Regression baseline |
| Dita 3 | `src/features/build_linguistic_features.py` | Ndërtimi i tabelës së linguistic features |
| Dita 4 | `src/features/analyze_linguistic_features.py` | Analiza statistikore e features |
| Dita 5 | `src/models/train_hybrid_model.py` | Linguistic-only dhe modeli hibrid |
| Ditët 6–9 | `src/models/analyze_model_quality.py`, `src/models/evaluate_app_system.py` | Error analysis, calibration baseline dhe testim sistemi |
| Ditët 10–11 | `src/data/validate_external_dataset.py`, `src/models/evaluate_external_dataset.py` | Kontrolli dhe vlerësimi i benchmark-ut të jashtëm |
| Dita 12 | `src/models/analyze_length_domain_shift.py` | Bias-i i gjatësisë dhe domain shift |
| Dita 13 | `src/models/compare_tfidf_representations.py` | Word, Character dhe Word + Character TF-IDF |
| Dita 14 | `src/models/compare_classifiers.py` | Logistic Regression, Linear SVM dhe Complement NB |
| Dita 15 | `src/models/tune_linear_svm.py` | Tuning i kufizuar i `C` |
| Dita 16 | `src/models/calibrate_linear_svm.py` | Sigmoid/isotonic calibration dhe pragjet |
| Dita 17 | `src/models/finalize_model.py` | Verifikimi dhe ngrirja e modelit final |

Output-et e ngrira janë te `reports/`, `reports/figures/` dhe `models/`.
Notebook-et përdoren vetëm për auditim dhe walkthrough.

Utility-t e përbashkëta nuk jetojnë më brenda një dite specifike:

- `src/evaluation/data_utils.py` për grouping, alignment dhe length cohorts;
- `src/evaluation/metrics.py` për metrikat e përbashkëta;
- `src/evaluation/experiment_utils.py` për hash dhe tabela raportimi;
- `src/models/builders.py` për konfigurimin Word + Character TF-IDF + SVM.

## Çfarë përdor aplikacioni

Rrjedha runtime nuk varet nga skriptet e mësipërme:

```text
app/streamlit_app.py
        -> src/models/predict_final.py
        -> src/preprocessing/clean_text.py
        -> models/final_word_char_linear_svm_calibrated_v1.joblib
```

`src/features/linguistic_features.py` përdoret pas prediction-it vetëm për
sinjalet shpjeguese të UI-së. Ai nuk ndryshon probabilitetin e modelit final.

Për mbrojtjen duhet kuptuar pyetja, metoda dhe përfundimi i çdo eksperimenti;
nuk është e nevojshme të mësohet kodi i eksportimit të CSV/JSON, grafikëve ose
gjenerimit të raporteve Markdown.
