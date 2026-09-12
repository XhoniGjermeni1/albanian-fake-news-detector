# Refactor-i i runtime-it final

## Qëllimi dhe kufijtë

Refactor-i ndryshoi vetëm organizimin dhe lexueshmërinë e kodit. Nuk u
ndryshuan modeli final, konfigurimi TF-IDF, Linear SVM `C=1.0`, sigmoid
calibration, preprocessing-u, pragjet `0.30/0.70`, linguistic features,
kontrata e `predict_final_news()` ose output-et zyrtare.

## Baseline para ndryshimeve

- test suite: **115 passed**, 0 failed;
- modeli: `models/final_word_char_linear_svm_calibrated_v1.joblib`;
- madhësia: **1,416,167 bytes**;
- SHA-256: `52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`;
- u ngrinë 6 raste: likely real, likely fake, uncertain, false positive,
  false negative dhe high-confidence error;
- snapshot-i: `reports/refactor_baseline_regression.json`.

## Varësitë reale të runtime-it

```text
app/streamlit_app.py
  -> app/streamlit_ui.py
  -> src/models/predict_final.py
       -> src/preprocessing/clean_text.py
       -> src/models/prediction_utils.py
            -> src/features/linguistic_features.py
```

Artefakti `.joblib` përmban Word + Character TF-IDF, Linear SVM dhe sigmoid
calibration. App-i nuk importon asnjë skript trajnimi ose analize ditore.

## Çfarë u ndryshua

1. `predict_final_news()` u kthye në orkestrues 15-rreshtash: zgjedh/ngarkon
   modelin, përgatit tekstin, merr probabilitetet dhe ndërton rezultatin.
2. Validimi i probabiliteteve u nda te `_predict_probabilities()` dhe ndërtimi
   i kontratës te `_build_prediction_result()`.
3. Pragjet finale merren nga një burim i vetëm, `prediction_utils.py`.
4. `normalize_spaces()` tani ka një implementim të vetëm te `clean_text.py`;
   u hoq kopja dhe importi i panevojshëm nga `linguistic_features.py`.
5. `streamlit_app.py` mban vetëm entrypoint-in, cache-in dhe rrjedhën e formës.
6. Validimi dhe rendering-u janë te `streamlit_ui.py`; CSS-i te `style.css`.
7. API-të që importojnë testet nga `streamlit_app.py` ruhen në mënyrë
   eksplicite me `__all__`.
8. Eksperimentet historike janë kataloguar te `experiments/README.md`.

## Statistikë para/pas

| Skedari | Para | Pas | Shpjegimi |
|---|---:|---:|---|
| `app/streamlit_app.py` | 522 | 138 | UI helpers dhe CSS u nxorën; entrypoint-i mbeti i qartë |
| `app/streamlit_ui.py` | 0 | 358 | validim, metadata, shembuj dhe rendering i zhvendosur |
| `app/style.css` | 0 | 73 | CSS-i statik u nxor nga kodi Python |
| `src/models/predict_final.py` | 90 | 112 | u shtuan dy helper-a të vegjël; funksioni publik ra 53 -> 15 rreshta |
| `src/features/linguistic_features.py` | 198 | 189 | u hoq normalizimi i duplikuar |
| `src/models/prediction_utils.py` | 47 | 47 | burimi i vetëm i pragjeve, i pandryshuar |
| `src/preprocessing/clean_text.py` | 45 | 45 | burimi i vetëm i preprocessing-ut, i pandryshuar |

Kodi Python i këtyre pjesëve, duke mos numëruar CSS-in, ra nga **902 në 889
rreshta**. Më e rëndësishmja, skedari kryesor i app-it ra
me 384 rreshta dhe funksioni publik i prediction-it me 38 rreshta. CSS-i ruhet
veçmas, jo fshihet. Kjo është ulje kompleksiteti, jo kompresim artificial.

## Auditimi i skedarëve të gjatë

| Skedari | Rreshta | Përgjegjësitë kryesore | Vendimi |
|---|---:|---|---|
| `calibrate_linear_svm.py` | 1,628 | group-safe calibration, thresholds, internal/external diagnostics, grafikë, raport | historik, pa ndryshim |
| `compare_tfidf_representations.py` | 1,454 | 3 përfaqësime, stability tests, metrics, grafikë, raport | historik, pa ndryshim |
| `compare_classifiers.py` | 1,413 | group CV, selection, internal/external evaluation, raport | historik, pa ndryshim |
| `finalize_model.py` | 1,329 | verifikim artefakti, hash, regression, eksportet finale | freeze tooling, jo runtime |
| `analyze_length_domain_shift.py` | 1,324 | length groups, stability, domain shift, grafikë, raport | historik, pa ndryshim |
| `tune_linear_svm.py` | 1,214 | CV për `C`, length analysis, internal/external diagnostics | historik, pa ndryshim |
| `evaluate_external_dataset.py` | 1,076 | external metrics, source/topic groups, errors, raport | historik, pa ndryshim |
| `analyze_model_quality.py` | 724 | baseline error analysis dhe calibration metrics | historik, pa ndryshim |
| `evaluate_app_system.py` | 638 | system cases, dataset predictions dhe demo selection | historik, pa ndryshim |
| `train_hybrid_model.py` | 616 | alignment, linguistic-only/hybrid training dhe raport | historik, pa ndryshim |

Funksionet më të gjata janë kryesisht `write_report()` me 230–344 rreshta dhe
orkestruesit `run_day*()` me 154–348 rreshta. U gjetën helper-a të përsëritur
si `dataframe_to_markdown`, `file_sha256`, `expected_calibration_error`,
`add_word_counts`, `prediction_table` dhe `verify_selection_hash`.

Këto nuk u bashkuan në këtë refactor sepse raportet dhe testet historike
importojnë implementime konkrete. Ndryshimi i tyre do të rriste rrezikun për
output-et e ngrira pa e bërë runtime-in final më të thjeshtë. Nuk u fshi kod
historik si "dead code" pa provë; ai mbetet material riprodhueshmërie.

## Struktura e re kryesore

- `src/preprocessing/clean_text.py`: kontrata e tekstit dhe Unicode NFC;
- `src/features/linguistic_features.py`: sinjalet vetëm për shpjegim;
- `src/models/prediction_utils.py`: pragjet dhe vendimi me tri nivele;
- `src/models/predict_final.py`: model loading dhe API finale;
- `app/streamlit_app.py`: rrjedha input -> model -> result;
- `app/streamlit_ui.py`: validimi dhe paraqitja;
- `app/style.css`: stili statik;
- `experiments/README.md`: harta e analizave historike.

## Core files për mësim

| Skedari | Pse duhet mësuar | Funksionet kryesore | Mund të injorosh |
|---|---|---|---|
| `src/data/load_dataset.py` | shpjegon raw -> DataFrame | `load_dataset`, `article_to_row` | fallback-et pasi kupton idenë |
| `src/preprocessing/clean_text.py` | tregon tekstin që sheh modeli | `normalize_spaces`, `combine_title_content` | asgjë thelbësore |
| `src/features/linguistic_features.py` | shpjegon sinjalet e UI-së | `extract_linguistic_features`, `find_phrases` | detajet e çdo regex-i |
| `src/models/prediction_utils.py` | tregon pragjet dhe explanation | `classify_probability`, `build_linguistic_explanation` | helper-in e formatimit |
| `src/models/predict_final.py` | tregon gjithë pipeline-in final | `load_final_model`, `predict_final_news` | asgjë thelbësore |
| `app/streamlit_app.py` | lidh formën me modelin | `get_cached_model`, `main` | detajet e widget-eve |
| `app/streamlit_ui.py` | tregon validimin dhe rendering-un | `validate_news_input`, `render_result` | shembujt dhe layout-in vizual |

Për calibration-in duhet kuptuar koncepti, jo kodi 1,628-rreshtësh: sigmoid
calibration është tashmë pjesë e artefaktit final dhe `predict_proba()` kthen
probabilitetet e kalibruara.

## Kontrolli pas refactor-it

- `python -m compileall -q app src tests`: kaloi;
- `python -m pytest -q`: **115 passed**, 0 failed;
- 6/6 vendime regresioni: identike;
- devijimi maksimal i probabilitetit: **0.0**;
- hash-i i modelit: identik byte-for-byte;
- Streamlit live smoke test: HTTP **200**, root-i i faqes u gjet;
- outpute zyrtare të modelit të ndryshuara: **asnjë**;
- modeli i vjetër nuk u rikthye në runtime.

Rezultati i plotë pas refactor-it është te
`reports/refactor_final_regression.json`.

## Përfundimi

Modeli dhe sjellja janë të pandryshuara. Kodi për mbrojtje tani ndiqet me një
rrjedhë të shkurtër:

```text
input -> NFC/whitespace -> artefakti final -> probabilitete
      -> pragjet 0.30/0.70 -> rezultat + sinjale shpjeguese
```

Skriptet e gjata duhen trajtuar si evidencë eksperimentale, jo si runtime që
duhet mësuar rresht për rresht.
