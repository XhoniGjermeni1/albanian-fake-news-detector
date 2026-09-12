# Report Index

Output-et historike mbeten në path-et origjinale që skriptet dhe referencat e
diplomës të vazhdojnë të funksionojnë. Ky indeks tregon ku duhet kërkuar pa i
lëvizur 137 artefaktet ekzistuese.

## Rezultatet finale

Hyrja më e shpejtë është `reports/final/README.md`.

- `day17_final_model.md`: raporti zyrtar i modelit final;
- `day17_final_metrics.json`: metrikat e ngrira;
- `day17_final_model_comparison.csv`: modeli i vjetër kundrejt modelit final;
- `day17_final_external_evaluation.csv`: benchmark-u pilot i jashtëm;
- `day17_final_length_metrics.csv`: rezultatet sipas gjatësisë;
- `day17_final_demo_cases.csv`: rastet e demonstrimit;
- `day20_final_closure.md`: kontrolli teknik i mbylljes.

## Evolucioni eksperimental

| Faza | Raporti kryesor |
|---|---|
| Dataset audit | `day1_dataset_audit.md` |
| Baseline TF-IDF | `day2_baseline_model.md` |
| Linguistic analysis | `day4_linguistic_feature_analysis.md` |
| Hybrid model | `day5_hybrid_model.md` |
| Error analysis/calibration baseline | `day6_model_quality.md` |
| External pilot | `day11_external_evaluation.md` |
| Length/domain shift | `day12_length_domain_shift.md` |
| TF-IDF representation | `day13_tfidf_representation_comparison.md` |
| Classifier comparison | `day14_classifier_comparison.md` |
| Linear SVM tuning | `day15_svm_tuning.md` |
| Final calibration/thresholds | `day16_calibration_thresholds.md` |
| Frozen final model | `day17_final_model.md` |

CSV-të ruajnë prediction-et dhe tabelat e plota; JSON-të ruajnë konfigurimin
dhe metrikat; figurat janë te `reports/figures/`.

## Rregulli shkencor

Dataset-i i jashtëm është **pilot evaluation only**. Ai nuk është përdorur për
tuning, model selection, calibration ose zgjedhjen e pragjeve.
