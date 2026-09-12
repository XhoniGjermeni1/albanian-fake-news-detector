# Final Thesis Results

Kjo dosje përmban vetëm rezultatet dhe figurat finale të përdorura nga diploma
dhe walkthrough-u. Raportet ditore dhe output-et diagnostike ruhen te
`archive/reports/`.

1. Modeli dhe metodologjia: [`model.md`](model.md)
2. Metrikat zyrtare: [`metrics.json`](metrics.json)
3. Manifesti: [`models/final_model_v1_manifest.json`](../../models/final_model_v1_manifest.json)
4. Krahasimi final: [`model_comparison.csv`](model_comparison.csv)
5. Vlerësimi i jashtëm pilot: [`external_evaluation.csv`](external_evaluation.csv)
6. Prediction-et e jashtme të walkthrough-ut: [`external_predictions.csv`](external_predictions.csv)
7. Rezultatet sipas gjatësisë: [`length_metrics.csv`](length_metrics.csv)
8. Rastet dhe skenari demo: [`demo_cases.csv`](demo_cases.csv), [`demo_guide.md`](demo_guide.md)
9. Figurat finale: [`figures/`](figures/)

Metrikat zyrtare të brendshme janë accuracy `0.911616`, F1 weighted
`0.911594`, F1 fake `0.909794`, Brier score `0.065768` dhe log loss `0.219176`.
Benchmark-u i jashtëm pilot ka accuracy `0.60`; nuk është përdorur për tuning
ose ndryshim të modelit.
