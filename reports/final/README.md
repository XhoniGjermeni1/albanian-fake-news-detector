# Final Thesis Results

Këto janë referencat zyrtare; skedarët ruhen një nivel më sipër për
kompatibilitet me pipeline-in historik.

1. Modeli dhe metodologjia: [`day17_final_model.md`](../day17_final_model.md)
2. Metrikat: [`day17_final_metrics.json`](../day17_final_metrics.json)
3. Manifesti: [`models/final_model_v1_manifest.json`](../../models/final_model_v1_manifest.json)
4. Krahasimi final: [`day17_final_model_comparison.csv`](../day17_final_model_comparison.csv)
5. Vlerësimi i jashtëm pilot: [`day17_final_external_evaluation.csv`](../day17_final_external_evaluation.csv)
6. Bias-i i gjatësisë: [`day17_final_length_metrics.csv`](../day17_final_length_metrics.csv)
7. Rastet demo: [`day17_final_demo_cases.csv`](../day17_final_demo_cases.csv)
8. Mbyllja teknike: [`day20_final_closure.md`](../day20_final_closure.md)

Metrikat zyrtare të brendshme janë accuracy `0.911616`, F1 weighted
`0.911594`, F1 fake `0.909794`, Brier score `0.065768` dhe log loss `0.219176`.
Benchmark-u i jashtëm pilot ka accuracy `0.60` dhe dokumenton domain shift;
nuk përdoret për ndryshimin e modelit.
