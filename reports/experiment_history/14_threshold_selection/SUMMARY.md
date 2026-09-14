# Zgjedhja e pragjeve finale

## 1. Qëllimi

Të krijohej një zonë `uncertain` që shmang vendimet e forta kur probabiliteti është afër kufirit dhe të zgjidheshin pragjet vetëm nga train OOF predictions.

## 2. Çfarë u provua

Me probabilitetet sigmoid u krahasuan zonat 0.30–0.70, 0.35–0.65 dhe 0.40–0.60. U matën mbulimi i vendimeve të forta, accuracy e tyre, numri i gabimeve të kapura në zonën uncertain dhe false positives/false negatives që mbetën të forta.

## 3. Rezultatet kryesore

| Pragjet | Mbulimi i fortë | Accuracy e fortë | Gabime në uncertain |
|---|---:|---:|---:|
| 0.30 / 0.70 | 0.8858 | 0.9488 | 132 |
| 0.35 / 0.65 | 0.9167 | 0.9403 | 102 |
| 0.40 / 0.60 | 0.9515 | 0.9283 | 59 |

Zona 0.30–0.70 dha accuracy më të lartë për vendimet e forta dhe kapi më shumë gabime si `uncertain`.

## 4. Përfundimi

Mbulimi më i ulët ishte një kompromis i pranueshëm për vendime më të kujdesshme. Pragjet nuk ndryshojnë modelin; ato ndryshojnë vetëm mënyrën si probabiliteti paraqitet si vendim.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U ngrinë pragjet 0.30/0.70. Me përfaqësimin, classifier-in, `C`, calibration-in dhe pragjet të zgjedhura, kandidati mund të vlerësohej dhe të ngrihej si model final.

## 6. Artefaktet dhe script-et përkatëse

- Entrypoint: `archive/experiments/models/calibrate_linear_svm.py`.
- Zbatim: `archive/experiments/models/experiment_support/day16_analysis.py`.
- Rezultate: [selection.json](selection.json), [metrics.json](metrics.json). Këto janë kopje të ngrira të rezultateve të përbashkëta me eksperimentin e calibration-it.
- Model kandidat: `../13_calibration_comparison/artifacts/word_char_linear_svm_calibrated.joblib`.
- Figurë: [figures/threshold_comparison.png](figures/threshold_comparison.png).
