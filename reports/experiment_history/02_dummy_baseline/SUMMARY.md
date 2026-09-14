# Dummy baseline

## 1. Qëllimi

Të përcaktohej rezultati minimal që çdo model tekstual duhet të kalojë, pa përdorur asnjë informacion nga fjalët e artikullit.

## 2. Çfarë u provua

`DummyClassifier(strategy="most_frequent")` mësoi vetëm klasën më të shpeshtë në train dhe parashikoi çdo artikull test si atë klasë. U përdor i njëjti test set prej 792 rastesh si në vlerësimin final.

## 3. Rezultatet kryesore

| Metrika | Rezultati |
|---|---:|
| Accuracy | 0.5038 |
| F1 weighted | 0.3376 |
| F1 fake | 0.0000 |

Train-i ka 1,598 raste real dhe 1,597 fake, prandaj modeli parashikon gjithmonë `real`. Output-et e veçanta `dummy_baseline_metrics.json` dhe `dummy_baseline_comparison.csv` nuk ekzistojnë aktualisht; vlerat e mësipërme verifikohen vetëm në raportin final.

## 4. Përfundimi

Një accuracy rreth 50% dhe F1 fake zero tregojnë se shpërndarja pothuajse e balancuar nuk mjafton për klasifikim. Modelet tekstuale duhet të demonstrojnë përmirësim të qartë mbi këtë kontroll trivial.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

U kalua te Word TF-IDF + Logistic Regression për të testuar nëse përmbajtja e fjalëve ofron sinjal real klasifikues.

## 6. Artefaktet dhe script-et përkatëse

- Script: `archive/experiments/baseline/dummy_baseline.py`.
- Dataset-e: `data/interim/train.csv`, `data/interim/test.csv`.
- Burim rezultatesh: `reports/final/FINAL_REPORT.md`.
- Model `.joblib`: nuk ruhet, sepse Dummy përdoret vetëm si kontroll akademik.
- CSV/JSON dhe figura të veçanta: të padisponueshme.
