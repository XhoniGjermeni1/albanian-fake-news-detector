# Albanian Fake News Detector

Projekt diplome bachelor dhe aplikacion Streamlit për klasifikimin gjuhësor të
lajmeve në gjuhën shqipe. Sistemi merr titullin dhe përmbajtjen e një lajmi,
llogarit probabilitetet `real/fake` dhe kthen një nga tri vendimet:

- `likely_real` kur `P(fake) < 0.30`;
- `uncertain` kur `0.30 <= P(fake) <= 0.70`;
- `likely_fake` kur `P(fake) > 0.70`.

> **Kufizim i rëndësishëm:** modeli analizon modele tekstuale dhe sinjale
> gjuhësore. Ai nuk kontrollon burime ose fakte dhe nuk zëvendëson fact-checking-un.

Për metodologjinë, krahasimin e eksperimenteve dhe përfundimet, shiko
[`reports/final/FINAL_REPORT.md`](reports/final/FINAL_REPORT.md).
Historia kronologjike me një dosje të veçantë për çdo eksperiment ndodhet te
[`reports/experiment_history/EXPERIMENT_HISTORY.md`](reports/experiment_history/EXPERIMENT_HISTORY.md).

## Modeli Final

Versioni final `v1.0.0` përdor:

- Word TF-IDF me n-grams `(1, 2)`, `min_df=2` dhe `max_features=30000`;
- Character TF-IDF `char_wb` me n-grams `(3, 5)`, `min_df=2` dhe
  `max_features=50000`;
- `LinearSVC` me `C=1.0` dhe `class_weight="balanced"`;
- sigmoid probability calibration me pesë fold-e group-safe;
- pragjet e ngrira `0.30/0.70`;
- Unicode NFC dhe normalizim hapësirash, pa hequr kapitalizimin, pikësimin ose
  shkronjat `ë/ç`;
- linguistic features vetëm për shpjegim në UI, jo si input të modelit final.

Artefaktet aktive janë:

```text
final_model/final_word_char_linear_svm_calibrated_v1.joblib
final_model/final_model_v1_manifest.json
```

SHA-256 i modelit final:

```text
52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5
```

## Arkitektura e Projektit

Repository ndahet fizikisht në kod aktiv, artefakte finale dhe histori
eksperimentale:

```text
albanian-fake-news-detector/
├── app/                  # entrypoint, UI dhe stili i Streamlit
├── src/
│   ├── data/             # ngarkimi, validimi dhe ndërtimi i dataset-it
│   ├── preprocessing/    # preprocessing-u determinist
│   ├── features/         # linguistic features
│   ├── models/           # model builders, contract dhe prediction final
│   └── evaluation/       # folds, kontrolle të të dhënave dhe metrika
├── data/
│   ├── raw/              # corpus-i origjinal si Git submodule
│   ├── processed/        # dataset-i i përpunuar dhe linguistic features
│   ├── interim/          # clean dataset dhe split-et e ngrira
│   └── external/         # benchmark-u pilot i jashtëm
├── final_model/
│   ├── final_word_char_linear_svm_calibrated_v1.joblib
│   └── final_model_v1_manifest.json
├── reports/
│   ├── final/            # raporti dhe rezultatet zyrtare
│   └── experiment_history/ # rezultate dhe modele sipas eksperimentit
├── tests/                # testet e të dhënave, modelit dhe aplikacionit
├── archive/
│   └── experiments/      # kodi i eksperimenteve që çoi te modeli final
├── requirements.txt
└── README.md
```

`app/` dhe prediction-i final nuk importojnë nga `archive/`. Arkiva ruan vetëm
kodin historik; rezultatet dhe modelet eksperimentale ruhen pranë njëri-tjetrit
në `reports/experiment_history/`. Dosja `final_model/` ruan vetëm modelin final dhe
manifestin që përdor runtime-i.

## Pipeline-i i të Dhënave dhe Modelit

```text
Albanian Fake News Corpus
        │
        ▼
src/data/load_dataset.py
        │
        ▼
src/data/validate_dataset.py
        │
        ▼
src/data/build_dataset.py
        │
        ├── data/processed/articles.csv
        └── data/processed/articles.parquet
        │
        ▼
src/preprocessing/clean_text.py
        │
        ▼
data/interim/train.csv + data/interim/test.csv
        │
        ▼
Word TF-IDF + Character TF-IDF
        │
        ▼
Linear SVM (C=1.0)
        │
        ▼
sigmoid calibration + thresholds 0.30/0.70
        │
        ▼
final_model/final_word_char_linear_svm_calibrated_v1.joblib
        │
        ▼
src/models/predict_final.py
        │
        ▼
app/streamlit_app.py
```

Split-et dhe modeli final janë të ngrirë. Nisja e aplikacionit nuk ndërton
dataset-in dhe nuk ritrajnon modelin.

## Rrjedha e Prediction-it Final

```text
titulli + përmbajtja
        │
        ▼
prepare_final_model_text()
        │
        ▼
model.predict_proba()
        │
        ▼
classify_probability()
        │
        ├── likely_real
        ├── uncertain
        └── likely_fake
        │
        ▼
probabilitetet + shpjegimi gjuhësor + paralajmërimi
```

TF-IDF, Linear SVM dhe calibration ndodhen brenda artefaktit `.joblib`.
`build_linguistic_explanation()` nxjerr vetëm sinjale të lexueshme për UI-në
dhe nuk ndryshon probabilitetin.

## Përgjegjësia e Skedarëve Aktivë

| Skedari | Përgjegjësia |
|---|---|
| `src/data/load_dataset.py` | Lexon artikujt raw dhe ndërton rreshtat e dataset-it. |
| `src/data/validate_dataset.py` | Kontrollon skemën, mungesat, dublikatat dhe shpërndarjet. |
| `src/data/build_dataset.py` | Orkestron loading, validation dhe eksportin CSV/Parquet. |
| `src/preprocessing/clean_text.py` | Normalizon Unicode/hapësirat dhe ndërton `model_text`. |
| `src/features/linguistic_features.py` | Nxjerr karakteristika gjuhësore për analizë dhe shpjegim. |
| `src/features/build_linguistic_features.py` | Gjeneron tabelën e linguistic features për eksperimente. |
| `src/evaluation/data_utils.py` | Menaxhon dublikatat, grupet leakage-safe, folds dhe grupet e gjatësisë. |
| `src/evaluation/metrics.py` | Centralizon metrikat e klasifikimit dhe decision scores. |
| `src/models/builders.py` | Përshkruan konfigurimin Word/Character TF-IDF dhe Linear SVM. |
| `src/models/model_contract.py` | Verifikon konfigurimin, preprocessing-un dhe integritetin e modelit. |
| `src/models/prediction_utils.py` | Zbaton thresholds dhe ndërton shpjegimet gjuhësore. |
| `src/models/predict_final.py` | Ngarkon modelin final dhe ekspozon kontratën e vetme të prediction-it. |
| `app/streamlit_app.py` | Orkestron aplikacionin dhe thërret prediction-in final. |
| `app/streamlit_ui.py` | Validon input-in dhe paraqet rezultatin. |
| `app/style.css` | Përmban stilin vizual të aplikacionit. |

## Eksperimentet e Arkivuara

`archive/experiments/` përmban kodin që dokumenton si u arrit konfigurimi
final:

| Eksperimenti | Skedari kryesor | Qëllimi |
|---|---|---|
| Baseline minimal | `baseline/dummy_baseline.py` | Jep një pikë reference pa sinjale informative. |
| Baseline TF-IDF | `models/train_model.py` | Ndërton split-et dhe Logistic Regression fillestar. |
| Analiza gjuhësore | `features/analyze_linguistic_features.py` | Mat dallimet e linguistic features ndërmjet klasave. |
| Modeli hybrid | `models/train_hybrid_model.py` | Krahason TF-IDF me linguistic dhe hybrid features. |
| Cilësia e baseline-it | `models/analyze_model_quality.py` | Analizon probabilitetet, calibration-in dhe thresholds fillestare. |
| Testimi i sistemit | `models/evaluate_app_system.py` | Kontrollon kontratën e prediction-it dhe vendimet e aplikacionit. |
| External benchmark | `data/validate_external_dataset.py`, `models/evaluate_external_dataset.py` | Validon dhe vlerëson dataset-in pilot të jashtëm. |
| Length/domain shift | `models/analyze_length_domain_shift.py` | Mat varësinë nga gjatësia dhe ndryshimin e domain-it. |
| Përfaqësimi TF-IDF | `models/compare_tfidf_representations.py` | Krahason Word, Character dhe Word+Character TF-IDF. |
| Classifier-i | `models/compare_classifiers.py` | Krahason Logistic Regression, Linear SVM dhe ComplementNB. |
| SVM tuning | `models/tune_linear_svm.py` | Zgjedh parametrin `C` me group-safe cross-validation. |
| Calibration finale | `models/calibrate_linear_svm.py` | Zgjedh metodën e calibration-it dhe thresholds. |
| Ngrirja e modelit | `models/finalize_model.py` | Verifikon dhe ngrin modelin final pa ritrajnim. |
| Interpretueshmëria | `model_interpretability/linear_feature_coefficients.py` | Inspekton koeficientët globalë të modelit linear. |

Skedarët në `experiment_support/` janë helper-a të këtyre eksperimenteve dhe
nuk ekzekutohen si pipeline i pavarur.

## Dataset-et

Projekti përdor **Albanian Fake News Corpus**:

- repository: <https://github.com/rexshijaku/alb-fake-news-corpus>;
- publikimi ACM: <https://dl.acm.org/doi/10.1145/3487288>.

Corpus-i raw ruhet i pandryshuar te `data/raw/alb-fake-news-corpus/` si Git
submodule. Të dhënat e përpunuara ruhen te `data/processed/`, ndërsa split-et
e ngrira te `data/interim/`.

Benchmark-u i jashtëm `data/external/external_news.csv` përmban 40 raste të
balancuara: 20 real dhe 20 fake, të shpërndara në politikë, shëndetësi,
ekonomi, çështje sociale dhe teknologji. Tekstet janë përmbledhje manuale dhe
benchmark-u përdoret vetëm për vlerësim pilot, jo për trajnim, tuning,
calibration ose zgjedhje thresholds.

## Rezultatet Kryesore

Test set-i zyrtar përmban 792 artikuj pas përjashtimit të shtatë dublikatave
ekzakte me train set-in.

| Metrika | Rezultati |
|---|---:|
| Accuracy | 91.16% |
| F1 weighted | 91.16% |
| F1 fake | 90.98% |
| Recall real | 92.48% |
| Recall fake | 89.82% |
| Brier score | 0.0658 |
| Log loss | 0.2192 |
| Strong-decision coverage | 91.04% |
| Strong-decision accuracy | 94.31% |

Në benchmark-un e jashtëm modeli arriti accuracy 60%. Rezultati dokumenton
domain shift dhe nuk është përdorur për të ndryshuar modelin.

## Instalimi

Kërkohet Git dhe Python 3.11. Klono repository-n bashkë me corpus-in:

```powershell
git clone --recurse-submodules https://github.com/XhoniGjermeni1/albanian-fake-news-detector.git
cd albanian-fake-news-detector
```

Nëse repository është klonuar pa submodule:

```powershell
git submodule update --init --recursive
```

Krijo environment-in dhe instalo varësitë:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Në macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Nisja e Aplikacionit

```powershell
python -m streamlit run app\streamlit_app.py
```

Prediction-i mund të përdoret edhe nga Python:

```python
from src.models.predict_final import predict_final_news

result = predict_final_news(
    title="Titulli i lajmit",
    content="Përmbajtja e lajmit në gjuhën shqipe.",
)
print(result["decision"], result["probability_fake"])
```

## Testet

```powershell
python -m pytest -q
```

Testet mbulojnë dataset-in, preprocessing-un, leakage checks, linguistic
features, eksperimentet e modelit, probabilitetet, thresholds, prediction
anchors, SHA-256 dhe Streamlit.

## Rindërtimi i Dataset-it

```powershell
git submodule update --init --recursive
python src\data\build_dataset.py
```

Ky proces rindërton dataset-et e përpunuara, por nuk ritrajnon ose zëvendëson
modelin final.

## Artefaktet Finale

| Artefakti | Përmbajtja |
|---|---|
| `reports/final/FINAL_REPORT.md` | Metodologjia dhe përfundimet e eksperimenteve. |
| `reports/experiment_history/EXPERIMENT_HISTORY.md` | Rrjedha kronologjike dhe artefaktet e çdo eksperimenti. |
| `reports/final/metrics.json` | Metrikat e plota dhe kontrollet e integritetit. |
| `reports/final/model_comparison.csv` | Krahasimi baseline–model final. |
| `reports/final/external_evaluation.csv` | Metrikat e benchmark-ut të jashtëm. |
| `reports/final/external_predictions.csv` | Prediction-et e benchmark-ut të jashtëm. |
| `reports/final/length_metrics.csv` | Rezultatet sipas gjatësisë. |
| `reports/final/demo_cases.csv` | Rastet e ngrira për demonstrim. |
| `reports/final/figures/` | Figurat e përdorura nga raporti final. |

## Kufizimet

- Modeli klasifikon ngjashmëri gjuhësore, jo vërtetësinë faktike.
- Corpus-i përmban lidhje ndërmjet label-it, gjatësisë dhe burimit.
- Tekstet shumë të shkurtra dhe lajmet fake shumë të gjata janë më të
  vështira.
- Performanca bie kur periudha, burimi, tema ose stili ndryshojnë nga
  training corpus.
- Calibration nuk eliminon domain shift-in.
- Linguistic features dhe koeficientët e modelit janë përshkrues, jo prova
  shkakësore.

## Versioni

`v1.0.0` është modeli klasik final dhe i ngrirë. BERT/XLM-RoBERTa, SHAP dhe
deployment-i publik mbeten zgjerime jashtë këtij versioni.
