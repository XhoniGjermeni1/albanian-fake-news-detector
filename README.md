# Albanian Fake News Detector

Aplikacion dhe projekt diplome bachelor për klasifikimin gjuhësor të lajmeve
në gjuhën shqipe. Përdoruesi vendos titullin dhe përmbajtjen e një lajmi;
sistemi kthen probabilitetet `real/fake` dhe një nga vendimet:

- `likely_real` kur `P(fake) < 0.30`;
- `uncertain` kur `0.30 <= P(fake) <= 0.70`;
- `likely_fake` kur `P(fake) > 0.70`.

> **Kujdes:** modeli analizon ngjashmëri dhe karakteristika gjuhësore. Ai nuk
> kontrollon burime ose fakte dhe nuk zëvendëson fact-checking-un.

## Modeli Final

Versioni klasik final është `v1.0.0`:

- Word TF-IDF me n-grams `(1, 2)`;
- Character TF-IDF `char_wb` me n-grams `(3, 5)`;
- Linear SVM (`LinearSVC`, `C=1.0`, class weights të balancuara);
- sigmoid probability calibration me fold-e group-safe;
- preprocessing Unicode NFC pa hequr pikësimin, kapitalizimin ose `ë/ç`;
- linguistic features vetëm për shpjegim, jo si input i modelit final.

Artefaktet runtime janë:

```text
models/final_word_char_linear_svm_calibrated_v1.joblib
models/final_model_v1_manifest.json
```

## Pipeline-i

```text
Albanian Fake News Corpus
        ↓
ngarkim dhe validim
        ↓
preprocessing bazë + Unicode NFC
        ↓
Word TF-IDF + Character TF-IDF
        ↓
Linear SVM + sigmoid calibration
        ↓
probabilitete + pragjet 0.30/0.70
        ↓
Streamlit + shpjegim i sinjaleve gjuhësore
```

## Dataset-i

Projekti përdor **Albanian Fake News Corpus**:

- repository: <https://github.com/rexshijaku/alb-fake-news-corpus>;
- artikulli ACM: <https://dl.acm.org/doi/10.1145/3487288>.

Dataset-i raw ruhet i pandryshuar te
`data/raw/alb-fake-news-corpus/`. Dataset-et e përpunuara dhe ndarjet e ngrira
ruhen te `data/processed/` dhe `data/interim/`. Dataset-i pilot i jashtëm ruhet
te `data/external/external_news.csv`.

## Rezultatet Kryesore

Test set-i i brendshëm ka 792 artikuj dhe nuk është përdorur për tuning.

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

Në dataset-in e jashtëm pilot me 40 përmbledhje të shkurtra, accuracy ishte
60%. Ky rezultat dokumenton domain shift-in dhe nuk është përdorur për tuning
ose ndryshim të modelit. Detajet finale janë te
`reports/day17_final_model.md`.

## Struktura Kryesore

```text
albanian-fake-news-detector/
├── app/                  # ndërfaqja Streamlit
├── data/
│   ├── raw/              # corpus-i origjinal, i pandryshuar
│   ├── interim/          # train/test i ngrirë dhe teksti i pastruar
│   ├── processed/        # dataset-i dhe linguistic features
│   └── external/         # benchmark-u pilot i jashtëm
├── models/               # modeli final, manifesti dhe modele lokale historike
├── notebooks/            # auditimi dhe walkthrough-u final
├── reports/              # raportet, tabelat dhe figurat e eksperimenteve
├── src/
│   ├── data/             # loader, validation dhe dataset build
│   ├── preprocessing/    # pastrimi bazë dhe Unicode NFC
│   ├── features/         # linguistic features
│   └── models/           # prediction final dhe analiza historike
├── tests/                # regression, data, model dhe Streamlit tests
├── requirements.txt
└── README.md
```

Skriptet e Ditëve 2–16 dhe modelet e tyre janë ruajtur për riprodhueshmëri
akademike. Ato nuk përdoren nga runtime-i final.

## Instalimi

Kërkohet Git dhe Python 3.11. Klono projektin bashkë me corpus-in:

```powershell
git clone --recurse-submodules https://github.com/XhoniGjermeni1/albanian-fake-news-detector.git
cd albanian-fake-news-detector
```

Nëse projekti është klonuar pa submodule, ekzekuto:

```powershell
git submodule update --init --recursive
```

Pastaj, në Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Në macOS/Linux, aktivizimi bëhet me:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Versionet në `requirements.txt` përputhen me environment-in ku u ngrirë dhe u
testua modeli final.

## Nisja e Aplikacionit

```powershell
python -m streamlit run app\streamlit_app.py
```

Streamlit ngarkon vetëm artefaktin final me cache. Nëse modeli ose manifesti
mungon, aplikacioni shfaq një gabim të qartë dhe nuk tenton prediction.

Prediction-i mund të përdoret edhe drejtpërdrejt nga Python:

```python
from src.models.predict_final import predict_final_news

result = predict_final_news(
    title="Titulli i lajmit",
    content="Përmbajtja e lajmit në gjuhën shqipe.",
)
print(result["decision"], result["probability_fake"])
```

## Notebook-u Walkthrough

```powershell
python -m jupyter lab notebooks\02_final_walkthrough.ipynb
```

Notebook-u ndjek rrjedhën nga dataset-i te prediction-i, lexon output-et e
ngrira dhe nuk ritrajnon modelin. Rastet e demonstrimit ruhen te
`reports/day19_demo_cases.csv`, ndërsa skenari te
`reports/day19_demo_guide.md`.

## Testet

```powershell
python -m pytest -q
```

Testet mbulojnë loader-in, preprocessing-un, linguistic features, leakage
checks, modelet historike, konfigurimin/hash-in e modelit final, probabilitetet,
pragjet, Unicode NFC/NFD, Streamlit dhe walkthrough-un.

## Rindërtimi i Dataset-it

Nëse corpus-i raw mungon:

```powershell
git submodule update --init --recursive
python src\data\build_dataset.py
```

Kjo krijon `data/processed/articles.parquet` dhe preview-t përkatëse pa
ndryshuar skedarët raw.

## Kufizimet

- modeli klasifikon stilin gjuhësor, jo vërtetësinë faktike;
- ekziston bias i lidhur me gjatësinë e tekstit;
- tekste shumë të shkurtra mund të japin prediction-e të paqëndrueshme;
- performanca bie kur periudha, burimi, tema ose stili ndryshojnë nga corpus-i;
- linguistic features janë për interpretim dhe nuk janë prova;
- benchmark-u i jashtëm është pilot i vogël me përmbledhje manuale.

## Skedarët Kryesorë për Mbrojtje

| Roli | Skedari |
|---|---|
| Ngarkimi i dataset-it | `src/data/load_dataset.py` |
| Preprocessing | `src/preprocessing/clean_text.py` |
| Linguistic features | `src/features/linguistic_features.py` |
| Vendimet dhe shpjegimi | `src/models/prediction_utils.py` |
| Prediction final | `src/models/predict_final.py` |
| Aplikacioni | `app/streamlit_app.py` |
| Regression tests | `tests/test_final_model.py`, `tests/test_streamlit_app.py` |
| Walkthrough | `notebooks/02_final_walkthrough.ipynb` |

## Versioni

`v1.0.0` përfaqëson modelin klasik final. BERT/XLM-RoBERTa, SHAP dhe deploy
online mbeten zgjerime opsionale dhe nuk janë pjesë e këtij versioni.

Raporti i mbylljes teknike ruhet te `reports/day20_final_closure.md`, ndërsa
ndryshimet e release-it te `CHANGELOG.md`.
