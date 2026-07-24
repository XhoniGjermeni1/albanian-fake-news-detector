# Albanian Fake News Detector

Projekt diplome bachelor për analizimin dhe detektimin e lajmeve të rreme në gjuhën shqipe.

Qëllimi final është ndërtimi i një aplikacioni në Python ku përdoruesi vendos titullin dhe përmbajtjen e një lajmi, ndërsa sistemi kthen një parashikim të bazuar në model:

- `real`
- `fake`
- `i pasigurt`

Rezultati do të paraqitet si probabilitet sipas modelit, jo si e vërtetë absolute.

## Dataset

Projekti përdor Albanian Fake News Corpus:

- Repository: `https://github.com/rexshijaku/alb-fake-news-corpus`
- Artikulli ACM: `https://dl.acm.org/doi/10.1145/3487288`

Dataseti origjinal ruhet i pandryshuar te:

```text
data/raw/alb-fake-news-corpus/
```

Dataseti i përpunuar krijohet te:

```text
data/processed/articles.csv
data/processed/articles.parquet
data/processed/articles_preview.csv
```

## Struktura Aktuale

Struktura mbahet minimale dhe rritet gradualisht me zhvillimin e projektit.

```text
Diploma/
  data/
    raw/
    interim/
    processed/
  notebooks/
    01_dataset_audit.ipynb
  reports/
    day1_dataset_audit.md
    day2_baseline_model.md
    day2_metrics.json
    day3_linguistic_features.md
    day3_feature_summary.csv
    day4_linguistic_feature_analysis.md
    day4_feature_quality.json
    day4_feature_comparison.csv
    day4_linguistic_only_model_metrics.json
    figures/
  src/
    data/
      load_dataset.py
      validate_dataset.py
      build_dataset.py
    preprocessing/
      clean_text.py
    models/
      train_model.py
      predict.py
    features/
      linguistic_features.py
      build_linguistic_features.py
      analyze_linguistic_features.py
  models/
    baseline_tfidf_logreg.joblib
    linguistic_features_logreg.joblib
  tests/
    test_load_dataset.py
    test_clean_text.py
    test_linguistic_features.py
    test_feature_analysis.py
  requirements.txt
  README.md
  .gitignore
```

## Instalimi

Krijo dhe aktivizo një virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalo dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Ndërtimi i Datasetit

Nëse dataseti nuk është shkarkuar ende, klonoje manualisht:

```powershell
git clone https://github.com/rexshijaku/alb-fake-news-corpus.git data\raw\alb-fake-news-corpus
```

Pastaj ekzekuto pipeline-in e Ditës 1:

```powershell
python src\data\build_dataset.py
```

Ky script lexon artikujt raw, krijon kolonat bazë, validon datasetin dhe ruan versionet e përpunuara.

## Trajnimi i Baseline Modelit

Ekzekuto:

```powershell
python src\models\train_model.py
```

Ky script bën preprocessing bazë, ndarje train/test, trajnon modelin e parë `TF-IDF + Logistic Regression`, ruan modelin dhe gjeneron metrikat.

Output-et kryesore:

```text
data/interim/articles_clean.csv
data/interim/train.csv
data/interim/test.csv
models/baseline_tfidf_logreg.joblib
reports/day2_metrics.json
```

## Nxjerrja e Karakteristikave Gjuhësore

Ekzekuto:

```powershell
python src\features\build_linguistic_features.py
```

Ky script nxjerr karakteristika strukturore, pikësimi, kapitalizimi, diakritika shqipe, shprehje sensacionale, tregues burimi dhe shprehje pasigurie.

Output-et kryesore:

```text
data/processed/linguistic_features.csv
reports/day3_feature_summary.csv
```

## Analiza e Karakteristikave Gjuhësore

Ekzekuto:

```powershell
python src\features\analyze_linguistic_features.py
```

Ky script kontrollon cilësinë e feature-ve, krahason real/fake, krijon grafikë dhe provon një model të vogël vetëm me karakteristika gjuhësore.

Output-et kryesore:

```text
reports/day4_feature_quality.json
reports/day4_feature_comparison.csv
reports/day4_linguistic_only_model_metrics.json
reports/figures/
models/linguistic_features_logreg.joblib
```

## Testet

Ekzekuto:

```powershell
python -m pytest
```

Testet aktuale kontrollojnë funksionimin bazë të loader-it të datasetit.

## Raportet

Detajet e punës ditore ruhen te `reports/`.

Raporti aktual:

- `reports/day1_dataset_audit.md`
- `reports/day2_baseline_model.md`
- `reports/day3_linguistic_features.md`
- `reports/day4_linguistic_feature_analysis.md`

README shërben vetëm si hyrje profesionale dhe udhëzues ekzekutimi. Gjetjet e detajuara, numrat e datasetit dhe problemet e validimit ruhen në raportet përkatëse.

## Roadmap

Hapat e ardhshëm të projektit:

- auditim dhe EDA më i thelluar
- parapërpunim i tekstit shqip
- nxjerrje e karakteristikave gjuhësore
- krijim i TF-IDF features
- trajnim i modelit bazë Machine Learning
- kalibrim probabilitetesh
- shpjegim i parashikimit
- aplikacion web me Streamlit

Modeli i parë i planifikuar do të jetë i thjeshtë dhe i interpretueshëm: TF-IDF, karakteristika gjuhësore dhe Logistic Regression ose Linear SVM.
