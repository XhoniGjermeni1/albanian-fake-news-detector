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
albanian-fake-news-detector/
  data/
    raw/
    interim/
    processed/
  notebooks/
    01_dataset_audit.ipynb
  app/
    streamlit_app.py
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
    day5_hybrid_model.md
    day5_metrics.json
    day5_model_comparison.csv
    day6_model_quality.md
    day6_metrics.json
    day6_error_analysis.csv
    day6_threshold_comparison.csv
    day7_streamlit_app.md
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
      train_hybrid_model.py
      analyze_model_quality.py
      predict.py
    features/
      linguistic_features.py
      build_linguistic_features.py
      analyze_linguistic_features.py
  models/
    baseline_tfidf_logreg.joblib
    linguistic_features_logreg.joblib
    hybrid_tfidf_linguistic_logreg.joblib
    hybrid_tfidf_linguistic_no_length_logreg.joblib
    calibrated_tfidf_logreg.joblib
  tests/
    test_load_dataset.py
    test_clean_text.py
    test_linguistic_features.py
    test_feature_analysis.py
    test_hybrid_model.py
    test_model_quality.py
    test_streamlit_app.py
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

## Modeli Hibrid

Ekzekuto:

```powershell
python src\models\train_hybrid_model.py
```

Ky script kontrollon përputhjen e tekstit me linguistic features sipas `article_id`, ritrajnon modelet për një krahasim të drejtë dhe vlerëson:

- TF-IDF only
- linguistic features only
- TF-IDF + linguistic features
- modelin hibrid pa feature-t direkte të gjatësisë

Output-et kryesore:

```text
models/hybrid_tfidf_linguistic_logreg.joblib
models/hybrid_tfidf_linguistic_no_length_logreg.joblib
reports/day5_metrics.json
reports/day5_model_comparison.csv
reports/figures/day5_model_comparison.png
```

Detajet dhe interpretimi ruhen te `reports/day5_hybrid_model.md`.

## Kontrolli i Cilësisë dhe Calibration

Ekzekuto:

```powershell
python src\models\analyze_model_quality.py
```

Ky script analizon false positives/false negatives të TF-IDF baseline, kontrollon shpërndarjen e probabiliteteve, trajnon sigmoid calibration me fold-e të grupuara dhe krahason pragjet për `likely_real`, `uncertain` dhe `likely_fake`.

Output-et kryesore:

```text
models/calibrated_tfidf_logreg.joblib
reports/day6_metrics.json
reports/day6_error_analysis.csv
reports/day6_interesting_errors.csv
reports/day6_probability_summary.csv
reports/day6_threshold_comparison.csv
reports/day6_calibration_bins.csv
reports/figures/day6_probability_calibration.png
```

Logjika e përgatitur për aplikacionin është funksioni `predict_news_for_app()` te `src/models/predict.py`. Pragjet fillestare janë 30% dhe 70%; zona midis tyre kthehet si `uncertain`.

## Aplikacioni Streamlit

Versioni i parë i aplikacionit përdor modelin e kalibruar `TF-IDF + Logistic Regression`. Përdoruesi mund të vendosë titullin, përmbajtjen ose të dyja dhe të marrë:

- vendimin `likely_real`, `uncertain` ose `likely_fake`;
- probabilitetin e modelit për klasat real dhe fake;
- karakteristika të vëzhguara në tekst, si gjatësia, pikëçuditëset, kapitalizimi, diakritikat dhe marker-at gjuhësorë;
- paralajmërimin se rezultati nuk është verifikim faktik.

Hape aplikacionin nga rrënja e projektit:

```powershell
streamlit run app\streamlit_app.py
```

Aplikacioni pranon edhe vetëm titull ose vetëm përmbajtje. Input-i bosh bllokohet, ndërsa titulli pa përmbajtje dhe tekstet me më pak se 20 fjalë shoqërohen me paralajmërim. Modeli analizon vetëm formën gjuhësore të tekstit; nuk kontrollon burime, URL, autorë, data ose fakte reale.

## Testet

Ekzekuto:

```powershell
python -m pytest
```

Testet kontrollojnë loader-in, preprocessing-un, linguistic features, modelet hibride, grupet pa leakage, pragjet, strukturën e prediction dhe sjelljen kryesore të aplikacionit Streamlit.

## Raportet

Detajet e punës ditore ruhen te `reports/`.

Raporti aktual:

- `reports/day1_dataset_audit.md`
- `reports/day2_baseline_model.md`
- `reports/day3_linguistic_features.md`
- `reports/day4_linguistic_feature_analysis.md`
- `reports/day5_hybrid_model.md`
- `reports/day6_model_quality.md`
- `reports/day7_streamlit_app.md`

README shërben vetëm si hyrje profesionale dhe udhëzues ekzekutimi. Gjetjet e detajuara, numrat e datasetit dhe problemet e validimit ruhen në raportet përkatëse.

## Roadmap

Janë përfunduar pipeline-i i datasetit, preprocessing-u bazë, TF-IDF baseline, linguistic features, modeli hibrid, error analysis, probability calibration dhe versioni i parë funksional i aplikacionit Streamlit.

Hapi i ardhshëm është testimi më i gjerë me lajme të reja në shqip, përmirësimi i paraqitjes së rezultateve sipas gjetjeve dhe përgatitja e një vlerësimi përfundimtar pa ngatërruar klasifikimin gjuhësor me fact-checking.
