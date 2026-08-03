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
    external/
      external_news.csv
      README.md
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
    day8_streamlit_improvements.md
    day9_system_testing.md
    day9_system_test_metrics.json
    day9_system_test_cases.csv
    day9_demo_examples.csv
    day10_external_dataset.md
    day10_external_dataset_audit.json
    day10_external_similarity_review.csv
    day11_external_evaluation.md
    day11_external_metrics.json
    day11_external_predictions.csv
    figures/
  src/
    data/
      load_dataset.py
      validate_dataset.py
      build_dataset.py
      validate_external_dataset.py
    preprocessing/
      clean_text.py
    models/
      train_model.py
      train_hybrid_model.py
      analyze_model_quality.py
      evaluate_app_system.py
      evaluate_external_dataset.py
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
    test_app_system.py
    test_streamlit_app.py
    test_external_dataset.py
    test_external_evaluation.py
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

Aplikacioni përdor modelin e kalibruar `TF-IDF + Logistic Regression`. Përdoruesi mund të vendosë titullin, përmbajtjen ose të dyja dhe të marrë:

- vendimin `likely_real`, `uncertain` ose `likely_fake`, të shpjeguar me formulim të kuptueshëm;
- probabilitetet Real/Fake si metrics dhe progress bars;
- shpjegim të posaçëm kur probabiliteti ndodhet në zonën `uncertain` 30%-70%;
- karakteristika të vëzhguara në tekst dhe interpretim njerëzor të gjatësisë, pikëçuditëseve, kapitalizimit, diakritikave dhe marker-ave gjuhësorë;
- paralajmërimin pranë rezultatit se aplikacioni nuk bën verifikim faktik.

Hape aplikacionin nga rrënja e projektit:

```powershell
streamlit run app\streamlit_app.py
```

Aplikacioni pranon edhe vetëm titull ose vetëm përmbajtje. Input-i bosh ose pa shkronja/numra bllokohet, ndërsa titulli pa përmbajtje, tekstet me më pak se 20 fjalë dhe tekstet mbi 20,000 karaktere shoqërohen me paralajmërim. Për të mbrojtur qëndrueshmërinë, input-i mbi 100,000 karaktere nuk analizohet. Input-i normalizohet në Unicode NFC për të trajtuar në mënyrë të njëjtë shkronjat e kombinuara shqipe. Modeli analizon vetëm formën gjuhësore të tekstit; nuk kontrollon burime, URL, autorë, data ose fakte reale.

## Testimi i Sistemit

Ekzekuto paketën e testimit të Ditës 9:

```powershell
python src\models\evaluate_app_system.py
```

Script-i teston input-et ideale dhe joideale, kontrollon probabilitetet dhe pragjet në test set-in pa leakage dhe përzgjedh shembuj demonstrues e gabime reale të modelit.

Output-et kryesore:

```text
reports/day9_system_test_metrics.json
reports/day9_system_test_cases.csv
reports/day9_demo_examples.csv
reports/day9_system_testing.md
```

## Dataseti i Jashtëm

Dataseti pilot i Ditës 10 përmban 40 raste të reja në shqip: 20 `real` dhe 20
`fake`. Pesë temat kanë nga 8 raste secila dhe çdo temë ndahet në 4 `real` dhe
4 `fake`. Tekstet janë përmbledhje manuale; URL-ja dhe prova e etiketimit ruhen
në kolona të veçanta.

Dataseti ruhet te:

```text
data/external/external_news.csv
```

Ekzekuto vetëm kontrollin e cilësisë dhe të mbivendosjes me train set-in:

```powershell
python src\data\validate_external_dataset.py
```

Kjo komandë nuk ngarkon modelin dhe nuk bën prediction. Ajo krijon:

```text
reports/day10_external_dataset_audit.json
reports/day10_external_similarity_review.csv
```

Metoda e mbledhjes dhe kufizimet shpjegohen te `data/external/README.md`, ndërsa
rezultatet e auditimit te `reports/day10_external_dataset.md`.

## Vlerësimi i Jashtëm

Vlerësimi i Ditës 11 përdor modelin e ruajtur dhe të njëjtin funksion
`predict_news_for_app()` si aplikacioni. Modeli nuk ritrajnohet dhe pragjet
`0.30/0.70` nuk ndryshohen.

Ekzekuto:

```powershell
python src\models\evaluate_external_dataset.py
```

Output-et kryesore janë:

```text
reports/day11_external_predictions.csv
reports/day11_external_metrics.json
reports/day11_external_by_topic.csv
reports/day11_external_by_label.csv
reports/day11_external_by_length.csv
reports/day11_external_by_source.csv
reports/day11_external_errors.csv
reports/day11_external_interesting_cases.csv
reports/day11_external_confusion_matrix.csv
reports/figures/day11_external_confusion_matrix.png
reports/day11_external_evaluation.md
```

Raporti krahason me kujdes rezultatet e jashtme me test set-in e brendshëm, por
nuk i trajton si dataset-e drejtpërdrejt të barabarta.

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
- `reports/day8_streamlit_improvements.md`
- `reports/day9_system_testing.md`
- `reports/day10_external_dataset.md`
- `reports/day11_external_evaluation.md`

README shërben vetëm si hyrje profesionale dhe udhëzues ekzekutimi. Gjetjet e detajuara, numrat e datasetit dhe problemet e validimit ruhen në raportet përkatëse.

## Roadmap

Janë përfunduar pipeline-i i datasetit, preprocessing-u bazë, TF-IDF baseline, linguistic features, modeli hibrid, error analysis, probability calibration, aplikacioni Streamlit, testimi end-to-end, përgatitja e datasetit të jashtëm dhe vlerësimi i parë jashtë corpus-it.

Hapi i ardhshëm është analiza e kontrolluar e domain shift-it dhe e bias-it të gjatësisë, duke ruajtur datasetin e jashtëm si benchmark të pandryshuar.
