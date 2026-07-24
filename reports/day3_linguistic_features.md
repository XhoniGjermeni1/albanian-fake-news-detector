# Dita 3 - Karakteristika gjuhësore

Qëllimi i Ditës 3 ishte ndërtimi i motorit të parë që analizon stilin gjuhësor të lajmit dhe nxjerr karakteristika të thjeshta, të kuptueshme dhe të përdorshme më vonë për analizë, shpjegim dhe përmirësim modeli.

Nuk u ndërtua aplikacion Streamlit, nuk u përdor XLM-RoBERTa dhe nuk u shtua SHAP.

## Kontrolli i Ditës 2

Dita 2 ishte në rregull dhe nuk kishte bllokues për vazhdim.

U verifikua që ekzistojnë dhe funksionojnë:

- `data/interim/articles_clean.csv`
- `data/interim/train.csv`
- `data/interim/test.csv`
- `models/baseline_tfidf_logreg.joblib`
- `reports/day2_metrics.json`
- `src/models/predict.py`

Kontrolli i shpejtë:

- Dataset i pastruar: 3994 rreshta
- Train: 3195 rreshta
- Test: 799 rreshta
- `model_text` bosh: 0
- Accuracy e baseline modelit të Ditës 2: 0.8986
- `predict_news(title, content)` kthen label dhe probabilitete

## Modulet e krijuara

U krijuan:

```text
src/features/linguistic_features.py
src/features/build_linguistic_features.py
tests/test_linguistic_features.py
```

Output-et e Ditës 3:

```text
data/processed/linguistic_features.csv
reports/day3_feature_summary.csv
```

`linguistic_features.csv` ka 3994 rreshta dhe 37 kolona.

## Karakteristikat e implementuara

Karakteristika strukturore:

- `word_count`
- `sentence_count`
- `character_count`
- `avg_word_length`
- `avg_sentence_length`
- `title_length`
- `content_length`

Karakteristika të pikësimit:

- `exclamation_count`
- `question_count`
- `comma_count`
- `quote_count`
- `ellipsis_count`
- `exclamation_ratio`
- `question_ratio`

Karakteristika të kapitalizimit:

- `uppercase_word_count`
- `uppercase_word_ratio`
- `uppercase_char_ratio`
- `title_excessive_uppercase`

Karakteristika specifike për shqipen:

- `e_count`
- `c_count`
- `diacritic_count`
- `diacritic_ratio`
- `possible_missing_diacritic_count`
- `possible_missing_diacritic_words`

Fjalë dhe shprehje sensacionale/clickbait:

- `sensational_count`
- `sensational_ratio`
- `sensational_found`

Tregues burimi ose atribuimi:

- `source_indicator_count`
- `source_indicator_ratio`
- `source_indicators_found`

Shprehje pasigurie ose spekulimi:

- `uncertainty_count`
- `uncertainty_ratio`
- `uncertainty_found`

## Si nxirren features për një lajm

Funksioni kryesor për një lajm është:

```python
from src.features.linguistic_features import extract_linguistic_features

features = extract_linguistic_features(
    title="Lajm i fundit",
    content="Ja çfarë ndodhi. Sipas policia, thuhet se ka gjasa të ketë zhvillime."
)
```

Ky funksion kthen një `dict` me karakteristikat gjuhësore për atë lajm.

## Si nxirren features për gjithë datasetin

Për gjithë datasetin përdoret:

```powershell
python src\features\build_linguistic_features.py
```

Ky script:

- lexon `data/interim/articles_clean.csv`
- nxjerr karakteristikat për çdo artikull
- ruan tabelën e plotë te `data/processed/linguistic_features.csv`
- ruan krahasimin mesatar real/fake te `reports/day3_feature_summary.csv`

## Shembuj konkretë

Shembuj real:

| Article | Words | Sentences | Source indicators | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| `true_1` | 296 | 11 | 2, `studimi` | 0 |
| `true_3` | 93 | 6 | 1, `policia` | 0 |
| `true_5` | 318 | 14 | 1, `sipas` | 0 |

Shembuj fake:

| Article | Words | Sentences | Sensational | Uncertainty |
| --- | ---: | ---: | ---: | ---: |
| `fake_8` | 77 | 1 | 2, `e pabesueshme` | 0 |
| `fake_24` | 66 | 10 | 1, `nuk do ta besoni` | 0 |
| `fake_25` | 146 | 5 | 1, `e pabesueshme` | 1, `mund të` |

## Dallime fillestare real/fake

Këto janë mesatare të thjeshta, jo përfundime finale.

| Feature | Fake | Real | Fake - Real |
| --- | ---: | ---: | ---: |
| `word_count` | 131.7420 | 291.4124 | -159.6704 |
| `sentence_count` | 6.8131 | 14.4695 | -7.6564 |
| `character_count` | 756.4128 | 1741.8148 | -985.4020 |
| `title_length` | 79.7174 | 69.6141 | 10.1033 |
| `comma_count` | 6.2886 | 16.5155 | -10.2269 |
| `ellipsis_count` | 0.2325 | 0.1391 | 0.0934 |
| `sensational_count` | 0.0451 | 0.0225 | 0.0226 |
| `source_indicator_count` | 0.2395 | 0.8148 | -0.5753 |
| `uncertainty_count` | 0.3592 | 0.9309 | -0.5717 |

Vëzhgime fillestare:

- Artikujt real janë mesatarisht më të gjatë.
- Artikujt fake kanë tituj mesatarisht më të gjatë.
- Artikujt real kanë më shumë presje dhe thonjëza, gjë që lidhet pjesërisht me gjatësinë më të madhe të teksteve.
- Fjalët sensacionale shfaqen pak më shpesh te fake.
- Treguesit e burimit shfaqen më shpesh te real.
- Shprehjet e pasigurisë dalin më shpesh te real në këtë matje fillestare, prandaj kjo duhet analizuar më tej para interpretimit.

## Kufizime

- Karakteristikat janë sinjale gjuhësore, jo prova faktike.
- Lista e fjalëve sensacionale është fillestare dhe duhet zgjeruar me kujdes.
- Disa fjalë mund të kenë kuptime të ndryshme sipas kontekstit.
- Numërimi i fjalive është bazik dhe bazohet te shenjat `.`, `!`, `?`.
- Kontrolli për mungesë diakritikash është vetëm sinjal i përafërt.
- Mesataret ndikohen nga fakti që artikujt real janë shumë më të gjatë.

## Hapi tjetër

Në Ditën 4 rekomandohet:

- analizë më e mirë statistikore e features
- krahasim me grafikë për real/fake
- normalizim i disa features sipas gjatësisë së tekstit
- bashkim i TF-IDF me karakteristika gjuhësore në një model të vetëm
- krahasim i baseline TF-IDF kundrejt modelit me features gjuhësore
