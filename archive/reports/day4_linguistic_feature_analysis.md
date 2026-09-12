# Dita 4 - Analiza e karakteristikave gjuhësore

Qëllimi i Ditës 4 ishte të kuptohej nëse karakteristikat gjuhësore të nxjerra në Ditën 3 japin sinjale dalluese midis lajmeve `real` dhe `fake`.

Nuk u ndërtua aplikacion Streamlit, nuk u përdor XLM-RoBERTa dhe nuk u shtua SHAP.

## Kontrolli i Ditës 3

Dita 3 ishte në rregull dhe nuk kishte bllokues për vazhdim.

U verifikua që ekzistojnë:

- `src/features/linguistic_features.py`
- `src/features/build_linguistic_features.py`
- `data/processed/linguistic_features.csv`
- `reports/day3_feature_summary.csv`
- `reports/day3_linguistic_features.md`

U kontrollua edhe që funksioni për një lajm të vetëm punon:

```python
extract_linguistic_features(title, content)
```

Dataset-i me features ka:

- 3994 rreshta
- 37 kolona
- 1998 artikuj real
- 1996 artikuj fake

## Kontrolli i cilësisë së features

U krijua script-i:

```text
src/features/analyze_linguistic_features.py
```

Ky script gjeneron:

```text
reports/day4_feature_quality.json
reports/day4_feature_comparison.csv
reports/day4_linguistic_only_model_metrics.json
reports/figures/day4_word_count_distribution.png
reports/figures/day4_marker_count_means.png
reports/figures/day4_ratio_feature_means.png
reports/figures/day4_top_effect_sizes.png
```

Kontrollet e cilësisë:

| Kontrolli | Rezultati |
| --- | ---: |
| Duplicate `article_id` | 0 |
| Duplicate rows | 0 |
| Missing values total | 0 |
| Missing numeric values | 0 |
| Infinite values | 0 |
| Ratio jashtë intervalit 0-1 | 0 |
| Count-e negative | 0 |
| Artikuj me `word_count < 20` | 1 |
| Numeric features konstante | 0 |

Kolonat tekstuale si `sensational_found`, `source_indicators_found` dhe `uncertainty_found` kanë vlera bosh kur nuk gjendet asnjë shprehje. Këto boshllëqe janë të pritshme dhe nuk janë problem numerik.

Përfundim: feature dataset është në gjendje të mirë për analizë fillestare.

## Karakteristikat e krahasuara

U krahasuan këto feature kryesore:

- `word_count`
- `sentence_count`
- `avg_sentence_length`
- `exclamation_count`
- `question_count`
- `uppercase_word_ratio`
- `sensational_count`
- `sensational_ratio`
- `source_indicator_count`
- `source_indicator_ratio`
- `uncertainty_count`
- `uncertainty_ratio`
- `diacritic_ratio`
- `title_length`
- `content_length`

Për secilën u llogarit:

- mean për real/fake
- median për real/fake
- min/max
- standard deviation
- diferenca `fake - real`
- Mann-Whitney U p-value
- t-test p-value
- Cohen's d si effect size i thjeshtë

Tabela e plotë ruhet te:

```text
reports/day4_feature_comparison.csv
```

## Dallime fillestare

Këto janë disa nga dallimet më të dukshme sipas mesatareve:

| Feature | Fake mean | Real mean | Fake - Real | Cohen's d |
| --- | ---: | ---: | ---: | ---: |
| `content_length` | 675.6954 | 1671.2007 | -995.5053 | -0.6982 |
| `word_count` | 131.7420 | 291.4124 | -159.6704 | -0.6649 |
| `diacritic_ratio` | 0.0620 | 0.0726 | -0.0106 | -0.6684 |
| `sentence_count` | 6.8131 | 14.4695 | -7.6563 | -0.5988 |
| `source_indicator_count` | 0.2395 | 0.8148 | -0.5753 | -0.4746 |
| `title_length` | 79.7174 | 69.6141 | 10.1033 | 0.4747 |
| `uncertainty_count` | 0.3592 | 0.9309 | -0.5717 | -0.3745 |
| `sensational_ratio` | 0.0005 | 0.0001 | 0.0004 | 0.1743 |

Interpretim fillestar:

- Artikujt real janë dukshëm më të gjatë në numër fjalësh, fjalish dhe karakteresh.
- Artikujt fake kanë tituj mesatarisht më të gjatë.
- Artikujt real kanë më shumë tregues burimi, si `sipas`, `studimi`, `policia`, etj.
- Artikujt fake kanë pak më shumë sinjal sensacional/clickbait, por dallimi është i vogël.
- `diacritic_ratio` del më i lartë te real, që mund të tregojë shkrim më standard, por duhet analizuar me kujdes.
- `exclamation_count` dhe `question_count` nuk duken shumë të forta si sinjale të vetme.

Këto nuk provojnë që një lajm është real ose fake. Janë vetëm sinjale gjuhësore që mund të ndihmojnë analizën dhe modelin.

## Grafikët e krijuar

U krijuan 4 grafikë:

```text
reports/figures/day4_word_count_distribution.png
reports/figures/day4_marker_count_means.png
reports/figures/day4_ratio_feature_means.png
reports/figures/day4_top_effect_sizes.png
```

Grafikët tregojnë:

- shpërndarjen e gjatësisë së tekstit për real/fake;
- krahasimin e count-eve për sensational/source/uncertainty/pikësim;
- krahasimin e ratio-ve;
- feature-t me dallimet më të mëdha sipas Cohen's d.

## Testet statistikore

U përdorën Mann-Whitney U dhe t-test për feature-t kryesore.

Shumë p-value dolën shumë të vogla, por kjo duhet interpretuar me kujdes sepse dataset-i ka shumë artikuj. Prandaj më e dobishme është të shikohet edhe effect size.

Feature me effect size më të dukshëm:

- `content_length`
- `diacritic_ratio`
- `word_count`
- `sentence_count`
- `source_indicator_count`
- `title_length`

Feature me effect size më të dobët:

- `exclamation_count`
- `question_count`
- `uppercase_word_ratio`
- `sensational_count`

## Model vetëm me karakteristika gjuhësore

U provua një model shumë i thjeshtë:

```text
Linguistic numeric features + Logistic Regression
```

U përdor e njëjta ndarje train/test nga Dita 2.

Rezultatet:

| Metrika | Vlera |
| --- | ---: |
| Accuracy | 0.8273 |
| Precision fake | 0.8355 |
| Recall fake | 0.8145 |
| F1 fake | 0.8249 |

Confusion matrix:

|  | Pred real | Pred fake |
| --- | ---: | ---: |
| True real | 336 | 64 |
| True fake | 74 | 325 |

Krahasim me Ditën 2:

| Model | Accuracy | F1 fake |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 0.8986 | 0.8936 |
| Linguistic features + Logistic Regression | 0.8273 | 0.8249 |

Modeli vetëm me features gjuhësore është më i dobët se TF-IDF, por rezultati tregon se këto features kanë sinjal parashikues.

Modeli u ruajt te:

```text
models/linguistic_features_logreg.joblib
```

## Feature-t më premtuese

Feature që duken më të dobishme:

- `content_length`
- `word_count`
- `sentence_count`
- `diacritic_ratio`
- `source_indicator_count`
- `source_indicator_ratio`
- `title_length`

Feature që duken më të dobëta si sinjale të vetme:

- `exclamation_count`
- `question_count`
- `uppercase_word_ratio`
- `sensational_count`

Feature që duhen përmirësuar:

- lista e fjalëve sensacionale;
- lista e source markers;
- lista e shprehjeve të pasigurisë;
- kontrolli për fjalë pa diakritika;
- matja e fjalive;
- normalizimi i count-eve sipas gjatësisë së tekstit.

## Kufizime

- Analiza është përshkruese dhe fillestare.
- Features nuk provojnë vërtetësinë faktike të lajmit.
- Shumë dallime lidhen me gjatësinë e artikujve.
- P-value mund të dalin shumë të vogla për shkak të madhësisë së datasetit.
- Listat e fjalëve/shprehjeve janë manuale dhe fillestare.
- Modeli feature-only nuk është model final.

## Hapi tjetër

Në Ditën 5 rekomandohet:

- ndërtimi i një modeli hibrid `TF-IDF + linguistic features`;
- krahasim i qartë midis TF-IDF baseline, feature-only model dhe modelit hibrid;
- analizë e gabimeve për rastet ku modelet gabojnë;
- ruajtje e tabelave të rezultateve për raportin final;
- fillimi i shpjegimeve të thjeshta për prediction bazuar në features.
