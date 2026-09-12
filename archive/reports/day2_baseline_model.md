# Dita 2 - Baseline model

Qëllimi i Ditës 2 ishte kalimi nga dataset i organizuar në një rrjedhë minimale që bën parashikim:

```text
dataset -> preprocessing -> train/test split -> TF-IDF -> Logistic Regression -> evaluation -> prediction
```

Nuk u ndërtua aplikacion Streamlit, nuk u përdor XLM-RoBERTa dhe nuk u shtuan karakteristika gjuhësore të avancuara.

## Kontrolli i datasetit

U përdor dataset-i i krijuar në Ditën 1:

```text
data/processed/articles.csv
```

Kontrolli fillestar:

- Rreshta total: 3994
- Kolona: 9
- Real/true: 1998
- Fake: 1996
- Kolonat kryesore ekzistojnë: `title`, `content`, `raw_text`, `label`, `label_name`, `pair_id`

Problemet e njohura nga Dita 1 mbeten të rëndësishme:

- ka 41 rreshta me tekste të duplikuara në 20 grupe
- disa `pair_id` nuk janë të pranishme në të dy klasat

## Preprocessing

U krijua moduli:

```text
src/preprocessing/clean_text.py
```

Preprocessing është qëllimisht minimal:

- trajton vlerat bosh
- normalizon hapësirat dhe rreshtat e rinj
- bashkon `title` dhe `content` në kolonën `model_text`
- ruan shkronjat shqipe si `ë` dhe `ç`
- nuk heq pikësimin
- nuk ndryshon shkronjat e mëdha
- nuk heq stopwords

Dataset-i i pastruar u ruajt te:

```text
data/interim/articles_clean.csv
```

## Ndarja train/test

U përdor ndarje rreth 80/20:

- Train: 3195 artikuj
- Test: 799 artikuj

Ndarja u bë sipas `pair_id` me `GroupShuffleSplit`, që artikujt me të njëjtin `pair_id` të mos ndahen midis train dhe test. Kjo ul rrezikun e data leakage.

Shpërndarja e labels:

| Split | Real | Fake |
| --- | ---: | ---: |
| Train | 1598 | 1597 |
| Test | 400 | 399 |

File-et e ndarjes:

```text
data/interim/train.csv
data/interim/test.csv
```

## Modeli

U krijua baseline model:

```text
TF-IDF + Logistic Regression
```

Parametrat kryesorë:

- `TfidfVectorizer(lowercase=False)`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_features=30000`
- `LogisticRegression(max_iter=1000, class_weight="balanced")`

`lowercase=False` u përdor që kapitalizimi të mos hiqet automatikisht.

Modeli u ruajt te:

```text
models/baseline_tfidf_logreg.joblib
```

## Rezultatet

Metrikat në test set:

| Metrika | Vlera |
| --- | ---: |
| Accuracy | 0.8986 |
| Precision për fake | 0.9392 |
| Recall për fake | 0.8521 |
| F1 për fake | 0.8936 |

Confusion matrix, rreshtat janë label reale dhe kolonat janë parashikime:

|  | Pred real | Pred fake |
| --- | ---: | ---: |
| True real | 378 | 22 |
| True fake | 59 | 340 |

Metrikat e plota u ruajtën te:

```text
reports/day2_metrics.json
```

## Prediction helper

U krijua:

```text
src/models/predict.py
```

Funksioni kryesor:

```python
predict_news(title, content)
```

Shembull nga test set:

```text
Titulli: Po, pajtohem me ambasadorin e ShBA-së
Label real: real
Parashikimi: real
Probabilitet real: 0.8995
Probabilitet fake: 0.1005
```

Shembull manual:

```text
Titulli: Qeveria njofton masa të reja ekonomike
Përmbajtja: Sipas njoftimit zyrtar, masat do të hyjnë në fuqi javën e ardhshme.
Parashikimi: fake
Probabilitet real: 0.4494
Probabilitet fake: 0.5506
```

Ky shembull manual tregon qartë kufizimin e modelit: ai nuk bën verifikim faktik, por vetëm klasifikim sipas modelit dhe fjalëve që ka mësuar nga dataset-i.

## Kufizime

- Modeli është vetëm baseline, jo model final.
- Nuk përdor ende karakteristika gjuhësore të dedikuara.
- Nuk përdor metadata.
- Nuk analizon burimin, datën, autorin ose faktet jashtë tekstit.
- TF-IDF nuk kupton kuptimin e thellë të fjalisë.
- Probabilitetet janë probabilitete sipas modelit, jo siguri absolute.
- Duplikimet në dataset duhet të analizohen më tej.

## Hapi tjetër

Në Ditën 3 duhet të punohet me:

- analizë më të thelluar të gabimeve
- krahasim i rezultateve train/test
- karakteristika të thjeshta gjuhësore, si gjatësia e titullit, numri i fjalëve, pikësimi, shkronjat e mëdha dhe prania e `ë/ç`
- pipeline më i qartë për features
- ruajtje e rezultateve në tabela raporti
