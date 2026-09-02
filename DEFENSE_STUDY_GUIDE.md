# Udhëzues intensiv për mbrojtjen e diplomës

Ky material është ndërtuar mbi implementimin dhe rezultatet reale të këtij
repository. Qëllimi nuk është të mësosh përmendësh çdo skript historik, por të
kuptosh rrjedhën shkencore dhe kodin që ekzekutohet sot.

## Si ta përdorësh

1. Ndiq programin 3-ditor në fund të materialit.
2. Për çdo pjesë, lexo shpjegimin dhe skedarin e treguar.
3. Përgjigju pyetjeve të kontrollit pa parë tekstin.
4. Ekzekuto komandat me environment-in `.venv`.
5. Në Ditën 3, bëj prezantimin dhe provimin simulues me zë.

Mos u përpiq të mësosh përmendësh të gjitha shifrat. Mëso mirë shifrat në
seksionin “Numrat që duhen ditur” dhe përdor raportet për detajet e tjera.

## Harta e repository-t për mbrojtje

### 1. Kodi kryesor që duhet ta kuptosh

Këto janë modulet e rrjedhës reale finale:

- `src/data/load_dataset.py`: lexon corpus-in dhe krijon rreshtat e dataset-it;
- `src/preprocessing/clean_text.py`: NFC, hapësirat dhe `title + content`;
- `src/features/linguistic_features.py`: sinjalet gjuhësore për analizë dhe UI;
- `src/models/prediction_utils.py`: shpjegimi dhe pragjet `0.30/0.70`;
- `src/models/predict_final.py`: kontrata finale e prediction-it;
- `app/streamlit_app.py`: entrypoint-i, cache dhe rrjedha e formularit;
- `app/streamlit_ui.py`: validimi, tekstet dhe shfaqja e rezultatit;
- `models/final_model_v1_manifest.json`: konfigurimi i ngrirë dhe metrikat;
- `tests/test_final_model.py` dhe `tests/test_streamlit_app.py`: testet
  përfaqësuese të regresionit dhe integrimit.

### 2. Analizat që duhet t’i kuptosh, jo t’i mësosh rresht për rresht

- `src/models/train_model.py`: baseline Word TF-IDF + Logistic Regression;
- `src/features/analyze_linguistic_features.py`: statistikat real/fake;
- `src/models/train_hybrid_model.py`: linguistic-only dhe modeli hibrid;
- `src/models/analyze_model_quality.py`: error analysis, groups dhe calibration-i
  i baseline-it;
- `src/models/analyze_length_domain_shift.py`: length bias dhe domain shift;
- `src/models/compare_tfidf_representations.py`: Word, Character dhe kombinimi;
- `src/models/compare_classifiers.py`: Logistic Regression, Linear SVM dhe
  Complement Naive Bayes;
- `src/models/tune_linear_svm.py`: krahasimi i vlerave të `C`;
- `src/models/calibrate_linear_svm.py`: sigmoid/isotonic dhe pragjet;
- `src/models/finalize_model.py`: verifikimi, hash-i dhe ngrirja e modelit;
- raportet kryesore `day1`, `day4`, `day5`, `day12` deri `day17`.

Për këto skripte duhet të dish pyetjen që zgjidhin, protokollin e vlerësimit dhe
përfundimin. Nuk ke nevojë të shpjegosh kodin e grafikëve, eksportet CSV apo
gjenerimin automatik të Markdown-it.

### 3. Skedarët historikë që nuk duhen mësuar

- `src/models/predict.py`: API e modeleve të vjetra;
- raportet ditore 2–11 dhe 18–20, përveç kur kërkon një rezultat specifik;
- CSV/JSON-të e detajuara të çdo fold-i;
- modelet eksperimentale lokale `*.joblib`;
- kodi i grafikëve dhe formatimit të raporteve;
- `app/style.css`, përveç nëse pyetesh posaçërisht për paraqitjen;
- `notebooks/01_dataset_audit.ipynb`, përveç përmbledhjes së auditimit.

Këta skedarë ruhen për riprodhueshmëri dhe histori akademike. Runtime-i final
nuk varet prej tyre.

## Numrat që duhen ditur

| Fakt | Vlera |
|---|---:|
| Artikuj total | 3,994 |
| Real / fake | 1,998 / 1,996 |
| Train / test fillestar | 3,195 / 799 |
| Kopje ekzakte train–test të përjashtuara | 7 |
| Test final i pastër | 792: 399 real, 393 fake |
| CV train groups | 1,586 |
| Modeli final | Word + Character TF-IDF + Linear SVM `C=1.0` |
| Calibration / pragje | sigmoid / `0.30–0.70` |
| Accuracy / F1 weighted i brendshëm | 91.16% / 91.16% |
| Recall real / fake | 92.48% / 89.82% |
| Confusion matrix finale | `[[369, 30], [40, 353]]` |
| Brier / log loss / ECE | 0.0658 / 0.2192 / 0.0285 |
| Coverage / strong accuracy | 91.04% / 94.31% |
| Benchmark i jashtëm | 40: 20 real, 20 fake |
| Accuracy baseline / final jashtë | 47.5% / 60.0% |
| Confusion matrix finale jashtë | `[[10, 10], [6, 14]]` |

---

# Pjesa 1 – Dataset-i dhe problemi

## Problemi që zgjidh projekti

Sistemi merr titullin dhe përmbajtjen e një lajmi në shqip dhe vlerëson sa i
ngjashëm është teksti me modelet gjuhësore të klasave `real` dhe `fake` në
corpus. Ai kthen probabilitetet sipas modelit dhe një vendim me tri nivele.

Formulimi i saktë për mbrojtje është:

> “Ky është klasifikues gjuhësor i lajmeve, jo motor fact-checking. Ai mëson
> pattern-e statistikore nga teksti i corpus-it dhe nuk kontrollon nëse ngjarja
> ka ndodhur.”

## Si është ndërtuar corpus-i

Corpus-i raw ka këto ndarje të rëndësishme:

- `true/`: 1,998 artikuj;
- `fake/`: 1,996 artikuj;
- `true-pos/` dhe `fake-pos/`: versione POS-tagged;
- `true-meta-information/` dhe `fake-meta-information/`: metadata.

Projekti final përdor vetëm tekstet e plota `true/` dhe `fake/`. POS dhe
metadata u audituan, por nuk u përdorën si input të modelit.

Në skedarët raw nuk ka kolonë të veçantë titulli. Loader-i interpreton rreshtin
e parë si `title` dhe rreshtat e tjerë si `content`. Raw-i nuk ndryshohet.

## Identifikuesit

- `article_id`: identifikues unik i rreshtit, p.sh. `true_74` ose `fake_74`;
- `pair_id`: numri i emrit të skedarit, p.sh. `74.txt → 74`;
- `label`: `0 = real`, `1 = fake`;
- `source_split`: `true` ose `fake`.

I njëjti `pair_id` zakonisht lidh artikullin real dhe fake të të njëjtit çift.
Ka 1,994 çifte të plota dhe gjashtë ID pa partner: katër vetëm `true` dhe dy
vetëm `fake`.

## Train/test dhe duplicate texts

Ndarja fillestare ishte rreth 80/20 me `GroupShuffleSplit`, duke përdorur
`pair_id` si group:

- train: 3,195 artikuj;
- test fillestar: 799 artikuj;
- nuk kishte mbivendosje `article_id` ose `pair_id`.

Auditimi kishte gjetur 41 rreshta të duplikuar në 20 grupe. Shtatë tekste
ekzakte në test kishin kopje në train, edhe pse `pair_id` ishte i ndryshëm:

`true_74`, `fake_110`, `fake_486`, `fake_786`, `fake_1665`, `fake_1853`,
`fake_1966`.

Ato u përjashtuan vetëm nga evaluation-i, duke lënë 792 raste. Arsyeja është se
një model mund ta njohë tekstin fjalë për fjalë dhe të japë një rezultat
artificialisht të lartë. Kjo është **data leakage**: informacion nga jashtë
train-it real hyn në mënyrë të drejtpërdrejtë ose indirekte në trajnim/zgjedhje.

## Pse Unicode NFC është i rëndësishëm

Shkronja `ë` mund të ruhet si një kod i vetëm (`ë`) ose si `e` plus një shenjë
kombinuese. Vizualisht duken njësoj, por për tokenizer-in dhe character
n-grams janë sekuenca të ndryshme. NFC i kthen në një përfaqësim kanonik të
njëjtë. Kështu:

- input-i NFD dhe NFC japin të njëjtin prediction;
- `ë/ç` nuk copëtohen në mënyra të ndryshme;
- duplicate checks dhe n-grams bëhen më të qëndrueshme.

Eksperimenti i Ditës 13 dha diferencë zero mes versionit NFD të normalizuar dhe
tekstit origjinal NFC.

### Pyetje kontrolli

1. Cili është ndryshimi mes `article_id` dhe `pair_id`?
2. Pse ndarja vetëm sipas rreshtave do të ishte e rrezikshme?
3. Pse u hoqën shtatë raste nga evaluation-i, por jo nga raw dataset-i?
4. Si mund të duken njësoj dy vargje Unicode, por të jenë ndryshe për modelin?
5. A përdor modeli POS tags ose metadata?

---

# Pjesa 2 – Preprocessing-u real

Kodi kryesor është `src/preprocessing/clean_text.py`.

## Çfarë bëhet

`normalize_spaces(text)`:

1. kthen vlerën bosh/`NaN` në `""`;
2. aplikon `unicodedata.normalize("NFC", text)`;
3. bashkon hapësirat, tab-et dhe rreshtat e shumëfishtë në një hapësirë;
4. heq hapësirat në fillim dhe fund.

`combine_title_content(title, content)` normalizon të dyja dhe krijon:

```text
Titulli. Përmbajtja
```

Nëse njëra pjesë mungon, përdoret vetëm pjesa që ekziston.

`prepare_text_dataframe(df)` krijon `title_clean`, `content_clean` dhe
`model_text` për dataset-in.

## Çfarë nuk pastrohet

- nuk bëhet lowercase;
- nuk hiqet pikësimi;
- nuk hiqen `ë/ç` ose diakritikat;
- nuk hiqen stopwords;
- nuk bëhet stemming ose lemmatization;
- nuk fshihen në mënyrë agresive numrat, URL-të ose fjalët e rralla.

Kjo është zgjedhje e qëllimshme. Kapitalizimi, pikësimi, variantet e shkrimit
dhe diakritikat janë pjesë e stilit që analizohet. Të dy vectorizer-at finalë
kanë `lowercase=False`. Word TF-IDF nuk përdor çdo shenjë pikësimi si token më
vete, por Character TF-IDF dhe shpjegimi gjuhësor mund ta shohin strukturën.

Preprocessing-u minimal gjithashtu e bën pipeline-in më të kuptueshëm dhe ul
rrezikun që të humbasim sinjal të dobishëm për shqipen.

### Pyetje kontrolli

1. Cilat janë tri punët kryesore të `normalize_spaces()`?
2. Si bashkohen titulli dhe përmbajtja?
3. Pse `lowercase=False` është në përputhje me qëllimin e projektit?
4. Pse nuk u hoqën stopwords në mënyrë agresive?
5. Cili funksion duhet të jetë identik në evaluation dhe në aplikacion?

---

# Pjesa 3 – TF-IDF

## Nga Bag of Words te TF-IDF

Marrim tre dokumente të vegjël:

```text
D1: lajm zyrtar
D2: lajm i fundit
D3: raport zyrtar
```

**Bag of Words** ndërton një fjalor dhe përfaqëson çdo dokument me numrin e
shfaqjeve të çdo fjale. Rendi i plotë i fjalisë humbet.

Një fjalor i thjeshtë mund të jetë:

```text
[lajm, zyrtar, i, fundit, raport]
D1 = [1, 1, 0, 0, 0]
D2 = [1, 0, 1, 1, 0]
D3 = [0, 1, 0, 0, 1]
```

**Term Frequency (TF)** mat sa shfaqet termi në dokument. Në scikit-learn
përdoret count-i i termit dhe më pas vektori normalizohet me normë L2.

**Document Frequency (DF)** është numri i dokumenteve ku termi shfaqet. Në
shembull, `lajm` ka `df=2`, ndërsa `fundit` ka `df=1`.

**Inverse Document Frequency (IDF)** i jep më pak peshë termave shumë të
zakonshëm dhe më shumë termave dallues. Me smoothing-un default të sklearn:

```text
idf(t) = log((1 + N) / (1 + df(t))) + 1
tfidf(t, d) = tf(t, d) × idf(t)
```

Pra një fjalë që shfaqet pothuajse kudo ka pak fuqi dalluese, ndërsa një fjalë
më specifike ka peshë më të madhe.

## Sparse feature vector

Modeli final ka deri në 80,000 dimensione:

- 30,000 word features;
- 50,000 character features.

Një artikull aktivizon vetëm një pjesë të vogël të tyre. Një **sparse matrix**
ruan vetëm pozicionet jo-zero, prandaj kursen shumë memorie dhe është shumë e
përshtatshme për modele lineare.

## Word n-grams

- unigram `(1)`: `lajm`, `i`, `fundit`;
- bigram `(2)`: `lajm i`, `i fundit`.

Modeli përdor `ngram_range=(1, 2)`, kështu ruan fjalë individuale dhe pak
kontekst lokal. P.sh. `lajm` është i përgjithshëm, ndërsa `lajm i fundit` ndahet
në bigram-e më informuese.

## Character n-grams

Për fjalën `lajme`, character 3-grams mund të jenë `laj`, `ajm`, `jme`.
N-grams `(3,5)` kapin:

- rrënjë dhe pjesë fjalësh;
- variante drejtshkrimore;
- prapashtesa/parapashtesa;
- fjalë të panjohura;
- ndryshime me `ë/ç`;
- pjesë të shprehjeve clickbait.

`analyzer="char_wb"` krijon n-grams brenda kufijve të fjalëve dhe shton
hapësira në skaje. Kjo kufizon n-grams arbitrare që kalojnë nga fundi i një
fjale te fillimi i tjetrës.

## `min_df` dhe `max_features`

- `min_df=2`: një feature duhet të shfaqet në të paktën dy dokumente; heq
  gabimet ose termat krejt unikë;
- `max_features`: kufizon fjalorin te features më të shpeshta, për memorie,
  shpejtësi dhe kontroll të kompleksitetit.

## Konfigurimi konkret final

```text
Word: analyzer=word, ngram=(1,2), min_df=2, max_features=30000
Char: analyzer=char_wb, ngram=(3,5), min_df=2, max_features=50000
Të dy: lowercase=False
```

Dy matricat bashkohen me `FeatureUnion`. Character TF-IDF u provua sepse
tekste të shkurtra, drejtshkrimi jo i njëtrajtshëm dhe morfologjia shqipe mund
të mos përfaqësohen mirë vetëm me fjalë të plota.

Në testin e brendshëm të Ditës 13:

| Përfaqësimi | Accuracy | F1 weighted | F1 fake |
|---|---:|---:|---:|
| Word | 88.38% | 88.38% | 88.08% |
| Character | 88.26% | 88.25% | 87.81% |
| Word + Character | **90.28%** | **90.27%** | **89.99%** |

Character-only nuk e kaloi Word në total, por ishte më i mirë në cohort-in e
vogël 30–60 fjalë: 88.89% kundrejt 77.78%. Kombinimi fitoi në total dhe uli
lidhjen e prediction-it me gjatësinë. Prandaj konfigurimi u fiksua për Ditën 14.

### Pyetje kontrolli

1. Çfarë humbet Bag of Words dhe çfarë ruajnë bigram-et?
2. Pse IDF ul peshën e fjalëve që shfaqen në shumë dokumente?
3. Çfarë do të thotë që vektori është sparse?
4. Çfarë problemi ndihmojnë të kapin character n-grams?
5. Pse kombinimi u zgjodh edhe pse Character-only nuk fitoi në total?

---

# Pjesa 4 – Logistic Regression baseline

Logistic Regression është classifier linear. Ai llogarit:

```text
score = w · x + b
P(fake) = sigmoid(score) = 1 / (1 + e^(-score))
```

- `x` është vektori TF-IDF;
- `w` janë koeficientët e mësuar;
- koeficient pozitiv shtyn drejt klasës `fake`;
- koeficient negativ shtyn drejt `real`;
- `w·x+b=0` është decision boundary;
- me pragun 0.5, `P(fake) >= 0.5` jep label `1`.

`class_weight="balanced"` rregullon peshën e çdo klase në raport me numrin e
rasteve. Dataset-i ynë është pothuajse i balancuar, prandaj efekti është i
vogël, por konfigurimi e bën qëllimin e balancimit eksplicit.

## Pse ishte baseline i mirë

- punon mirë me vektorë sparse dhe shumë-dimensionalë;
- është i shpejtë;
- jep probabilitete drejtpërdrejt;
- koeficientët mund të interpretohen;
- krijon një pikë reference të fortë para modeleve më të ndërlikuara.

Rezultati fillestar i Ditës 2 mbi 799 raste ishte:

- accuracy 89.86%;
- F1 fake 89.36%;
- confusion matrix `[[378, 22], [59, 340]]`.

Pas përjashtimit të shtatë kopjeve dhe me pipeline-in e kalibruar të krahasimit
final, baseline-i Word TF-IDF mori 88.38% accuracy/F1 weighted në 792 raste.
Këto shifra nuk janë kontradiktë: protokolli dhe numri i rasteve ndryshuan.

Nuk mbeti final sepse Word + Character + Linear SVM dha F1 më të lartë,
më pak FP/FN, calibration më të mirë dhe rezultat më të balancuar jashtë
corpus-it. Baseline-i u ruajt për krahasim dhe riprodhueshmëri.

### Pyetje kontrolli

1. Çfarë përfaqëson shenja e një koeficienti?
2. Çfarë është decision boundary në një model linear?
3. Çfarë ndodh në pragun 0.5?
4. Pse Logistic Regression ishte baseline i arsyeshëm për TF-IDF?
5. Pse rezultati i Ditës 2 nuk krahasohet verbërisht me rezultatin final?

---

# Pjesa 5 – Linguistic features

Funksioni kryesor është
`src/features/linguistic_features.py::extract_linguistic_features()`.

## Grupet e features

1. **Gjatësi/strukturë:** `word_count`, `sentence_count`, `character_count`,
   `avg_word_length`, `avg_sentence_length`, `title_length`, `content_length`.
2. **Pikësim:** count/ratio për `!`, `?`, presje, thonjëza dhe tri pika.
3. **Kapitalizim:** fjalë uppercase, ratio e shkronjave uppercase dhe titull me
   kapitalizim të tepruar.
4. **Diakritika:** numri/ratio i `ë` dhe `ç`, plus fjalë që mund t’i kenë humbur.
5. **Sensational markers:** p.sh. `tronditëse`, `skandal`, `ekskluzive`.
6. **Source markers:** p.sh. `sipas`, `konfirmoi`, `ministria`, `raporti`.
7. **Uncertainty markers:** p.sh. `thuhet`, `dyshohet`, `ndoshta`, `mund të`.

Listat e marker-ave janë manuale dhe fillestare. Ato nuk janë fjalor i plotë i
shqipes dhe nuk provojnë label-in.

## Çfarë tregoi analiza real/fake

| Feature | Fake mean | Real mean | Cohen's d, fake−real |
|---|---:|---:|---:|
| `content_length` | 675.70 | 1,671.20 | -0.698 |
| `word_count` | 131.74 | 291.41 | -0.665 |
| `diacritic_ratio` | 0.0620 | 0.0726 | -0.668 |
| `sentence_count` | 6.81 | 14.47 | -0.599 |
| `title_length` | 79.72 | 69.61 | +0.475 |
| `source_indicator_count` | 0.239 | 0.815 | -0.475 |
| `sensational_ratio` | 0.00046 | 0.00008 | +0.174 |

Në këtë corpus, fake ishin mesatarisht më të shkurtra, kishin tituj më të
gjatë, më pak diakritika dhe më pak source markers. Sensational markers kishin
drejtimin intuitiv, por effect size të vogël.

## Effect size, Cohen's d dhe p-value

**Effect size** mat sa i madh është dallimi, jo vetëm nëse dallimi mund të
zbulohet statistikisht.

```text
Cohen's d = (mean_fake − mean_real) / pooled_standard_deviation
```

- shenja tregon drejtimin;
- afërsisht `0.2` konsiderohet efekt i vogël, `0.5` mesatar, `0.8` i madh;
- kufijtë janë orientues, jo ligj universal.

**p-value** pyet sa të papajtueshme janë të dhënat me hipotezën e mungesës së
dallimit. Me rreth 4,000 raste edhe ndryshime të vogla mund të japin p-value
shumë të ulët. Prandaj duhen parë bashkë: madhësia e efektit, shpërndarja,
vlera praktike dhe bias-et e mundshme.

## Modelet linguistic-only dhe hybrid

| Modeli | Accuracy | F1 fake |
|---|---:|---:|
| TF-IDF only | 89.77% | 89.19% |
| Linguistic-only | 82.70% | 82.37% |
| TF-IDF + linguistic | 89.02% | 88.63% |
| Hybrid pa length features | 89.14% | 88.83% |

Hybrid-i nuk e kaloi TF-IDF; diferenca e accuracy ishte -0.75 pikë
përqindjeje. Heqja e features direkte të gjatësisë e përmirësoi hybrid-in vetëm
0.12 pikë, por përsëri nuk e kaloi baseline-in.

Arsyet e mundshme janë:

- TF-IDF tashmë kap shumë prej të njëjtave sinjale;
- features numerike janë pjesërisht redundante;
- sinjali i gjatësisë lidhet me dataset bias;
- listat manuale janë të kufizuara;
- shtimi i features nuk garanton informacion të ri.

Përfundimi final është i rëndësishëm:

> Linguistic features kanë vlerë për analizën e corpus-it dhe për explanation
> në aplikacion, por nuk përdoren si input i classifier-it final.

### Pyetje kontrolli

1. Cilat janë shtatë grupet e linguistic features?
2. Çfarë do të thotë `d=-0.665` për `word_count`?
3. Pse një p-value shumë e vogël nuk mjafton?
4. Çfarë rezultati dha modeli linguistic-only?
5. Pse linguistic explanation nuk duhet paraqitur si provë faktike?

---

# Pjesa 6 – Evaluation metrics

Në projekt, `fake=1` është klasa pozitive. Confusion matrix finale është:

```text
                    Prediction real   Prediction fake
True real (0)              369               30
True fake (1)               40              353
```

Pra:

- **TN = 369:** real i klasifikuar real;
- **FP = 30:** real i klasifikuar fake;
- **FN = 40:** fake i klasifikuar real;
- **TP = 353:** fake i klasifikuar fake.

## Formulat

```text
accuracy       = (TP + TN) / të gjitha rastet
precision_fake = TP / (TP + FP)
recall_fake    = TP / (TP + FN)
F1_fake        = 2 × precision × recall / (precision + recall)
recall_real    = TN / (TN + FP)
```

Për modelin final:

- accuracy: `(353+369)/792 = 91.16%`;
- precision fake: `353/(353+30) = 92.17%`;
- recall fake: `353/(353+40) = 89.82%`;
- F1 fake: 90.98%;
- recall real: `369/(369+30) = 92.48%`.

**Weighted F1** llogarit F1 për secilën klasë dhe i peshon sipas numrit të
rasteve të klasës. Këtu klasat janë pothuajse të balancuara, prandaj weighted
F1 është afër mesatares së dy F1-ve.

## Pse FP dhe FN kanë rëndësi

- FP: një lajm real shënohet fake; mund të krijojë alarm të rremë ose të dëmtojë
  besueshmërinë e burimit;
- FN: një lajm fake shënohet real; mund të ndihmojë përhapjen e keqinformimit.

Në këtë projekt FN janë veçanërisht të rrezikshme, por nuk mund të minimizohen
duke e quajtur çdo gjë fake, sepse atëherë FP bëhen shumë të larta. Duhet
balancë dhe raportim i të dyjave.

### Pyetje kontrolli

1. Cila klasë trajtohet si pozitive?
2. Çfarë kuptimi praktik ka një false positive këtu?
3. Si llogaritet recall fake nga confusion matrix finale?
4. Pse accuracy vetëm nuk tregon llojin e gabimeve?
5. Çfarë ndryshimi ka F1 fake nga F1 weighted?

---

# Pjesa 7 – Error analysis

Accuracy përmbledh sa raste janë të sakta, por nuk tregon:

- cilat lloje lajmesh gabohen;
- nëse gabimet lidhen me gjatësi, stil ose temë;
- nëse modeli gabon me confidence të lartë;
- nëse një klasë trajtohet shumë më keq se tjetra.

**Error analysis** do të thotë të lexosh dhe gruposh FP/FN, të kontrollosh
probabilitetet dhe sinjalet, dhe të formosh hipoteza të testueshme për sjelljen
e modelit.

## Raste reale nga raporti final

### Correct `likely_real`

- `true_1594`: “Shqipëri, një i vdekur dhe 53 raste të reja me COVID-19”;
- `P(real)=99.99%`, `P(fake)=0.01%`;
- 169 fjalë, source markers `sipas`, `ministria`.

Teksti përputhet me pattern-e institucionale të klasës real. Kjo është
interpretim i sjelljes së modelit, jo provë e vërtetësisë.

### Correct `likely_fake`

- `fake_531`: “EKSKLUZIVE: Albin Kurti President i Kosoves?”;
- `P(fake)=99.99997%`;
- 86 fjalë, marker `ekskluzive`, shumë pak diakritika.

### False positive

- `true_586`: “Menjëherë fshijeni nësë ju vjen ky mesazh në telefon”;
- label real, por `P(fake)=89.64%`;
- artikulli real po përgënjeshtron një pretendim problematik.

Për shkak se artikulli citon përmbajtjen paralajmëruese/clickbait, fjalët e
tekstit mund t’i ngjajnë klasës fake. Modeli nuk kupton automatikisht që autori
po e kundërshton pretendimin.

### False negative

- `fake_1104`: “Nuk hapen xhamitë ... edhe ne Hoxhallarët duam pushim”;
- label fake, por `P(real)=89.41%`;
- 378 fjalë, stil më i gjatë dhe formal.

Ky rast përputhet me bias-in ku fake të gjata mund të shtyhen drejt real.

### High-confidence error

- `fake_1932`: “Flet mjekja shqiptare në Kanada: Si të trajtoni virusin...”;
- label fake, por `P(real)=99.92%`;
- 2,376 fjalë, stil formal dhe vetëm një pikëçuditëse.

Ky është shembulli më i fortë për të treguar se confidence nuk është garanci.

## Pse 99% nuk do të thotë “99% faktikisht e vërtetë”

Probabiliteti është vlerësimi i modelit pas calibration-it, duke supozuar se
input-i sillet si të dhënat ku modeli u mësua dhe u kalibrua. Ai mund të jetë
gabim për shkak të:

- pattern-eve të rreme ose spurioze në corpus;
- domain shift-it;
- tekstit që citon një pretendim për ta përgënjeshtruar;
- mungesës së fakteve, burimeve dhe kontekstit;
- rasteve të rralla jashtë shpërndarjes.

Thuaj “confidence/probabilitet sipas modelit”, jo “shansi që lajmi është
faktikisht i vërtetë”.

### Pyetje kontrolli

1. Çfarë informacioni jep error analysis që nuk e jep accuracy?
2. Pse `true_586` mund të duket fake për një model tekstual?
3. Si lidhet `fake_1932` me length bias?
4. Çfarë është high-confidence error?
5. Pse probability e kalibruar nuk është provë faktike?

---

# Pjesa 8 – Probability calibration

## Classification score kundrejt probability

Linear SVM mëson një hyperplane me margin. `decision_function()` jep një score
me shenjë dhe distancë relative nga kufiri:

- score negativ: drejt real;
- score pozitiv: drejt fake;
- madhësi më e madhe: më larg kufirit.

Ky score nuk është probability: p.sh. `2.0` nuk do të thotë 200% ose ndonjë
probabilitet të përcaktuar. LinearSVC nuk implementon `predict_proba()`.

**Calibration** mëson një mapping nga score te probabiliteti duke përdorur
prediction-e out-of-fold, pa e mësuar mapping-un mbi të njëjtat score in-sample.

## Sigmoid calibration

Sigmoid, e ngjashme me Platt scaling, mëson një funksion parametrik:

```text
P(fake | score) ≈ 1 / (1 + exp(A × score + B))
```

Ka pak parametra, është e lëmuar dhe zakonisht më e qëndrueshme me kampion të
kufizuar.

## Isotonic calibration

Isotonic regression mëson një funksion monoton pjesë-pjesë pa imponuar formë
sigmoid. Është më fleksibël, por mund të overfit-ojë më lehtë dhe kërkon më
shumë raste calibration-i.

## Metrikat e calibration-it

**Brier score** është gabimi mesatar katror:

```text
Brier = mean((P(fake) − y)^2)
```

Sa më i ulët, aq më mirë. Ai vlerëson së bashku calibration-in dhe aftësinë
dalluese.

**Log loss** është:

```text
-mean(y log(p) + (1-y) log(1-p))
```

Ai ndëshkon shumë fort prediction-et e sigurta dhe të gabuara.

**Expected Calibration Error (ECE)**:

1. ndan probabilitetet në bins;
2. për çdo bin krahason confidence mesatare me frekuencën reale;
3. merr diferencën absolute të ponderuar.

ECE më e ulët është më mirë, por varet nga numri dhe kufijtë e bins. Projekti
përdor 10 bins.

**Calibration curve** vizaton probabilitetin mesatar kundrejt frekuencës së
vëzhguar. Modeli perfekt do të ishte afër diagonales: nga rastet me rreth 70%
fake probability, afërsisht 70% duhet të jenë fake.

## Rezultatet reale të Ditës 16

Sigmoid dhe isotonic u krahasuan me nested 5×5 group-safe CV vetëm mbi 3,195
rastet train dhe 1,586 groups.

| Metoda | Brier | Log loss | ECE | F1 weighted | High-conf. errors |
|---|---:|---:|---:|---:|---:|
| Sigmoid | **0.0653** | **0.2175** | 0.0149 | **0.9133** | **49** |
| Isotonic | 0.0659 | 0.2475 | **0.0138** | 0.9114 | 57 |

Isotonic kishte ECE pak më të mirë, por sigmoid kishte Brier dhe log loss më
të mira, F1 pak më të lartë, më pak high-confidence errors dhe rrezik më të
ulët overfitting. Prandaj u zgjodh sigmoid.

Në testin final të brendshëm, pa ndryshuar më modelin:

- Brier: 0.0658;
- log loss: 0.2192;
- ECE: 0.0285;
- high-confidence errors: 15.

Calibration nuk e zgjidh length bias-in ose domain shift-in; ai vetëm e bën
interpretimin e score-ve si probabilitete më të arsyeshëm.

### Pyetje kontrolli

1. Pse score i Linear SVM nuk është probability?
2. Cili është ndryshimi kryesor mes sigmoid dhe isotonic?
3. Çfarë ndëshkon fort log loss?
4. Si interpretohet një calibration curve ideale?
5. Pse u zgjodh sigmoid edhe pse isotonic kishte ECE pak më të ulët?

---

# Pjesa 9 – Zona `uncertain`

Një sistem real/fake vetëm me prag 0.5 do të detyronte një vendim edhe kur
`P(fake)=0.51`. Projekti përdor:

```text
P(fake) < 0.30             → likely_real
0.30 <= P(fake) <= 0.70    → uncertain
P(fake) > 0.70             → likely_fake
```

Vlerat fiks `0.30` dhe `0.70` janë `uncertain`, sepse kufijtë janë inkluzivë.

## Coverage dhe strong accuracy

```text
strong decisions = likely_real + likely_fake
coverage = strong decisions / të gjitha rastet
strong accuracy = raste të sakta mes strong decisions / strong decisions
```

Ka kompromis: një zonë uncertain më e gjerë ul coverage, por zakonisht rrit
saktësinë e vendimeve që mbeten të forta.

Pragjet u zgjodhën nga OOF train predictions, jo nga test-i:

| Zona | Coverage | Strong accuracy | Gabime në uncertain |
|---|---:|---:|---:|
| 30–70 | 88.58% | **94.88%** | **132 / 277** |
| 35–65 | 91.67% | 94.03% | 102 / 277 |
| 40–60 | 95.15% | 92.83% | 59 / 277 |

30–70 kishte strong accuracy më të lartë dhe kapi 47.65% të gabimeve binare në
zonën uncertain.

Në testin final me 792 raste:

- 368 `likely_real`;
- 71 `uncertain`;
- 353 `likely_fake`;
- coverage 91.04%;
- strong accuracy 94.31%;
- 29 nga 70 gabimet binare kaluan në `uncertain`.

`Uncertain` nuk do të thotë se lajmi është “gjysmë real, gjysmë fake”. Do të
thotë se modeli nuk ka siguri të mjaftueshme për vendim të fortë dhe kërkohet
verifikim i jashtëm.

### Pyetje kontrolli

1. Si klasifikohet `P(fake)=0.30`, `0.50` dhe `0.70`?
2. Çfarë mat coverage?
3. Çfarë mat strong accuracy?
4. Pse nuk u zgjodh zona 40–60 me coverage më të lartë?
5. Çfarë duhet t’i thuhet përdoruesit kur rezultati është uncertain?

---

# Pjesa 10 – Data leakage dhe cross-validation

## Train, validation/CV dhe test

- **train:** përdoret për të mësuar parametrat e modelit;
- **validation/CV:** përdoret për të krahasuar konfigurimet dhe hyperparameters;
- **test:** përdoret vetëm pasi vendimi është ngrirë;
- **external:** benchmark pilot për generalization, jo validation set.

Në k-fold cross-validation, train ndahet në `k` fold-e. Çdo herë një fold
përdoret për validation dhe të tjerët për trajnim; rezultatet mesatarizohen.

## StratifiedGroupKFold

`StratifiedGroupKFold` bën dy gjëra njëkohësisht:

1. përpiqet të ruajë përqindjen real/fake në çdo fold;
2. nuk ndan të njëjtin group mes fit dhe validation.

Projekti ndërton groups duke bashkuar çdo rresht që ka:

- të njëjtin `pair_id`; ose
- të njëjtin `model_text` ekzakt.

Kjo bëhet te
`src/models/analyze_model_quality.py::build_leakage_safe_groups()`. Algoritmi
union-find siguron se lidhjet transitive mbeten në të njëjtin group.

## Protokolli real i Ditëve 13–16

1. Dita 13: konfigurimi character dhe përfaqësimi u zgjodhën vetëm me train/CV
   dhe test të brendshëm të ngrirë; external u hap pas ruajtjes së selection-it.
2. Dita 14: classifier-i u zgjodh me 5-fold group-safe CV vetëm mbi train.
3. Dita 15: `C` u zgjodh me të njëjtat fold-e; vendimi u ruajt dhe hash-ua para
   test-it dhe external-it.
4. Dita 16: calibration/pragjet u zgjodhën me nested group-safe OOF train
   predictions; test dhe external u përdorën vetëm pas ngrirjes.

Tuning mbi test do ta kthente test-in në validation dhe metrika do të bëhej
optimiste. Tuning mbi 40 rastet e jashtme do ta konsumonte benchmark-un: ai nuk
do të ishte më provë e pavarur e generalization-it.

### Pyetje kontrolli

1. Pse nuk mjafton `StratifiedKFold` i zakonshëm këtu?
2. Cilat dy lidhje përdoren për të krijuar leakage groups?
3. Çfarë do të ndodhte nëse zgjidhnim modelin nga rezultati external?
4. Pse calibration-i përdori nested CV?
5. Në cilën fazë lejohet të shihet test set-i?

---

# Pjesa 11 – Krahasimi i modeleve

## Përfaqësimet e Ditës 13

| Përfaqësimi me Logistic Regression | F1 weighted internal | Accuracy external diagnostike |
|---|---:|---:|
| Word | 88.38% | 47.5% |
| Character | 88.25% | 65.0% |
| Word + Character | **90.27%** | 52.5% |

Përzgjedhja u bë nga rezultati i brendshëm, prandaj fitoi Word + Character.
Character-only doli më mirë jashtë, por ndryshimi i modelit pas këtij rezultati
do të ishte tuning mbi external. Ky është një shembull i mirë i disiplinës
eksperimentale.

## Classifier-at e Ditës 14

Me Word + Character të ngrirë:

| Kandidati më i mirë i familjes | CV F1 weighted | Std | Test internal F1 |
|---|---:|---:|---:|
| Logistic Regression `C=1.0` | 89.52% | 0.0041 | 90.38% |
| Linear SVM `C=1.0` | **91.10%** | **0.0021** | **91.41%** |
| Complement NB `alpha=0.5` | 88.10% | 0.0106 | 88.62% |

Linear SVM fitoi sepse dha F1 më të lartë, devijim të ulët mes fold-eve dhe
balancë më të mirë mes recall real/fake se Logistic Regression.

Complement Naive Bayes është i përshtatshëm për features sparse jo-negative,
por supozimi i tij i thjeshtë për features dhe rezultatet më të dobëta/variabël
në CV nuk e bënë kandidat final. Ai mori 80% në external diagnostik, por ky
rezultat nuk lejohej të ndryshonte zgjedhjen.

## Çfarë do të thotë `C` te Linear SVM

- `C` më i vogël: regularizim më i fortë, pranon më shumë gabime train për një
  margin më të thjeshtë;
- `C` më i madh: penalizon më fort gabimet train, mund të përshtatet më shumë,
  por rrit rrezikun e variancës/overfitting.

Dita 15 provoi `0.25, 0.5, 1, 2, 4`:

| C | CV F1 weighted | Std F1 | Recall real | Recall fake |
|---:|---:|---:|---:|---:|
| 0.25 | 0.9074 | 0.0030 | 0.9562 | 0.8591 |
| 0.5 | 0.9065 | 0.0040 | 0.9506 | 0.8629 |
| **1.0** | **0.9110** | **0.0021** | 0.9487 | 0.8735 |
| 2.0 | 0.9110 | 0.0059 | 0.9462 | 0.8760 |
| 4.0 | 0.9123 | 0.0060 | 0.9456 | 0.8792 |

`C=4` kishte mesatare vetëm 0.13 pikë përqindjeje më të lartë, por pothuajse
tri herë devijim standard më të madh. `C=1` ishte më i qëndrueshëm, më i
regularizuar dhe në testin e brendshëm doli edhe pak më mirë (91.41% kundrejt
91.16%). Prandaj stabiliteti dhe thjeshtësia fituan ndaj një ndryshimi shumë të
vogël të mesatares.

### Pyetje kontrolli

1. Pse nuk u zgjodh Character-only nga rezultati external 65%?
2. Çfarë avantazhi pati Linear SVM ndaj Logistic Regression në CV?
3. Pse Complement NB ishte baseline i vlefshëm, por jo final?
4. Si ndryshon regularizimi kur rritet `C`?
5. Pse `C=1` ishte vendim më i mbrojtshëm se `C=4`?

---

# Pjesa 12 – Length bias dhe domain shift

## Përkufizimet

**Dataset bias**: dataset-i ka lidhje sistematike që nuk përfaqësojnë domosdo
fenomenin real, p.sh. klasa fake është mesatarisht shumë më e shkurtër.

**Length bias**: modeli përdor drejtpërdrejt ose indirekt gjatësinë si sinjal;
tekste të shkurtra shtyhen fake dhe të gjata real.

**Domain shift**: shpërndarja e input-eve në përdorim ndryshon nga ajo e
trajnimit, p.sh. periudhë, burim, temë, stil dhe gjatësi të ndryshme.

## Çfarë zbuloi Dita 12 për baseline-in Word

Në testin e brendshëm:

| Gjatësia | Raste | Real/Fake | Accuracy | FP | FN | Mean P(fake) |
|---|---:|---:|---:|---:|---:|---:|
| ≤60 | 9 | 6/3 | 77.78% | 2 | 0 | 0.6475 |
| 61–120 | 341 | 75/266 | 92.67% | 18 | 7 | 0.7965 |
| 121–250 | 269 | 174/95 | 85.87% | 17 | 21 | 0.3769 |
| >250 | 173 | 144/29 | 84.39% | 2 | 25 | 0.0940 |

Klasat dhe gjatësia janë të ndërthurura: shumica e teksteve 61–120 janë fake,
ndërsa shumica mbi 250 janë real. Modeli mëson këtë pattern.

## Spearman correlation

Spearman `rho` mat lidhjen monotone mbi renditjen, jo domosdo lidhje lineare.

Për baseline-in:

- të gjitha rastet: `rho=-0.7132`;
- vetëm real: `rho=-0.5883`;
- vetëm fake: `rho=-0.4971`.

Shenja negative do të thotë se me rritjen e `word_count`, `P(fake)` priret të
ulet. Lidhja ekziston edhe brenda secilës klasë, kështu nuk shpjegohet vetëm
nga përzierja e klasave. Correlation nuk provon shkakësi, por eksperimentet e
shkurtimit japin provë diagnostike shtesë.

## Eksperimenti i stabilitetit

Për katër artikuj real dhe katër fake u krahasuan teksti i plotë, titull + 120
fjalë, rreth 46 fjalë dhe vetëm titulli.

| Label | Variant | Mean P(fake) |
|---|---|---:|
| Real | i plotë | 0.2115 |
| Real | rreth 46 fjalë | **0.7879** |
| Real | vetëm titulli | **0.9393** |
| Fake | i plotë | 0.4691 |
| Fake | rreth 46 fjalë | 0.8021 |
| Fake | vetëm titulli | 0.8979 |

Në 7 nga 8 rastet, versioni 46-fjalësh mori P(fake) më të lartë. Sidomos
artikujt real ndryshuan rëndë. Shkurtimi ndryshon edhe fjalorin, jo vetëm
numrin e fjalëve; prandaj është eksperiment diagnostik, jo provë e pastër
shkakësore.

## Domain shift përtej gjatësisë

| Faktor | Internal | External |
|---|---|---|
| Mean word count | 210.46 | 45.58 |
| Periudha | prill–korrik 2020 | korrik 2024–korrik 2026 |
| Formati | artikuj të plotë | përmbledhje manuale |
| Tema | shpërndarje e corpus-it | 5 tema të balancuara me dorë |
| Burime | corpus-i origjinal | real institucionale, fake pretendime sociale |

Ndryshonin edhe diacritic ratio, source markers, sensational markers dhe stili.
Këto janë kandidatë për domain shift, jo prova individuale shkakësore.

## A e zgjidhi modeli final?

Modeli final e uli Spearman nga `-0.7132` te rreth `-0.6122`, dhe brenda
klasave te `-0.3045` real / `-0.4351` fake. Por bias-i mbeti:

- real 30–60 fjalë: 5/6 të sakta;
- fake mbi 250 fjalë: vetëm 13/29 të sakta, recall 44.83%.

Pra Word + Character + SVM e zbuti problemin, por nuk e eliminoi.

### Pyetje kontrolli

1. Cili është ndryshimi mes dataset bias dhe domain shift?
2. Si interpretohet `rho=-0.7132`?
3. Çfarë ndodhi me artikujt real kur u shkurtuan në 46 fjalë?
4. Pse eksperimenti i shkurtimit nuk provon vetëm efektin e gjatësisë?
5. Cila gjetje tregon se modeli final ende ka length bias?

---

# Pjesa 13 – External validation

Dataset-i `data/external/external_news.csv` u krijua për të testuar modelin
jashtë corpus-it ku u trajnua.

## Përbërja dhe kontrolli

- 40 raste: 20 real dhe 20 fake;
- 5 tema: politikë, shëndetësi, ekonomi, sociale, teknologji;
- 8 raste për temë, 4 real dhe 4 fake;
- përmbajtje manuale 32–43 fjalë; me titullin rreth 38–51 fjalë;
- asnjë dublikatë ose overlap ekzakt me train;
- ngjashmëria maksimale TF-IDF me train ishte vetëm 0.3399;
- çdo rast kishte burim, URL, datë dhe evidence të etiketimit.

Rastet fake nuk u etiketuan nga stili sensacional. U përdorën raste me
vlerësim të qartë “Rrenë” dhe evidence nga Krypometër. Rastet real u lidhën me
burime institucionale/të besueshme. Evidence u ruajt veç dhe nuk iu dha modelit.

Dataset-i u **fiksua** para evaluation-it. Nuk u ndryshua për të rritur metrikat
dhe nuk u përdor për tuning, zgjedhje modelesh, calibration ose pragje.

## Baseline kundrejt modelit final

| Modeli | Accuracy | Recall real | Recall fake | Confusion matrix |
|---|---:|---:|---:|---|
| Word + Logistic baseline | 47.5% | 5% | 90% | `[[1,19],[2,18]]` |
| Word+Char + SVM final | **60.0%** | **50%** | 70% | `[[10,10],[6,14]]` |

Baseline-i e quajti fake 19 nga 20 lajmet real. Gjatësia e shkurtër i ngjante
anës fake të corpus-it. Modeli final u bë më i balancuar dhe rriti accuracy me
12.5 pikë përqindjeje, por 60% mbetet generalization vetëm pjesërisht i mirë.

Për modelin final:

- 8 `likely_real`, 19 `uncertain`, 13 `likely_fake`;
- strong coverage 52.5%;
- strong accuracy 71.43%;
- 10 nga 16 gabimet binare kaluan në `uncertain`;
- Brier 0.2377, log loss 0.6710.

## Pse ky nuk është dështim i projektit

Një rezultat i dobët i raportuar me protokoll të pastër është gjetje
shkencore. Ai tregon se:

- metrika brenda corpus-it nuk garanton generalization;
- stili dhe gjatësia e corpus-it mund të mësohen si shortcut;
- external validation është e domosdoshme;
- sistemi duhet të përdoret si sinjal ndihmës, jo si arbitër i së vërtetës;
- duhen të dhëna trajnimi më të larmishme dhe benchmark më i madh.

Duhet të thuash “modeli përgjithëson pjesërisht dhe benchmark-u është pilot”, jo
të fshehësh 60% ose ta krahasosh si të ishte i njëjti domain me testin e
brendshëm.

### Pyetje kontrolli

1. Pse dataset-i external ishte i balancuar 20/20?
2. Pse etiketa fake kërkonte evidence?
3. Pse benchmark-u u ngrirë para prediction-it?
4. Pse baseline-i klasifikoi 19 nga 20 real si fake?
5. Çfarë përfundimi shkencor nxjerrim nga 91% internal dhe 60% external?

---

# Pjesa 14 – Modeli final

Duhet të jesh në gjendje ta vizatosh këtë pa parë shënime:

```text
title + content
      ↓
Unicode NFC + normalizim hapësirash
      ↓
Word TF-IDF (1,2) + Character char_wb TF-IDF (3,5)
      ↓
Linear SVM, C=1.0
      ↓
sigmoid calibration
      ↓
P(real) / P(fake)
      ↓
pragjet 0.30 / 0.70
      ↓
likely_real / uncertain / likely_fake
```

## Çfarë ndodh hap pas hapi

1. `predict_final_news(title, content)` merr input-in.
2. `prepare_final_model_text()` thërret `combine_title_content()`; ky aplikon
   NFC, bashkon hapësirat dhe lidh titullin me përmbajtjen.
3. Artefakti sklearn ka një `FeatureUnion`: dega Word prodhon 30,000 features,
   dega Character 50,000; bashkohen në një vektor sparse 80,000-dimensional.
4. `LinearSVC(C=1.0, class_weight="balanced")` llogarit decision score.
5. `CalibratedClassifierCV(method="sigmoid")` e kthen score-n në
   `predict_proba()` për klasat `[0,1]`.
6. `predict_final_news()` krijon hartën sipas `model.classes_`, kontrollon që
   probabilitetet janë finite, në `[0,1]` dhe shuma është 1.
7. Prediction-i binar përdor 0.5; `classify_probability()` përdor pragjet
   0.30/0.70 për vendimin e UI-së.
8. `build_linguistic_explanation()` nxjerr sinjalet veçmas. Ato nuk hyjnë në
   model dhe nuk ndryshojnë probabilitetet.
9. Output-i përmban model ID/version, probabilitetet, vendimin, thresholds,
   shpjegimin dhe warning-un për fact-checking.

## Artefakti i ngrirë

```text
models/final_word_char_linear_svm_calibrated_v1.joblib
models/final_model_v1_manifest.json
```

- model ID: `albanian_fake_news_word_char_svm_sigmoid_v1`;
- version: `1.0.0`;
- madhësi: 1.351 MB;
- SHA-256:
  `52ccbc976b10b4a5749e9814d736661ec66c95e1218a19692bdb0ea53dab11d5`.

Manifesti dokumenton konfigurimin, preprocessing-un, pragjet, versionet,
metrikat dhe hash-in. Hash-i provon se artefakti final është byte-identik me
kandidatin e ngrirë të Ditës 16.

### Pyetje kontrolli

1. Thuaje pipeline-in final pa parë tekstin.
2. Ku ndodh preprocessing-u në prediction runtime?
3. Çfarë objekti sklearn ruhet në artefaktin final?
4. Pse kontrollohet `model.classes_` në vend që të supozohet rendi?
5. A ndryshojnë linguistic features probabilitetin final?

---

# Pjesa 15 – Streamlit

## Si niset

Nga rrënja e repository-t:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app\streamlit_app.py
```

## Rrjedha reale

```text
st.text_input + st.text_area
          ↓
st.form_submit_button
          ↓
validate_news_input()
          ↓
get_cached_model() me @st.cache_resource
          ↓
predict_with_final_model()
          ↓
predict_final_news()
          ↓
render_result() + render_decision()
```

## Pjesët që duhen kuptuar

- `st.set_page_config`: konfigurimi bazë i faqes;
- `st.form`: mban input-et dhe butonin në një submission;
- `st.text_input`: titulli;
- `st.text_area`: përmbajtja;
- `st.form_submit_button`: nis analizën;
- `st.session_state`: ruan input-in dhe rezultatin gjatë rerun-eve;
- `@st.cache_resource`: ngarkon modelin vetëm një herë për procesin Streamlit;
- `inspect_model_assets`: kontrollon modelin/manifestin dhe jep gabim të qartë;
- `validate_news_input`: ndalon input bosh, punctuation-only dhe mbi 100,000
  karaktere; paralajmëron për vetëm titull, nën 20 fjalë ose mbi 20,000
  karaktere;
- `render_decision`: shpjegon vendimin sipas pragjeve;
- `render_result`: probabilitetet, linguistic signals dhe warning-u.

Modeli final importohet nga `src.models.predict_final`. App-i nuk përdor më
`src/models/predict.py` ose modelin e vjetër Logistic Regression.

`app/streamlit_app.py` mban vetëm rrjedhën dhe cache-in. Funksionet e validimit
dhe paraqitjes janë te `app/streamlit_ui.py`, ndërsa CSS-i te `app/style.css`.

`st.cache_resource` është i përshtatshëm sepse modeli është resource i madh dhe
i pandryshueshëm gjatë sesionit; pa cache do të deserializohej në çdo rerun.

Warning-u është gjithmonë i dukshëm para prediction-it dhe përsëritet te
rezultati. Formulimi “Real/Fake sipas modelit” shmang pretendimin e verifikimit
faktik.

### Pyetje kontrolli

1. Pse Streamlit e ekzekuton script-in përsëri pas ndërveprimeve?
2. Pse përdoret `st.cache_resource` për modelin?
3. Cilat input-e janë bllokuese dhe cilat japin vetëm warning?
4. Cili funksion lidh UI-në me pipeline-in final?
5. Ku sigurohet që app-i nuk bën crash kur mungon artefakti?

---

# Pjesa 16 – Strategjia e testimit

Projekti final ka 115 teste që kalojnë. Nuk duhet të mësosh çdo test, por duhet
të dallosh tri nivele.

## Unit tests

Testojnë një funksion të vogël në izolim:

- `normalize_spaces()` ruan `ë/ç` dhe prodhon NFC;
- `combine_title_content()` trajton title/content bosh;
- `classify_probability()` respekton kufijtë 0.30/0.70;
- `extract_linguistic_features()` numëron marker-at dhe pikësimin;
- `validate_news_input()` dallon input bosh, të shkurtër dhe punctuation-only.

## Regression tests

Sigurojnë që sjellja e ngrirë nuk ndryshon pa dashje:

- hash-i i artefaktit përputhet me manifestin;
- modeli ka Word + Character, Linear SVM `C=1`, sigmoid dhe klasat `[0,1]`;
- të njëjtat gjashtë raste finale japin të njëjtat vendime/probabilitete;
- reload-i i modelit jep rezultat identik;
- NFC dhe NFD japin prediction identik;
- preprocessing-u i evaluation dhe prediction është i njëjtë;
- linguistic explanation nuk ndryshon probability.

## Integration/system tests

Testojnë disa pjesë së bashku:

- artefakti ngarkohet dhe `predict_final_news()` prodhon output-in e plotë;
- probabilitetet janë në `[0,1]` dhe shuma është 1;
- Streamlit përdor vetëm modelin final;
- model/manifest i munguar jep mesazh dhe jo crash;
- input-et title-only, content-only, Unicode, shumë të gjata dhe joideale
  kalojnë në rrjedhën e pritur;
- notebook-u final ekzekutohet pa ritrajnuar modelin.

## Pse pytest

Pytest jep discovery automatik, assertions të lexueshme, fixtures/monkeypatch
për izolim dhe një komandë të vetme për regression suite. Testet nuk provojnë
që modeli është “i vërtetë”; ato provojnë që implementimi sillet sipas
kontratës së dokumentuar.

Komanda finale:

```powershell
python -m pytest -q
```

### Pyetje kontrolli

1. Cili është ndryshimi mes unit, regression dhe integration test?
2. Pse hash-i i modelit është regression check?
3. Cilat tri kushte kontrollohen për probabilitetet?
4. Pse testohet NFC kundrejt NFD?
5. A provojnë 115 testet se prediction-i është faktikisht i saktë?

---

# Pjesa 17 – Kufizimet dhe formulimi shkencor

## Kufizimet që duhet t’i thuash hapur

1. **Nuk është fact-checker.** Nuk kërkon burime, dokumente apo prova; analizon
   vetëm tekstin.
2. **Dataset bias.** Corpus-i ka lidhje mes label-it dhe gjatësisë/stilit që
   mund të jenë shortcuts.
3. **Length bias.** Tekstet e shkurtra priren fake dhe fake të gjata real.
4. **Domain shift.** Periudha, burimi, temat dhe formati i jashtëm ndryshojnë.
5. **External dataset i vogël.** 40 raste japin benchmark pilot, jo përfundim
   universal.
6. **Probability nuk është garanci.** Ka 15 high-confidence errors në testin e
   brendshëm.
7. **Metadata nuk përdoret.** Modeli nuk sheh URL, reputacion burimi, datë apo
   evidence.
8. **Explanation jo shkakësor.** Marker-at janë vëzhgime, jo arsye e provuar e
   prediction-it dhe as prova e label-it.
9. **Corpus/periudhë e kufizuar.** Train-i vjen kryesisht nga një interval i
   shkurtër i vitit 2020.
10. **Model klasik.** Nuk modelon semantikë të thellë, ironi, citim ose kontekst
    si një model transformer i avancuar.

## Si t’i paraqesësh pa e dëmtuar projektin

Mos thuaj: “Modeli është vetëm 60% i mirë, prandaj nuk funksionon.”

Thuaj:

> “Brenda corpus-it, me protokoll group-safe, modeli arriti 91.16%. Në
> benchmark-un e jashtëm pilot arriti 60%, çka tregon domain shift të fortë.
> Kjo kufizon përdorimin praktik, por është një gjetje e rëndësishme: validimi
> vetëm brenda një corpus-i do të jepte një ide tepër optimiste. Prandaj app-i
> përdor probabilitete, zonë uncertain dhe warning të qartë.”

Mos premto se BERT “do ta zgjidhë”. Thuaj se XLM-RoBERTa mund të kapë më mirë
semantikën dhe kontekstin, por kërkon benchmark më të mirë, computation,
calibration dhe të njëjtat kontrolle leakage/domain shift.

Kufizimet nuk janë dobësi e prezantimit kur janë matur, dokumentuar dhe lidhur
me hapa të ardhshëm. Kjo tregon pjekuri shkencore.

### Pyetje kontrolli

1. Pse modeli nuk mund të quhet fact-checker?
2. Cilat tri forma bias/domain shift janë më të rëndësishme këtu?
3. Pse linguistic explanation nuk është causal explanation?
4. Si do ta shpjegoje 60% external pa e fshehur dhe pa e zhvlerësuar projektin?
5. Çfarë do të kërkohej para përdorimit real në prodhim?

---

# Programi intensiv 3-ditor

Planifiko rreth 7–8 orë pune efektive në ditë. Bëj pushim 10–15 minuta pas çdo
blloku 60–90 minutësh. Çdo përgjigje dhe provë prezantimi bëje me zë.

## Dita 1 – Nga corpus-i te baseline-i

### 09:00–09:30 – Harta e projektit

- Lexo: hyrjen, hartën e repository-t dhe “Numrat që duhen ditur”.
- Hape: `README.md`, `models/final_model_v1_manifest.json`.
- Rezultat: shpjego me 60 sekonda problemin, output-in dhe kufizimin
  “jo fact-checking”.

### 09:30–10:45 – Dataset, IDs dhe leakage

- Studio Pjesën 1.
- Lexo në kod: `src/data/load_dataset.py`, sidomos `LABELS`,
  `split_title_content()`, `article_to_row()`, `load_dataset()`.
- Lexo raportin: `reports/day1_dataset_audit.md`.
- Mos u ndal te logging-u ose renditja e skedarëve.
- Ekzekuto:

```powershell
python -m pytest tests\test_load_dataset.py -q
```

- Rezultat: vizato train/test dhe shpjego `article_id`, `pair_id`, 7 dublikatat.

### 11:00–11:45 – Preprocessing dhe Unicode

- Studio Pjesën 2.
- Lexo të gjithë `src/preprocessing/clean_text.py`.
- Ekzekuto:

```powershell
python -c "import unicodedata; from src.preprocessing.clean_text import combine_title_content; t=unicodedata.normalize('NFD','Lajm për Shqipërinë'); print(combine_title_content(t,'  Përmbajtje   me  hapësira. '))"
python -m pytest tests\test_clean_text.py -q
```

- Rezultat: shpjego NFC me shembullin `ë = e + combining mark`.

### 12:00–14:00 – TF-IDF me laps dhe kod

- Studio Pjesën 3 ngadalë.
- Llogarit me dorë Bag of Words dhe idf për shembullin me tre dokumente.
- Lexo vetëm `build_baseline_model()` në `src/models/train_model.py` dhe
  konfigurimin në manifest.
- Hape modelin me:

```powershell
python -c "import joblib; m=joblib.load('models/final_word_char_linear_svm_calibrated_v1.joblib'); print(m.estimator.named_steps['features'])"
```

- Rezultat: shpjego sparse vector, word bigrams, char 3–5, `char_wb`, `min_df`.

### 15:00–16:00 – Logistic Regression baseline

- Studio Pjesën 4.
- Lexo: `src/models/train_model.py::build_baseline_model()` dhe
  `evaluate_model()`.
- Lexo: pjesën e rezultateve në `reports/day2_baseline_model.md`.
- Rezultat: vizato `w·x+b → sigmoid → 0.5` dhe thuaj pse ishte baseline.

### 16:15–17:30 – Linguistic features

- Studio Pjesën 5.
- Lexo: konstantet e marker-ave dhe `extract_linguistic_features()` te
  `src/features/linguistic_features.py`.
- Shiko: `reports/figures/day4_top_effect_sizes.png` dhe
  `reports/day5_model_comparison.csv`.
- Ekzekuto:

```powershell
python -m pytest tests\test_linguistic_features.py tests\test_feature_analysis.py -q
```

- Rezultat: shpjego Cohen's d, p-value dhe pse features mbetën explanation-only.

### 17:30–18:00 – Përsëritje pa shënime

Duhet të shpjegosh:

1. corpus-in dhe split-in;
2. preprocessing-un;
3. TF-IDF nga zero;
4. baseline-in;
5. përfundimin e linguistic features.

### Mini-quiz i Ditës 1

1. Sa artikuj ka corpus-i dhe si ndahen klasat?
2. Pse `pair_id` përdoret për grouping?
3. Shpjego TF, DF dhe IDF me nga një fjali.
4. Çfarë krijojnë `(1,2)` te Word dhe `(3,5)` te Character?
5. Pse vectorizer-at kanë `lowercase=False`?
6. Çfarë tregon shenja negative e Cohen's d për `word_count`?
7. Pse hybrid-i nuk u bë modeli final?
8. Shpjego në 30 sekonda pse modeli nuk është fact-checker.

Përgjigjet janë në Pjesët 1–5. Kalo në Ditën 2 vetëm nëse i përgjigjesh të
paktën 6/8 pa shënime.

## Dita 2 – Si u zgjodh dhe u vlerësua modeli

### 09:00–10:15 – Metrics dhe error analysis

- Studio Pjesët 6–7.
- Rillogarit precision/recall nga `[[369,30],[40,353]]`.
- Lexo: `reports/day17_final_demo_cases.csv`.
- Shiko gjashtë rastet dhe shpjego dy prej tyre pa tekstin e raportit.
- Rezultat: dallon menjëherë TN/FP/FN/TP dhe shpjegon high-confidence error.

### 10:30–11:30 – Leakage dhe group-safe CV

- Studio Pjesën 10.
- Lexo vetëm:
  - `src/models/analyze_model_quality.py::build_leakage_safe_groups()`;
  - `src/models/compare_classifiers.py::build_group_safe_folds()`.
- Mos studio union-find rresht për rresht; kupto që bashkon pair/duplicate.
- Rezultat: shpjego pse test/external nuk përdoren për tuning.

### 11:45–13:15 – Krahasimi i përfaqësimeve dhe classifier-ëve

- Studio Pjesën 11.
- Lexo raportet:
  - `reports/day13_tfidf_representation_comparison.md`, seksionet e rezultateve;
  - `reports/day14_classifier_comparison.md`, CV dhe përfundimi;
  - `reports/day15_svm_tuning.md`, CV dhe zgjedhja e `C`.
- Shiko figurat:
  - `day13_internal_model_comparison.png`;
  - `day14_cv_classifier_comparison.png`;
  - `day15_cv_c_tuning.png`.
- Rezultat: mbro zgjedhjen Word+Character, SVM dhe `C=1`.

### 14:15–15:45 – Calibration dhe uncertain

- Studio Pjesët 8–9.
- Lexo vetëm këto funksione te `src/models/calibrate_linear_svm.py`:
  `expected_calibration_error()`, `probability_metrics()`,
  `nested_oof_calibration()`, `select_calibration_method()`,
  `select_thresholds()`.
- Shiko:
  - `reports/figures/day16_oof_calibration_comparison.png`;
  - `reports/figures/day16_threshold_comparison.png`.
- Ekzekuto:

```powershell
python -m pytest tests\test_svm_calibration.py -q
```

- Rezultat: shpjego score vs probability, sigmoid vs isotonic dhe 30/70.

### 16:00–17:30 – Length bias, domain shift dhe external

- Studio Pjesët 12–13.
- Lexo përfundimet e:
  - `reports/day12_length_domain_shift.md`;
  - `reports/day17_final_model.md`.
- Shiko:
  - `reports/figures/day12_probability_vs_length.png`;
  - `reports/figures/day17_final_length_performance.png`.
- Rezultat: shpjego me shifra 91% vs 60% dhe pse gjatësia nuk është i vetmi
  domain shift.

### 17:30–18:00 – Përsëritje pa shënime

Duhet të jesh në gjendje të mbrosh të gjithë zinxhirin e vendimeve:

```text
Word baseline → Word+Character → SVM → C=1 → sigmoid → 0.30/0.70
```

### Mini-quiz i Ditës 2

1. Llogarit recall fake nga confusion matrix finale.
2. Pse 99.9% confidence mund të jetë gabim?
3. Si krijohen leakage groups?
4. Pse SVM fitoi ndaj Logistic Regression?
5. Pse nuk u zgjodh `C=4`?
6. Çfarë matin Brier, log loss dhe ECE?
7. Çfarë kompromisi ka coverage me strong accuracy?
8. Jep tri prova të length/domain shift-it.
9. Pse 65% external i Character-only nuk ndryshoi përzgjedhjen?
10. Çfarë vlere shkencore ka rezultati 60% external?

Syno të paktën 8/10 pa shënime.

## Dita 3 – Runtime-i, demonstrimi dhe mbrojtja

### 09:00–10:30 – Pipeline-i final në kod

- Studio Pjesën 14.
- Lexo të gjithë:
  - `src/models/prediction_utils.py`;
  - `src/models/predict_final.py`;
  - `models/final_model_v1_manifest.json`.
- Ekzekuto një prediction:

```powershell
python -c "from src.models.predict_final import predict_final_news; r=predict_final_news('Titull zyrtar','Sipas ministrisë, raporti u publikua sot për qytetarët.'); print(r)"
```

- Rezultat: ndiq çdo field të dictionary-t nga input-i te output-i.

### 10:45–11:45 – Streamlit

- Studio Pjesën 15.
- Lexo vetëm: `validate_news_input()`, `get_cached_model()`,
  `inspect_model_assets()`, `predict_with_final_model()`, `render_decision()`,
  `render_result()` dhe `main()`.
- Injoro CSS-in dhe tekstin e shembujve.
- Nise app-in:

```powershell
python -m streamlit run app\streamlit_app.py
```

- Provo një rast real, fake, uncertain, vetëm titull dhe punctuation-only.

### 12:00–12:45 – Testing

- Studio Pjesën 16.
- Lexo 5–6 teste përfaqësuese në `tests/test_final_model.py` dhe
  `tests/test_streamlit_app.py`.
- Ekzekuto:

```powershell
python -m pytest -q
```

- Rezultat: jep nga një shembull unit, regression dhe integration test.

### 13:45–14:30 – Kufizimet

- Studio Pjesën 17.
- Thuaji të dhjetë kufizimet me gjuhë të qetë dhe shkencore.
- Ushtro përgjigjen “A është ky fact-checker?” dhe “Pse vetëm 60% jashtë?”.

### 14:30–15:15 – Demo e kontrolluar

- Lexo `reports/day19_demo_guide.md`.
- Përdor tre rastet kryesore nga `reports/day19_demo_cases.csv`.
- Mbaj gati false positive/negative vetëm nëse komisioni pyet për kufizime.
- Bëj një provë pa internet; modeli punon lokalisht.

### 15:30–17:00 – Prezantimi 10–15 minuta

- Ndiq skenarin në seksionin “Prezantimi verbal”.
- Regjistro veten dy herë:
  - prova 1: pa u ndalur, mat kohën;
  - prova 2: hiq përsëritjet dhe syno 12–13 minuta.
- Mos lexo slide-t; përdori vetëm si orientim.

### 17:00–18:00 – Pyetjet dhe provimi simulues

- Përgjigju bankës së pyetjeve me maksimum 30–45 sekonda secila.
- Bëj provimin simulues pa shënime.
- Shëno vetëm tri pikat ku ngece dhe përsëriti para gjumit.

### Mini-quiz i Ditës 3

1. Thuaje pipeline-in final në më pak se 40 sekonda.
2. Cilat checks bën `predict_final_news()` mbi probabilitetet?
3. Si garanton app-i që modeli nuk ngarkohet në çdo klikim?
4. Cilat linguistic features shfaq UI dhe a ndikojnë prediction-in?
5. Çfarë bën app-i me input bosh, shumë të shkurtër dhe punctuation-only?
6. Jep nga një unit, regression dhe integration test.
7. Cilat janë tri metrikat zyrtare më të rëndësishme?
8. Jep tri kufizime dhe nga një përmirësim të mundshëm.

Syno 8/8. Nëse ngec, përsërit vetëm pjesën përkatëse, jo të gjithë materialin.

---

# Walkthrough i vogël i kodit

## `src/data/load_dataset.py`

**Roli:** kthen skedarët raw `true/fake` në një DataFrame të strukturuar.

**Kupto:**

- `LABELS` dhe kolonat e pritshme;
- `read_text_file()` dhe fallback-u i encoding;
- `extract_pair_id()`;
- `split_title_content()`;
- `article_to_row()`;
- loop-i kryesor i `load_dataset()`.

**Mund të injorosh:** logging-un, type conversion rutinë dhe helper-in e
sortimit pas kuptimit të idesë.

## `src/preprocessing/clean_text.py`

**Roli:** kontrata e vetme e tekstit që sheh modeli.

**Kupto:** të tre funksionet: `normalize_spaces()`,
`combine_title_content()`, `prepare_text_dataframe()`.

**Mund të injorosh:** asgjë e rëndësishme; skedari është vetëm 45 rreshta.

## `src/features/linguistic_features.py`

**Roli:** prodhon 29 features numerike dhe listat e marker-ave.

**Kupto:** listat e frazave, tokenizimin bazë, `safe_ratio()`,
`find_phrases()`, `extract_linguistic_features()` dhe grupet e output-it.

**Mund të injorosh:** detajet e çdo regex-i dhe përsëritjen e count-eve, për sa
kohë kupton matching-un case-insensitive dhe mbrojtjen nga pjesët e fjalëve.

## `src/models/prediction_utils.py`

**Roli:** ndan logjikën e vendimit dhe explanation-it nga UI.

**Kupto:** të gjithë skedarin, sidomos `classify_probability()` dhe kufijtë
inkluzivë të `uncertain`.

**Mund të injorosh:** vetëm helper-in trivial `_marker_list()` pasi e kupton.

## `src/models/predict_final.py`

**Roli:** API e vetme stabile e modelit final.

**Kupto:** konstantet e versionit/path-eve, `prepare_final_model_text()`,
`load_final_model()` dhe çdo kontroll në `predict_final_news()`.

**Mund të injorosh:** asgjë thelbësore; është rreth 110 rreshta dhe duhet të
jesh në gjendje ta shpjegosh nga fillimi në fund.

## `app/streamlit_app.py`

**Roli:** entrypoint-i që lidh formularin me modelin final.

**Kupto:**

- `get_cached_model()`;
- `predict_with_final_model()`;
- `main()`.

**Mund të injorosh:** detajet e widget-eve pasi kupton rendin validation →
model → result.

## `app/streamlit_ui.py`

**Roli:** validimi i input-it dhe paraqitja e rezultatit pa logjikë modeli.

**Kupto:** `validate_news_input()`, `inspect_model_assets()`,
`render_decision()` dhe `render_result()`.

**Mund të injorosh:** listën e shembujve, CSS-in dhe detajet e kolonave.

## `tests/test_final_model.py`

**Roli:** mbron artefaktin dhe kontratën e prediction-it.

**Studio vetëm testet për:** config/hash, probability sum, threshold,
determinism pas reload, NFC/NFD, preprocessing identik dhe explanation-only.

**Mund të injorosh:** fixtures dhe helper-at e testit pasi kupton çfarë
assert-ojnë.

## `tests/test_streamlit_app.py`

**Roli:** mbron validimin dhe integrimin e UI-së me modelin final.

**Studio vetëm testet për:** input bosh/shkurtër, assets që mungojnë, rastet e
ngrira, Unicode dhe tekste të gjata.

## `notebooks/02_final_walkthrough.ipynb`

**Roli:** demonstron rrjedhën end-to-end duke përdorur funksionet dhe output-et
e ruajtura, pa ritrajnuar modelin.

**Kupto:** rendin e qelizave dhe çfarë provon secila.

**Mund të injorosh:** formatimin e tabelave dhe kodin e vizualizimit.

## Refactor-i i runtime-it

Runtime-i është ndarë pa ndryshuar versionin `v1.0.0`: normalizimi i hapësirave
ka një implementim të vetëm, `predict_final_news()` delegon hapa të vegjël dhe
UI-ja është ndarë nga entrypoint-i. Skriptet e gjata të eksperimenteve mbeten
historike dhe katalogohen te `experiments/README.md`.

Ky refactor u kontrollua me hash-in e modelit, gjashtë rastet e ngrira dhe gjithë
test suite-in. Për përmirësime të ardhshme mund të centralizohen helper-at e
evaluation-it, por vetëm nëse ruhet byte-for-byte output-i historik.

---

# Prezantimi verbal 10–15 minuta

Syno rreth 12–13 minuta plus pyetjet. Përdor 10–12 slide të pastra. Shifrat e
detajuara mbaji në slide rezervë.

## 1. Problemi – 45 sekonda

**Thuaj:**

“Qëllimi ishte analiza e karakteristikave gjuhësore të lajmeve real/fake në
shqip dhe ndërtimi i një aplikacioni që jep probabilitet sipas modelit. Sistemi
nuk verifikon faktet; ai klasifikon pattern-et e tekstit dhe e shpreh
pasigurinë me një zonë uncertain.”

**Trego:** titullin e projektit dhe një diagram input → output.

**Mos humb kohë:** me histori të përgjithshme të fake news ose definicione të
gjata sociale.

## 2. Dataset-i – 60 sekonda

**Thuaj:**

“Corpus-i kishte 3,994 artikuj: 1,998 real dhe 1,996 fake. Rreshti i parë u
interpretua si titull. `pair_id` lidhi çiftet dhe u përdor për ndarje
group-safe. Pas auditimit përjashtova 7 kopje ekzakte train–test nga evaluation,
duke lënë 792 raste të pastra.”

**Trego:** tabelën e numrave nga `reports/day1_dataset_audit.md` dhe një skemë
train 3,195 / test 799 → test i pastër 792.

**Mos humb kohë:** me kolonat rutinë të pandas ose çdo dosje raw.

## 3. Analiza gjuhësore – 60 sekonda

**Thuaj:**

“Analizova gjatësi, pikësim, uppercase, diakritika dhe marker-a sensacionalë,
burimi e pasigurie. Dallimet më të forta ishin gjatësia dhe diakritikat; fake
ishin mesatarisht 132 fjalë kundrejt 291 për real. Linguistic-only arriti 82.7%
dhe hybrid-i nuk e kaloi TF-IDF, prandaj këto features mbetën për analizë dhe
explanation, jo për prediction final.”

**Trego:** `reports/figures/day4_top_effect_sizes.png`.

**Mos humb kohë:** duke listuar të 29 features një nga një.

## 4. Pipeline-i ML – 75 sekonda

**Thuaj:**

“Preprocessing-u ruan kapitalizimin, pikësimin dhe `ë/ç`, aplikon NFC dhe
bashkon titullin me përmbajtjen. Word TF-IDF përdor unigram/bigram, ndërsa
Character `char_wb` përdor 3–5 grams. Vektorët sparse bashkohen dhe futen në një
classifier linear.”

**Trego:** pipeline-in e plotë me shigjeta.

**Mos humb kohë:** me derivim matematikor të plotë të TF-IDF; jep vetëm një
shembull të vogël nëse pyetesh.

## 5. Modelet e provuara – 75 sekonda

**Thuaj:**

“Nisa me Word TF-IDF + Logistic Regression. Pastaj krahasova Word, Character
dhe kombinimin; kombinimi fitoi me 90.27% F1 të brendshëm. Me këtë
përfaqësim të ngrirë, Linear SVM fitoi ndaj Logistic Regression dhe Complement
NB në group-safe CV. `C=1` u zgjodh për stabilitet: 0.9110 ± 0.0021 F1.”

**Trego:** `day13_internal_model_comparison.png` dhe
`day14_cv_classifier_comparison.png`, ose një tabelë të përbashkët.

**Mos humb kohë:** me çdo konfigurim të refuzuar.

## 6. Modeli final – 45 sekonda

**Thuaj:**

“Modeli final është Word + Character TF-IDF, Linear SVM `C=1.0`, version
1.0.0. Artefakti dhe manifesti janë të versionuar; hash-i garanton që app-i
përdor pikërisht kandidatin e ngrirë.”

**Trego:** një card/tabelë me katër komponentët dhe versionin.

**Mos humb kohë:** me path-et e çdo modeli eksperimental.

## 7. Calibration dhe uncertain – 75 sekonda

**Thuaj:**

“Linear SVM jep decision scores, jo probabilitete. Krahasova sigmoid dhe
isotonic me nested group-safe CV. Zgjodha sigmoid për Brier/log loss më të mirë
dhe rrezik më të ulët overfitting. Me pragjet e zgjedhura vetëm nga OOF train,
nën 30% është likely real, 30–70% uncertain dhe mbi 70% likely fake.”

**Trego:** `day16_oof_calibration_comparison.png` dhe një vijë me pragjet.

**Mos humb kohë:** me implementimin e bins të ECE.

## 8. Rezultatet e brendshme – 75 sekonda

**Thuaj:**

“Në 792 raste të pastra, accuracy dhe F1 weighted ishin 91.16%. Recall real
92.48%, recall fake 89.82%, me matrix `[[369,30],[40,353]]`. Brier ishte 0.0658.
Zona uncertain dha 91.04% coverage dhe 94.31% accuracy mes vendimeve të forta.”

**Trego:** confusion matrix dhe vetëm 5–6 metrika kryesore.

**Mos humb kohë:** duke lexuar classification report të plotë.

## 9. External validation dhe bias – 90 sekonda

**Thuaj:**

“Krijova një benchmark të ngrirë me 40 raste të reja, të balancuara dhe me
evidence. Baseline-i arriti 47.5% dhe modeli final 60%. Rënia vjen nga domain
shift: përmbledhjet e jashtme kishin rreth 46 fjalë kundrejt 210 brenda,
periudhë, burime dhe stil të ndryshëm. Spearman mes gjatësisë dhe P(fake) u ul
nga -0.713 te -0.612, por fake mbi 250 fjalë mbetën problem.”

**Trego:** tabelën internal vs external dhe
`reports/figures/day17_final_length_performance.png`.

**Mos humb kohë:** duke justifikuar 60% si të ishte e barabartë me 91%; thekso
se është benchmark pilot me domain tjetër.

## 10. Streamlit – 90 sekonda demo

**Thuaj para klikimit:**

“Përdoruesi jep title/content. App-i ngarkon modelin e ngrirë me cache, validon
input-in dhe shfaq probabilitetet, vendimin me tri nivele dhe sinjale
gjuhësore. Sinjalet janë vetëm vëzhgime.”

**Trego:** një `likely_real`, një `likely_fake` dhe një `uncertain`. Për kohë,
analizo live vetëm një rast; dy të tjerat mund t’i kesh si screenshots.

**Mos humb kohë:** me CSS, sidebar ose të gjitha validation edge cases.

## 11. Kufizimet – 60 sekonda

**Thuaj:**

“Kufizimet kryesore janë: nuk ka fact-checking, corpus bias/length bias, domain
shift, benchmark i vogël i jashtëm, high-confidence errors dhe explanation jo
shkakësor. Këto kufizime janë arsyeja për probabilitete, uncertain dhe warning,
jo arsye për t’i fshehur rezultatet.”

**Trego:** 5 kufizime dhe një high-confidence error, `fake_1932`.

**Mos humb kohë:** me një listë të gjatë teknologjish të ardhshme.

## 12. Përfundimi – 30–45 sekonda

**Thuaj:**

“Projekti realizoi një pipeline të riprodhueshëm nga corpus-i te aplikacioni,
me ndarje group-safe, calibration dhe external validation. Modeli klasik është
i fortë brenda corpus-it, por generalization-i është i pjesshëm. Hapi i ardhshëm
më i vlefshëm është zgjerimi i benchmark-ut dhe të dhënave para krahasimit me
XLM-RoBERTa.”

**Trego:** një slide me kontributet dhe QR/path të repository-t.

**Mos humb kohë:** duke përsëritur metrikat e çdo slide-i.

## Rastet e rekomanduara për demo

1. **Likely real:** `true_1594`, P(real) 99.99%.
2. **Likely fake:** `fake_531`, P(fake) 99.99997%.
3. **Uncertain:** `true_585`, P(fake) 49.72%.
4. **Kufizim rezervë FP:** `true_586`, P(fake) 89.64%.
5. **Kufizim rezervë FN:** `fake_1104`, P(real) 89.41%.
6. **High-confidence error:** `fake_1932`, P(real) 99.92%.

Tekstet e plota ruhen te `reports/day19_demo_cases.csv`. Mos i shkruaj live;
përdor sidebar-in ose mbaji gati në një skedar.

---

# 30 pyetje të mundshme të komisionit

## 1. Çfarë problemi zgjidh projekti?

Klasifikon stilin/pattern-et gjuhësore të një lajmi shqip si më të ngjashme me
klasën real ose fake, jep probabilitete dhe një zonë uncertain. Nuk verifikon
faktet ose burimet.

## 2. Pse zgjodhët TF-IDF?

Është përfaqësim i fortë, i shpejtë dhe i interpretueshëm për corpus tekstual
me madhësi të moderuar. Prodhon vektorë sparse që punojnë shumë mirë me modele
lineare dhe krijon baseline të riprodhueshëm pa kërkuar hardware të avancuar.

## 3. Pse jo vetëm Bag of Words?

Count-et e thjeshta mbivlerësojnë fjalët që shfaqen pothuajse kudo. IDF ul
peshën e tyre dhe rrit peshën e termave më dallues; normalizimi gjithashtu e bën
krahasimin më të qëndrueshëm mes dokumenteve me gjatësi të ndryshme.

## 4. Pse Character TF-IDF?

Character n-grams kapin pjesë fjalësh, morfologji, variante drejtshkrimore dhe
fjalë të panjohura. Kjo është e dobishme për shqipen dhe tekstet jo të
standardizuara. Në cohort-in 30–60 fjalë Character-only doli 88.89% kundrejt
77.78% të Word.

## 5. Pse modeli final kombinon Word dhe Character?

Word kap semantikë lokale të fjalëve/bigram-eve, Character kap formën dhe
variantet nënfjalore. Kombinimi dha F1 internal 90.27% në Ditën 13, kundrejt
88.38% Word dhe 88.25% Character, prandaj u zgjodh para krahasimit të
classifier-ëve.

## 6. Çfarë është një sparse vector?

Është vektor me shumë dimensione, por pak vlera jo-zero për një dokument. Ruhet
vetëm informacioni jo-zero. Modeli ka deri 80,000 features, por çdo artikull
aktivizon vetëm një pjesë të vogël.

## 7. Çfarë bëjnë `min_df=2` dhe `max_features`?

`min_df=2` heq features që shfaqen vetëm në një dokument. `max_features`
kufizon fjalorin në 30,000 word dhe 50,000 character features për të kontrolluar
memorien, shpejtësinë dhe kompleksitetin.

## 8. Pse ruajtët uppercase, pikësimin dhe `ë/ç`?

Janë sinjale të stilit dhe character n-grams. Heqja e `ë/ç` ndryshoi dukshëm
prediction-et diagnostike. NFC standardizon përfaqësimin Unicode pa humbur
shkronjat shqipe.

## 9. Pse Logistic Regression ishte baseline?

Është i shpejtë, linear, i interpretueshëm, punon mirë me sparse TF-IDF dhe jep
probabilitete. Arriti rreth 89.9% fillimisht, pra ishte një pikë reference e
fortë.

## 10. Pse Linear SVM fitoi ndaj Logistic Regression?

Max-margin learning është shumë efektiv në hapësira sparse me shumë dimensione.
Me të njëjtin Word+Character, SVM `C=1` dha CV F1 0.9110 ± 0.0021, kundrejt
0.8952 ± 0.0041 të Logistic Regression, dhe balancë më të mirë recall.

## 11. Pse zgjodhët `C=1.0` dhe jo `C=4.0`?

`C=4` ishte vetëm +0.0013 në mesataren CV, por kishte std 0.0060 kundrejt
0.0021. `C=1` ishte më i qëndrueshëm, më i regularizuar dhe doli pak më mirë në
testin e brendshëm. Fitimi shumë i vogël nuk justifikonte variancën shtesë.

## 12. Complement Naive Bayes mori 80% external; pse nuk e zgjodhët?

Sepse selection-i ishte ngrirë nga train/CV përpara external-it. Në CV,
Complement NB kishte vetëm 0.8810 F1 dhe std 0.0106. Zgjedhja nga 40 rastet
external do të ishte tuning mbi benchmark dhe data leakage metodologjike.

## 13. Si interpretohet confusion matrix finale?

`[[369,30],[40,353]]`: 369 real të sakta, 30 real të shënuara fake, 40 fake të
shënuara real dhe 353 fake të sakta. Klasa pozitive është fake.

## 14. Cili gabim është më i rëndë, FP apo FN?

FN mund të lejojë keqinformim të duket real; FP mund të delegjitimojë një lajm
real. Në projekt i raportoj të dyja dhe përdor uncertain për të mos optimizuar
njërën klasë duke dëmtuar rëndë tjetrën.

## 15. Pse accuracy nuk mjafton?

Nuk tregon klasën ose llojin e gabimit dhe mund të fshehë sjellje të pabalancuar.
Prandaj raportohen precision/recall/F1 për fake e real, confusion matrix,
calibration dhe analiza sipas gjatësisë/domain-it.

## 16. Çfarë është probability calibration?

Është mësimi i një mapping-u nga score i classifier-it te probabiliteti që
përputhet më mirë me frekuencat e vëzhguara. U bë me prediction-e out-of-fold
group-safe, jo mbi score in-sample.

## 17. Pse Linear SVM nuk jep probability direkt?

LinearSVC optimizon margin dhe prodhon distancë/decision score nga hyperplane,
jo model probabilistik. Sigmoid calibration e transformon score-n në
`predict_proba()`.

## 18. Pse sigmoid dhe jo isotonic?

Sigmoid kishte Brier 0.0653 dhe log loss 0.2175, më të mira se 0.0659/0.2475,
F1 më të lartë dhe më pak high-confidence errors. Isotonic kishte ECE pak më të
mirë, por është më fleksibël dhe më i rrezikuar nga overfitting.

## 19. Çfarë është Brier score?

Mesatarja e `(p-y)^2` për probability fake. Është 0 për probabilitete perfekte
dhe sa më i ulët aq më mirë. Modeli final mori 0.0658 internal.

## 20. Çfarë ndryshimi kanë log loss dhe ECE?

Log loss vlerëson çdo probabilitet dhe ndëshkon shumë prediction-et e sigurta
e të gabuara. ECE grupon probabilitetet në bins dhe mat hendekun mes confidence
dhe frekuencës së saktë; varet nga binning-u.

## 21. Pse përdorni `uncertain`?

Sepse një vendim i detyruar pranë 0.5 është i rrezikshëm. Zona uncertain i
drejton rastet me confidence të pamjaftueshme te verifikimi. Në testin final,
strong accuracy ishte 94.31% me coverage 91.04%.

## 22. Si u zgjodhën pragjet 0.30/0.70?

U krahasuan 30–70, 35–65 dhe 40–60 vetëm me OOF train predictions. 30–70 dha
strong accuracy më të lartë, 94.88%, dhe zhvendosi 132/277 gabime në uncertain.
Test/external nuk u përdorën për zgjedhje.

## 23. Si kontrolluat data leakage?

Ndarja fillestare mbajti të njëjtin `pair_id` në një split. Para evaluation-it
u hoqën 7 kopje ekzakte train–test. Në CV, groups bashkojnë të njëjtin `pair_id`
dhe `model_text`, dhe verifikohet zero group overlap mes fit/validation.

## 24. Çfarë është `StratifiedGroupKFold`?

Krijon fold-e që ruajnë afërsisht shpërndarjen e label-ave, por nuk ndajnë
asnjë group mes train dhe validation. Kjo mbron çiftet dhe dublikatat nga
leakage.

## 25. Pse modeli mori 91% brenda dhe vetëm 60% jashtë?

Sepse të dhënat ndryshonin: 210 kundrejt 46 fjalë mesatarisht, artikuj të plotë
kundrejt përmbledhjeve manuale, 2020 kundrejt 2024–2026, tema dhe burime të
ndryshme. Kjo është domain shift; testi internal dhe external nuk janë të
barabartë.

## 26. Çfarë prove keni për length bias?

Baseline-i kishte Spearman `rho=-0.713` mes word count dhe P(fake), edhe brenda
secilës klasë. Shkurtimi i artikujve real nga tekst i plotë në rreth 46 fjalë e
rriti mean P(fake) nga 0.2115 në 0.7879. Modeli final e uli correlation në
rreth -0.612, por fake të gjata kishin recall vetëm 44.83%.

## 27. Si mund të gabojë modeli me 99% confidence?

Calibration pasqyron frekuencat e domain-it të trajnimit, jo të vërtetën
universale. Pattern-e spurioze, domain shift, citime, raste të rralla ose tekste
shumë të gjata mund ta çojnë modelin larg kufirit në drejtimin e gabuar.
`fake_1932` është shembull: 99.92% real, por label fake.

## 28. Pse linguistic features nuk janë pjesë e modelit final?

Linguistic-only mori 82.7%, ndërsa hybrid-i 89.02% dhe nuk e kaloi TF-IDF
89.77%. Ato janë të dobishme për analizë dhe explanation, por ishin redundante
dhe të ekspozuara ndaj length bias. Testet sigurojnë që explanation nuk ndryshon
probabilitetin.

## 29. Si e testuat sistemin dhe a do ta përdornit direkt në prodhim?

115 teste mbulojnë data/preprocessing, model config/hash, probabilitete,
thresholds, Unicode, determinism, input validation, assets dhe Streamlit.
Megjithatë nuk do ta përdorja si vendimmarrës autonom: external-i është i vogël
dhe accuracy 60%; do të duhej benchmark më i madh, monitoring dhe human
fact-checking.

## 30. Çfarë do të sillte BERT/XLM-RoBERTa dhe çfarë do të bëje më tej?

Mund të kapë semantikë, kontekst, negacion dhe ngjashmëri përtej n-grams. Por
nuk zgjidh automatikisht dataset bias ose domain shift. Fillimisht do të
zgjeroja dataset-in e jashtëm dhe train-in me burime/periudha/gjatësi të
larmishme, pastaj do ta krahasoja XLM-RoBERTa me të njëjtin group-safe protocol,
calibration dhe test të ngrirë.

---

# Provimi simulues final

## Rregullat

- Kohë: 30–40 minuta.
- Mos përdor shënime ose repository gjatë përgjigjes.
- Përgjigju me zë; një person tjetër le t’i lexojë pyetjet në rend të rastësishëm.
- Çdo pyetje vlen 5 pikë: 3 për përmbajtjen, 1 për saktësinë e termave dhe 1 për
  përgjigje të qartë brenda 60–90 sekondave.
- Kalimi: 80/100. Nën 80, përsërit vetëm temat ku humbe pikë.

## Pyetjet

1. Prezanto problemin, kontributin dhe kufizimin kryesor në 90 sekonda.
2. Vizato pipeline-in final dhe shpjego secilin hap.
3. Shpjego strukturën e corpus-it, `article_id`, `pair_id` dhe split-in.
4. Komisioni thotë: “Nuk kishte pair overlap, pse hoqët edhe 7 artikuj?”
5. Me shembull të vogël, shpjego BoW, TF, DF, IDF dhe sparse vector.
6. Krahaso Word TF-IDF me Character `char_wb` dhe mbro kombinimin final.
7. Shpjego si punon Logistic Regression dhe pse nuk mbeti classifier final.
8. Shpjego Linear SVM, decision boundary, margin dhe rolin e `C`.
9. Mbro `C=1` përballë argumentit se `C=4` kishte mean F1 më të lartë.
10. Nga `[[369,30],[40,353]]`, identifiko TN/FP/FN/TP dhe interpreto recall fake.
11. Shpjego score kundrejt probability dhe pse calibration ishte e nevojshme.
12. Krahaso sigmoid me isotonic duke përdorur rezultatet e Ditës 16.
13. Shpjego Brier, log loss, ECE dhe calibration curve pa formula të tepërta.
14. Mbro zonën uncertain dhe shpjego coverage/strong accuracy.
15. Përshkruaj group-safe CV dhe pse external-i nuk u përdor për tuning.
16. Jep provat kryesore të length bias dhe shpjego Spearman `rho=-0.713`.
17. Shpjego 91.16% internal kundrejt 60% external pa e zhvlerësuar projektin.
18. Analizo `fake_1932`, gabimin me 99.92% P(real), në mënyrë shkencore.
19. Ndiq një input nga widget-i Streamlit deri te probabiliteti dhe UI.
20. Jep tri kufizime, tri masa mbrojtëse aktuale dhe tri punë të ardhshme.

## Checklist i përgjigjeve të forta

Pas provimit kontrollo nëse përmende këto pika:

- **1–4:** jo fact-checking; 3,994; 3,195/799; pair grouping; 7 exact copies;
- **5–6:** IDF ul termat e zakonshëm; sparse; `(1,2)` dhe `char_wb (3,5)`;
- **7–9:** baseline linear/probabilistik; SVM max-margin; stabiliteti i `C=1`;
- **10:** 30 FP, 40 FN, 353 TP; recall fake 89.82%;
- **11–13:** SVM score jo probability; sigmoid; Brier 0.0658, log 0.2192,
  ECE 0.0285;
- **14:** 0.30/0.70; 91.04% coverage; 94.31% strong accuracy;
- **15:** pair + exact-text groups; train-only CV; test/external të ngrira;
- **16:** Spearman negativ, truncation experiment, fake të gjata;
- **17:** gjatësi, periudhë, format, burim dhe source-label confounding;
- **18:** confidence sipas modelit, pattern-e spurioze, jo e vërtetë faktike;
- **19:** validation → cache → `predict_final_news()` → render + warning;
- **20:** bias/domain shift/external i vogël; uncertain/tests/warning; data më të
  larmishme, benchmark më i madh dhe XLM-RoBERTa me të njëjtin protocol.

## Testi i fundit 60-sekondësh

Je gati kur mund të thuash pa u ndalur:

> “Nga 3,994 artikuj krijova train 3,195 dhe test të pastër 792 me kontrolle
> pair/duplicate leakage. Pas preprocessing-ut NFC, modeli final kombinon Word
> TF-IDF `(1,2)` dhe Character `char_wb (3,5)`, përdor Linear SVM `C=1`, sigmoid
> calibration dhe pragjet 0.30/0.70. Arriti 91.16% F1/accuracy brenda dhe 60%
> në benchmark-un e jashtëm me domain shift. Prandaj aplikacioni raporton
> probability sipas modelit, uncertain dhe warning: nuk është fact-checker.”
