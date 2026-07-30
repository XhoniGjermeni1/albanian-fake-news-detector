# Dita 6: Error analysis, calibration dhe pragjet

## Qëllimi

Qëllimi i Ditës 6 ishte kontrolli i cilësisë së modelit TF-IDF përpara përdorimit në aplikacion. U analizuan gabimet, shpërndarja e probabiliteteve, calibration dhe tre zona të mundshme për vendimin `likely_real / uncertain / likely_fake`.

Modeli analizon modele tekstuale dhe gjuhësore. Ai nuk verifikon faktet e lajmit.

## Kontrolli i output-eve të Ditës 5

Të gjitha output-et e nevojshme ekzistonin dhe funksionuan:

- modeli TF-IDF baseline;
- modeli hibrid;
- metrikat e Ditës 5;
- train/test me `model_text`;
- 3994 rreshta me karakteristika gjuhësore;
- funksioni i parashikimit.

Train ka 3195 artikuj dhe testi fillestar 799. Si në Ditën 5, u përjashtuan 7 artikuj të testit me tekst identik me train-in. Error analysis dhe calibration u vlerësuan mbi 792 artikuj.

## Gabimet e TF-IDF baseline

Confusion matrix ishte:

```text
[[377, 22],
 [ 59, 334]]
```

Kjo do të thotë:

- 22 false positives: lajme real të parashikuara fake;
- 59 false negatives: lajme fake të parashikuara real;
- 81 gabime gjithsej;
- 2 gabime me confidence të paktën 90%;
- 20 gabime me probabilitet fake midis 45% dhe 55%;
- 47 gabime brenda zonës 35%–65%.

False negatives janë më të shumta. Kjo tregon se disa lajme fake kanë fjalor dhe formë të ngjashme me artikujt real, veçanërisht kur janë të gjatë ose shkruhen me ton formal.

Të 81 gabimet ruhen te `reports/day6_error_analysis.csv`. Skedari përmban ID, label-in real, prediction, probabilitetet, titullin, content excerpt, linguistic features dhe interpretimin e kujdesshëm të sinjaleve.

## Shembuj interesantë

### `fake_1932`: gabim me confidence shumë të lartë

- Label real: `fake`
- Prediction: `real`
- Probabilitet real: 98.63%
- Titull: “Flet mjekja shqiptare në Kanada: Si të trajtoni virusin në kushtet e shtëpisë”
- `word_count`: 2376
- Sensational/source markers: asnjë
- `diacritic_ratio`: 0.0833
- `uppercase_ratio`: 0.0215

Artikulli është shumë i gjatë dhe nuk ka shenja të dukshme clickbait të kapura nga lista jonë. Kjo mund ta bëjë stilin e tij të ngjajë me artikujt real. Gabimi tregon se gjatësi dhe ton formal nuk janë prova vërtetësie.

### `fake_702`: titull clickbait, por tekst i gjatë

- Label real: `fake`
- Prediction: `real`
- Probabilitet real: 94.14%
- Titull: “LAJMI I FUNDIT – Thaçi: Nëse nuk arrihet marrëveshja...”
- `word_count`: 2233
- Sensational markers të gjetura: asnjë

Lista manuale përmban “lajm i fundit”, por nuk kapi variantin “lajmi i fundit”. Teksti i gjatë dominoi sinjalet e dukshme të titullit. Ky rast tregon kufizimin e përputhjes ekzakte të shprehjeve.

### `fake_603`: lajm fake me stil mjekësor formal

- Label real: `fake`
- Prediction: `real`
- Probabilitet real: 89.11%
- `word_count`: 501
- Pa sensational markers dhe pa pikëçuditëse

Titulli dhe fjalori mjekësor mund t'i ngjajnë raportimit real. TF-IDF njeh kombinime fjalësh, por nuk kontrollon nëse pretendimi mjekësor është i vërtetë.

### `true_232`: lajm real i parashikuar fake

- Label real: `real`
- Prediction: `fake`
- Probabilitet fake: 85.50%
- Source marker: `deklaroi`
- `diacritic_ratio`: 0.0000

Titulli përmban disa shkronja me Unicode të dekompozuar. Numërimi aktual i `ë/ç` nuk i njeh ato si formatet e zakonshme të parakompozuara. Ky është kufizim i normalizimit Unicode dhe mund të shtrembërojë disa linguistic features.

### `true_786` dhe `true_221`: stil emocional/clickbait te artikuj real

`true_786` kishte vetëm 46 fjalë dhe titullin “Ekskluzive: ... vaksinë ... shkakton vdekje”. Modeli dha 67.32% fake. `true_221` kishte titull emocional dhe mori 64.93% fake.

Këto raste tregojnë se stil sensacional ose emocional mund të ekzistojë edhe te artikujt e etiketuar real. Stili nuk duhet barazuar me pavërtetësinë.

### Gabime afër 50/50

`fake_343` mori 49.83% fake dhe `fake_756` mori 49.99% fake. Këto janë raste ku zona `uncertain` është më e ndershme se një vendim i fortë.

Shembujt e përzgjedhur ruhen te `reports/day6_interesting_errors.csv`.

## Shpërndarja e probabiliteteve

| Treguesi | Pa calibration | Pas sigmoid calibration |
| --- | ---: | ---: |
| Probabilitete ≤10% ose ≥90% | 129 | 509 |
| Gabime me confidence ≥90% | 2 | 19 |
| Norma e gabimit brenda confidence ≥90% | 1.55% | 3.73% |
| Raste 45%–55% | 55 | 26 |
| Raste 35%–65% | 172 | 79 |

Baseline ishte relativisht i përmbajtur dhe kishte shumë raste në mes. Sigmoid calibration i zhvendosi shumë raste drejt skajeve. Prandaj calibration përmirësoi besueshmërinë mesatare, por nuk eliminon gabimet me probabilitet të lartë.

Grafiku i calibration curve dhe histogrami ruhen te `reports/figures/day6_probability_calibration.png`.

## Probability calibration

U përdor `CalibratedClassifierCV` me metodën `sigmoid`, 5 fold-e dhe `ensemble=False`. Fold-et u ndërtuan me `StratifiedGroupKFold`. Artikujt me të njëjtin `pair_id` ose tekst identik u mbajtën në të njëjtin grup, ndaj calibration nuk pa kopje të fold-it të validimit gjatë trajnimit.

| Metrika | Pa calibration | Pas calibration | Ndryshimi |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8977 | 0.8826 | -0.0151 |
| F1 weighted | 0.8975 | 0.8825 | -0.0150 |
| F1 fake | 0.8919 | 0.8797 | -0.0122 |
| Brier score | 0.1008 | 0.0819 | -0.0189 |
| Log loss | 0.3512 | 0.2868 | -0.0644 |
| Expected calibration error | 0.1317 | 0.0480 | -0.0837 |

Për Brier score, log loss dhe calibration error, vlera më e ulët është më e mirë. Calibration i përmirësoi të tre treguesit probabilistikë, por uli pak metrikat e klasifikimit me pragun 50%.

Kjo është një marrëveshje e arsyeshme për aplikacionin vetëm nëse probabiliteti paraqitet si rezultat statistikor, përdoret zona `uncertain` dhe ruhet paralajmërimi për verifikimin faktik.

## Pragjet e testuara

Pragjet u testuan mbi probabilitetin fake të modelit të kalibruar.

| Varianti | likely_real | uncertain | likely_fake | Gabime të kaluara në uncertain | Gabime në vendime të forta | FN të forta | FP të forta | Coverage | Accuracy e vendimeve të forta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35%–65% | 368 | 79 | 345 | 39 | 54 | 35 | 19 | 90.03% | 92.43% |
| 40%–60% | 380 | 56 | 356 | 32 | 61 | 39 | 22 | 92.93% | 91.71% |
| 30%–70% | 351 | 110 | 331 | 51 | 42 | 32 | 10 | 86.11% | 93.84% |

Varianti 30%–70% është më konservatori dhe dha accuracy më të lartë për vendimet e forta. Ai vendos 110 artikuj, ose 13.89% të testit, në `uncertain` dhe kap 51 nga 93 gabimet që do të bëheshin me pragun e zakonshëm 50%.

Pragjet janë zgjedhur në mënyrë eksploruese mbi këtë test set. Ato duhet të kontrollohen përsëri me cross-validation ose me të dhëna të reja përpara një përdorimi real.

## Kandidati për aplikacionin

Rekomandimi fillestar është:

- prediction: TF-IDF + Logistic Regression me sigmoid calibration;
- `probability_fake < 0.30`: `likely_real`;
- `0.30 ≤ probability_fake ≤ 0.70`: `uncertain`;
- `probability_fake > 0.70`: `likely_fake`;
- linguistic features përdoren vetëm për shpjegim, jo për ndryshimin e prediction;
- çdo rezultat shfaq paralajmërimin se modeli nuk verifikon faktet.

Logjika është përgatitur te `predict_news_for_app()` në `src/models/predict.py`. Output-i përmban probabilitetet, vendimin me tre nivele, thresholds, fjalët sensacionale, source markers, pikëçuditëset, word count, gjatësinë, `diacritic_ratio`, `uppercase_ratio` dhe paralajmërimin.

## Kufizime

- Error analysis bazohet në një test set të vetëm.
- Pragjet u krahasuan mbi të njëjtin test set dhe janë ende pragje fillestare.
- Calibration përmirëson treguesit mesatarë, por mund të gabojë me probabilitet ekstrem.
- Dataseti mund të përmbajë sinjale të burimit, temës dhe gjatësisë.
- Listat e shprehjeve janë manuale dhe nuk kapin çdo variant gramatikor.
- Tekstet me Unicode të dekompozuar duan normalizim të veçantë.
- As TF-IDF dhe as linguistic features nuk bëjnë fact-checking.

## Hapi i rekomanduar për Ditën 7

Të ndërtohet aplikacioni minimal Streamlit mbi `predict_news_for_app()`. App-i duhet të ngarkojë modelin vetëm një herë, të pranojë titull/përmbajtje, të shfaqë tre vendimet dhe probabilitetet, të paraqesë linguistic explanation dhe të mbajë paralajmërimin e verifikimit faktik gjithmonë të dukshëm.
