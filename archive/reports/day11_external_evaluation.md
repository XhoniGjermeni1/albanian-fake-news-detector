# Dita 11 - Vlerësimi në datasetin e jashtëm

## Qëllimi dhe integriteti i eksperimentit

Modeli ekzistues `models/calibrated_tfidf_logreg.joblib` u ngarkua vetëm për
inference. Çdo rresht u analizua me `predict_news_for_app()`, i cili përdor të
njëjtin `combine_title_content()` dhe të njëjtat pragje si aplikacioni:

- `probability_fake < 0.30`: `likely_real`;
- `0.30 <= probability_fake <= 0.70`: `uncertain`;
- `probability_fake > 0.70`: `likely_fake`.

Nuk u thirr asnjë metodë `fit`, modeli nuk u ritrajnua, dataset-i nuk u ndryshua
dhe fingerprint-et SHA-256 të modelit dhe datasetit mbetën të pandryshuara para
dhe pas vlerësimit.

## Metrikat binare

Prediction-i binar përdor pragun standard `probability_fake >= 0.50` për `fake`.
Precision, recall dhe F1 kryesore janë mesatare të ponderuara, në përputhje me
raportet e mëparshme të projektit.

| Metrika | Rezultati |
| --- | ---: |
| Accuracy | 47.50% |
| Precision weighted | 40.99% |
| Recall weighted | 47.50% |
| F1 weighted | 35.93% |
| Precision macro | 40.99% |
| Recall macro | 47.50% |
| F1 macro | 35.93% |
| Precision fake | 48.65% |
| Recall fake | 90.00% |
| F1 fake | 63.16% |
| Precision real | 33.33% |
| Recall real | 5.00% |
| F1 real | 8.70% |

Confusion matrix, me rreshta `true` dhe kolona `predicted` në rendin
`[real, fake]`:

```text
[[1, 19], [2, 18]]
```

- True real: **1**;
- false positives, real të klasifikuara fake: **19**;
- false negatives, fake të klasifikuara real: **2**;
- true fake: **18**.

Dataset-i është i balancuar, prandaj një klasifikues konstant do të kishte 50%
accuracy. Rezultati 47.50% është
2.50 pikë përqindjeje
nën këtë baseline të thjeshtë. Modeli
dalloi 18 nga 20 rastet fake,
por vetëm 1 nga 20 rastet
real. Pra recall-i i mirë për fake vjen së bashku me një numër
shumë të lartë false positives.

## Vendimet me tri nivele

- `likely_real`: **1**;
- `uncertain`: **8**;
- `likely_fake`: **31**;
- coverage e vendimeve të forta: **80.00%**;
- accuracy e vendimeve të forta: **56.25%**;
- gabime binare të zhvendosura në `uncertain`: **7**
  nga 21 (33.33%);
- gabime që mbetën vendime të forta: **14**.

Zona `uncertain` ishte e dobishme si sinjal paralajmërues:
7 nga 8 rastet
e saj ishin gabime binare. Megjithatë, ajo nuk e zgjidhi zhvendosjen e
përgjithshme drejt klasës fake; 14 gabime
mbetën vendime të forta.

## Sipas label-it

| true_label | rows | correct | accuracy | likely_real | uncertain | likely_fake | mean_probability_fake | mean_word_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fake | 20 | 18 | 0.9000 | 1 | 1 | 18 | 0.8167 | 46.9500 |
| real | 20 | 1 | 0.0500 | 0 | 7 | 13 | 0.7474 | 44.2000 |

Të gjitha burimet institucionale i përkasin klasës real dhe të gjitha pretendimet
e dokumentuara nga Krypometër klasës fake. Për këtë arsye ndikimi i burimit nuk
mund të ndahet statistikisht nga ndikimi i label-it në këtë dataset.

## Sipas temës

| topic | rows | accuracy | precision_fake | recall_fake | false_positives | false_negatives | uncertain | strong_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ekonomi | 8 | 0.3750 | 0.4000 | 0.5000 | 3 | 2 | 3 | 0.4000 |
| politikë | 8 | 0.5000 | 0.5000 | 1.0000 | 4 | 0 | 2 | 0.6667 |
| shëndetësi | 8 | 0.5000 | 0.5000 | 1.0000 | 4 | 0 | 0 | 0.5000 |
| sociale | 8 | 0.5000 | 0.5000 | 1.0000 | 4 | 0 | 3 | 0.8000 |
| teknologji | 8 | 0.5000 | 0.5000 | 1.0000 | 4 | 0 | 0 | 0.5000 |

Tema me rezultatin më të dobët ishte `ekonomi` me
37.50%. Rezultatin më të lartë
50.00% e arritën: politikë, shëndetësi, sociale, teknologji. Kjo nuk nënkupton
domosdoshmërisht balancë të mirë. Në temat politikë, shëndetësi, sociale, teknologji, modeli
gjeti të gjitha rastet fake dhe humbi të gjitha rastet real.

## Sipas gjatësisë

Grupet u përcaktuan mbi tekstin e plotë që pa modeli: 38-44 fjalë, 45-47 fjalë
dhe 48-51 fjalë.

| length_group | rows | accuracy | false_positives | false_negatives | uncertain | mean_probability_fake |
| --- | --- | --- | --- | --- | --- | --- |
| long_48_51 | 13 | 0.4615 | 5 | 2 | 1 | 0.7678 |
| medium_45_47 | 15 | 0.6667 | 5 | 0 | 3 | 0.7786 |
| short_38_44 | 12 | 0.2500 | 9 | 0 | 4 | 0.8017 |

Të gjitha tekstet e jashtme janë pranë kufirit minimal të test set-it të
brendshëm. Gjatësia nuk ndahet qartë mes real/fake në datasetin e jashtëm, ndërsa
në test set-in e brendshëm mediana ishte
206 fjalë për real dhe
100 për fake. Kjo e bën
shkurtësinë një shpjegim të mundshëm për prirjen e fortë drejt fake.

## Sipas burimit

| source | rows | real_rows | fake_rows | accuracy | uncertain | mean_probability_fake |
| --- | --- | --- | --- | --- | --- | --- |
| Banka e Shqipërisë | 2 | 2 | 0 | 0.0000 | 1 | 0.7378 |
| INSTAT | 2 | 2 | 0 | 0.5000 | 1 | 0.5572 |
| Krypometër / ofertë në Facebook | 1 | 0 | 1 | 1.0000 | 0 | 0.8049 |
| Krypometër / postim në Facebook | 16 | 0 | 16 | 0.8750 | 1 | 0.8192 |
| Krypometër / postim në Instagram | 2 | 0 | 2 | 1.0000 | 0 | 0.7470 |
| Krypometër / postim në rrjete sociale | 1 | 0 | 1 | 1.0000 | 0 | 0.9270 |
| Këshilli i Ministrave i Shqipërisë | 16 | 16 | 0 | 0.0000 | 5 | 0.7724 |

Burimi nuk futet në model. Kjo tabelë analizon sjelljen pas prediction-it dhe nuk
provon shkakësi. Rezultati tregon megjithatë se formati i shkurtër institucional
nuk u trajtua si artikujt real të gjatë të corpus-it.

## Rastet uncertain

| external_id | true_label | binary_prediction | probability_real | probability_fake | topic | word_count | title |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXT-R-002 | real | fake | 0.4086 | 0.5914 | politikë | 43 | Shqipëria merr pjesë në Konferencën e tetë Ndërqeveritare me BE-në |
| EXT-R-003 | real | fake | 0.3305 | 0.6695 | politikë | 43 | Shqipëria mbyll provizorisht tre kapituj negociatash me BE-në |
| EXT-R-009 | real | fake | 0.4654 | 0.5346 | ekonomi | 45 | Banka e Shqipërisë mban të pandryshuar normën bazë të interesit |
| EXT-R-011 | real | real | 0.6435 | 0.3565 | ekonomi | 47 | INSTAT raporton rritje të PBB-së në tremujorin e parë të vitit 2026 |
| EXT-R-013 | real | fake | 0.3461 | 0.6539 | sociale | 45 | Fondi Social 2026 financon katërmbëdhjetë shërbime të reja |
| EXT-R-014 | real | fake | 0.3268 | 0.6732 | sociale | 41 | Mbahet finalja e Festivalit Kombëtar të Shkencës |
| EXT-R-015 | real | fake | 0.3802 | 0.6198 | sociale | 40 | Shkolla 22 Tetori në Berat rikonstruktohet për rreth 500 nxënës |
| EXT-F-012 | fake | real | 0.5889 | 0.4111 | ekonomi | 50 | Mega Mall dhuron qindra soba dhe enë gatimi për ndjekësit |

## Gabimet me probabilitet të lartë

U gjetën **3** gabime me të paktën 90% siguri:
`EXT-R-004, EXT-R-010, EXT-R-019`. Kjo tregon se probabiliteti i kalibruar është besimi i
modelit brenda sinjaleve që ka mësuar, jo garanci faktike ose garanci
përgjithësimi jashtë shpërndarjes.

## Analiza e çdo gabimi

Interpretimet e mëposhtme janë hipoteza të kujdesshme mbi stilin dhe sinjalet që
modeli sheh; ato nuk provojnë shkakun e saktë të çdo prediction-i.

### EXT-F-010 - false_negative

- Titulli: Arben Gashi është ndër pesë personat më të pasur në Gjermani
- Label / prediction: `fake` / `real`
- Probabiliteti Real / Fake: 70.63% / 29.37%
- Vendimi / tema / fjalët: `likely_real` / `ekonomi` / 51
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.75%; diakritika=7.07%
- Interpretim: Pretendimi fake është përmbledhur me stil neutral; prova e fact-check-ut ruhet veçmas dhe nuk i jepet modelit, prandaj formulimi mund të duket si lajm real. Nuk u gjet asnjë marker nga lista e kufizuar sensacionale.

### EXT-F-012 - false_negative

- Titulli: Mega Mall dhuron qindra soba dhe enë gatimi për ndjekësit
- Label / prediction: `fake` / `real`
- Probabiliteti Real / Fake: 58.89% / 41.11%
- Vendimi / tema / fjalët: `uncertain` / `ekonomi` / 50
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.75%; diakritika=7.81%
- Interpretim: Pretendimi fake është përmbledhur me stil neutral; prova e fact-check-ut ruhet veçmas dhe nuk i jepet modelit, prandaj formulimi mund të duket si lajm real. Nuk u gjet asnjë marker nga lista e kufizuar sensacionale. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

### EXT-R-010 - false_positive

- Titulli: Fondi i Ndërhyrjes së Jashtëzakonshme arrin 8.22 miliardë lekë
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 5.90% / 94.10%
- Vendimi / tema / fjalët: `likely_fake` / `ekonomi` / 41
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=4.00%; diakritika=6.71%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (41 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Gabimi mbeti me të paktën 90% siguri; calibration nuk garanton saktësi për një rast individual jashtë shpërndarjes së trajnimit.

### EXT-R-019 - false_positive

- Titulli: Prezantohet shoqëria publike Albanian Digital Solutions
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 7.20% / 92.80%
- Vendimi / tema / fjalët: `likely_fake` / `teknologji` / 41
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.96%; diakritika=2.24%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (41 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Gabimi mbeti me të paktën 90% siguri; calibration nuk garanton saktësi për një rast individual jashtë shpërndarjes së trajnimit.

### EXT-R-004 - false_positive

- Titulli: Qeveria diskuton kalendarin e negociatave me Bashkimin Europian
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 9.53% / 90.47%
- Vendimi / tema / fjalët: `likely_fake` / `politikë` / 49
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=4.17%; diakritika=2.95%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (49 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Gabimi mbeti me të paktën 90% siguri; calibration nuk garanton saktësi për një rast individual jashtë shpërndarjes së trajnimit.

### EXT-R-020 - false_positive

- Titulli: Administrata publike përdor mjete të inteligjencës artificiale në procesin e integrimit
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 12.78% / 87.22%
- Vendimi / tema / fjalët: `likely_fake` / `teknologji` / 46
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.01%; diakritika=4.90%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (46 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-018 - false_positive

- Titulli: Samiti Shqipëri-Izrael mbledh kompani të teknologjisë dhe sigurisë kibernetike
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 13.00% / 87.00%
- Vendimi / tema / fjalët: `likely_fake` / `teknologji` / 41
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=3.32%; diakritika=3.79%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (41 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-006 - false_positive

- Titulli: Prezantohet projekti për qendrën spitalore Health Hub Hospital në Tiranë
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 16.48% / 83.52%
- Vendimi / tema / fjalët: `likely_fake` / `shëndetësi` / 49
- Sinjale: sensacionalë=asnjë; burimi=sipas; !=0; uppercase=5.00%; diakritika=3.92%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (49 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. U gjetën markerë burimi: sipas.

### EXT-R-001 - false_positive

- Titulli: Prezantohet programi EU4Municipalities II për bashkitë shqiptare
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 16.63% / 83.37%
- Vendimi / tema / fjalët: `likely_fake` / `politikë` / 48
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=5.94%; diakritika=3.68%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (48 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-008 - false_positive

- Titulli: Buxhetit të shëndetësisë i shtohen fonde për barnat onkologjike
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 17.17% / 82.83%
- Vendimi / tema / fjalët: `likely_fake` / `shëndetësi` / 40
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=1.24%; diakritika=6.19%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (40 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-007 - false_positive

- Titulli: Tetë spitale rajonale ofrojnë njësi kimioterapie
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 19.02% / 80.98%
- Vendimi / tema / fjalët: `likely_fake` / `shëndetësi` / 45
- Sinjale: sensacionalë=asnjë; burimi=raportoi; !=0; uppercase=6.12%; diakritika=8.39%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (45 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. U gjetën markerë burimi: raportoi.

### EXT-R-017 - false_positive

- Titulli: Nis faza e parë e projektit Smart City në njëzet shkolla të Tiranës
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 21.07% / 78.93%
- Vendimi / tema / fjalët: `likely_fake` / `teknologji` / 48
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=3.56%; diakritika=5.61%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (48 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-012 - false_positive

- Titulli: Eksportet dhe importet e mallrave rriten në qershor 2026
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 24.22% / 75.78%
- Vendimi / tema / fjalët: `likely_fake` / `ekonomi` / 38
- Sinjale: sensacionalë=asnjë; burimi=raportoi; !=0; uppercase=4.08%; diakritika=5.36%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (38 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. U gjetën markerë burimi: raportoi.

### EXT-R-005 - false_positive

- Titulli: Spitali Rajonal i Lezhës zgjeron shërbimet e specializuara
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 25.69% / 74.31%
- Vendimi / tema / fjalët: `likely_fake` / `shëndetësi` / 48
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.57%; diakritika=6.83%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (48 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-016 - false_positive

- Titulli: Dyfishohet mbështetja për fëmijët dhe të rinjtë me aftësi të kufizuara
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 26.44% / 73.56%
- Vendimi / tema / fjalët: `likely_fake` / `sociale` / 46
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=1.16%; diakritika=7.86%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (46 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature.

### EXT-R-014 - false_positive

- Titulli: Mbahet finalja e Festivalit Kombëtar të Shkencës
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 32.68% / 67.32%
- Vendimi / tema / fjalët: `uncertain` / `sociale` / 41
- Sinjale: sensacionalë=asnjë; burimi=sipas; !=0; uppercase=3.56%; diakritika=6.31%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (41 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. U gjetën markerë burimi: sipas. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

### EXT-R-003 - false_positive

- Titulli: Shqipëria mbyll provizorisht tre kapituj negociatash me BE-në
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 33.05% / 66.95%
- Vendimi / tema / fjalët: `uncertain` / `politikë` / 43
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=4.20%; diakritika=6.13%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (43 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

### EXT-R-013 - false_positive

- Titulli: Fondi Social 2026 financon katërmbëdhjetë shërbime të reja
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 34.61% / 65.39%
- Vendimi / tema / fjalët: `uncertain` / `sociale` / 45
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.20%; diakritika=8.77%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (45 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

### EXT-R-015 - false_positive

- Titulli: Shkolla 22 Tetori në Berat rikonstruktohet për rreth 500 nxënës
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 38.02% / 61.98%
- Vendimi / tema / fjalët: `uncertain` / `sociale` / 40
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=3.12%; diakritika=7.80%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (40 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

### EXT-R-002 - false_positive

- Titulli: Shqipëria merr pjesë në Konferencën e tetë Ndërqeveritare me BE-në
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 40.86% / 59.14%
- Vendimi / tema / fjalët: `uncertain` / `politikë` / 43
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=4.87%; diakritika=7.28%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (43 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

### EXT-R-009 - false_positive

- Titulli: Banka e Shqipërisë mban të pandryshuar normën bazë të interesit
- Label / prediction: `real` / `fake`
- Probabiliteti Real / Fake: 46.54% / 53.46%
- Vendimi / tema / fjalët: `uncertain` / `ekonomi` / 45
- Sinjale: sensacionalë=asnjë; burimi=asnjë; !=0; uppercase=2.81%; diakritika=9.77%
- Interpretim: Rasti real u shty drejt fake; përmbledhja shumë e shkurtër mund të ketë ngjarë me anën e shkurtër të corpus-it (45 fjalë kundrejt medianës 206 për real në test set). Burimi institucional është vetëm metadata dhe modeli nuk e sheh atë si feature. Pragjet e zhvendosën këtë gabim binar në zonën uncertain.

## Krahasimi me test set-in e brendshëm

| Treguesi | Test set-i i brendshëm | Dataseti i jashtëm |
| --- | ---: | ---: |
| Raste | 792 | 40 |
| Accuracy binare | 88.38% | 47.50% |
| Coverage e fortë | 86.11% | 80.00% |
| Accuracy e fortë | 93.99% | 56.25% |
| Uncertain | 110 | 8 |
| Mesatarja e fjalëve | 210.46 | 45.58 |
| Mediana e fjalëve | 139.00 | 46.00 |

Krahasimi nuk është eksperiment i barabartë:

- test set-i i brendshëm ka artikuj të plotë dhe 792 raste; dataseti i jashtëm
  ka 40 përmbledhje manuale;
- mesatarja ra nga 210.46 në 45.58
  fjalë;
- corpus-i i brendshëm mbulon periudhën
  2020-04-27 deri
  2020-07-29, ndërsa rastet e jashtme periudhën
  2024-07-30 deri
  2026-07-31;
- temat e jashtme janë të balancuara me dorë;
- burimet dhe label-et e jashtme janë të ndërthurura: institucionale për real dhe
  pretendime sociale të fact-check-uara për fake;
- provat e etiketimit nuk iu dhanë modelit, sepse aplikacioni analizon vetëm
  titullin dhe përmbajtjen.

Rënia e accuracy ishte
**40.88**
pikë përqindjeje. Për shkak të këtyre ndryshimeve, kjo nuk mat vetëm cilësinë e
modelit; mat edhe domain shift-in mes artikujve të corpus-it dhe përmbledhjeve të
shkurtra të jashtme.

## Përfundimi

Në këtë vlerësim modeli përgjithësoi **dobët**. Ai ruajti recall
90.00% për fake, por accuracy totale
47.50%, recall 5.00%
për real dhe 19 false positives tregojnë se nuk e ndan
në mënyrë të besueshme dy klasat jashtë corpus-it. Probabilitetet dhe `uncertain`
ndihmuan të shënohen 7 gabime, por
modeli dha ende 14 vendime të forta të gabuara
dhe 3 gabime mbi 90% siguri.

Ky rezultat nuk duhet përdorur për të ndryshuar datasetin ose për të zgjedhur
pragje të reja pas shikimit të përgjigjeve. Dataset-i i Ditës 10 duhet të mbetet
i ngrirë si kontroll i jashtëm.

## Rekomandimi për Ditën 12

1. Të analizohet në mënyrë të kontrolluar bias-i i gjatësisë, duke krahasuar
   artikuj të brendshëm dhe të jashtëm me gjatësi të ngjashme, pa ndryshuar këtë
   benchmark.
2. Të mblidhen më vonë artikuj të jashtëm më të plotë dhe burime të kryqëzuara,
   ku edhe real edhe fake vijnë nga disa lloje burimesh.
3. Të kontrollohet stabiliteti i probabiliteteve ndaj versionit të shkurtër dhe
   të zgjeruar të të njëjtit lajm.
4. Vetëm pas dokumentimit të këtyre analizave të vendoset nëse duhet një model i
   përmirësuar, linguistic features, ribalancim ose ndryshim i politikës
   `uncertain`. Vlerësimi i sotëm duhet të ruhet i pandryshuar.

## Output-et

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
