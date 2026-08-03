# Dita 9 - Testimi i sistemit të plotë

## Qëllimi

Qëllimi i Ditës 9 ishte testimi serioz i aplikacionit nga input-i deri te rezultati, pa trajnuar ose ndryshuar modelin. U testuan validimi, prediction-i, probabilitetet, pragjet, shpjegimi gjuhësor dhe sjellja me input joideal.

## Gjendja para testimit

App-i dhe `predict_news_for_app(title, content)` ishin funksionalë. Të gjitha 39 testet ekzistuese kaluan para ndryshimeve. Modeli i përdorur mbeti `models/calibrated_tfidf_logreg.joblib` dhe nuk u ritrajnua.

## Metoda e testimit

U krijua `src/models/evaluate_app_system.py`, i cili:

- ngarkon modelin vetëm një herë;
- ekzekuton 16 raste sintetike dhe joideale;
- kontrollon 792 artikujt e test set-it pa 7 dublikatat ekzakte të train-it;
- verifikon kufijtë dhe shumën e probabiliteteve;
- verifikon vendimin ndaj pragjeve 30%/70%;
- përzgjedh gjashtë shembuj demo dhe gabime reale;
- ruan rezultatet në CSV dhe JSON.

## Rastet e testuara

U testuan:

- input bosh dhe input vetëm me pikësim/emoji;
- vetëm titull dhe vetëm përmbajtje;
- tekst shumë i shkurtër;
- tekst normal;
- tekst me 24,000+ karaktere dhe tekst mbi kufirin 100,000;
- shumë shkronja të mëdha;
- shumë pikëçuditëse;
- stil clickbait dhe stil zyrtar;
- tregues burimi;
- tekst pa `ë/ç`;
- Unicode i dekompozuar, emoji, thonjëza tipografike dhe vizë Unicode;
- shembuj real/fake nga test set-i;
- raste `likely_real`, `uncertain` dhe `likely_fake`.

Rezultati: **16/16 raste kaluan**. Tri input-e u bllokuan siç pritej dhe 13 u analizuan pa crash.

Pas shtimit të testeve për Unicode, input-et jo-informuese, invariantët e sistemit dhe ndërfaqen Streamlit, e gjithë paketa përfundoi me:

```text
49 passed
```

## Kontrolli i probabiliteteve dhe pragjeve

Në 792 artikujt e test set-it:

- probabilitete jashtë intervalit `[0, 1]`: **0**;
- gabimi maksimal i shumës `P(Real) + P(Fake)`: **0.0**;
- vendime që nuk përputheshin me pragjet: **0**;
- `likely_real`: **352**;
- `uncertain`: **110**;
- `likely_fake`: **330**.

Zona `uncertain` përmbante 70 artikuj real dhe 40 fake sipas label-it të corpus-it. Ajo zhvendosi 51 nga 92 gabimet e klasifikimit binar në një vendim të pasigurt.

Vendimet e forta mbuluan **86.11%** të test set-it dhe kishin saktësi **93.99%**. Megjithatë, mbetën 41 vendime të forta të gabuara: 9 false positives dhe 32 false negatives.

## Rezultatet e klasifikimit binar

- Accuracy: **88.38%**;
- artikuj të saktë: **700/792**;
- confusion matrix: `[[360, 39], [53, 340]]`;
- false positives: **39**;
- false negatives: **53**;
- gabime me siguri të paktën 90%: **18**.

Këto shifra tregojnë se calibration dhe zona `uncertain` e bëjnë rezultatin më të kujdesshëm, por nuk eliminojnë gabimet me probabilitet të fortë.

## Problemet e gjetura dhe të rregulluara

### 1. Unicode i dekompozuar

Disa tekste përdornin `e + diaeresis` ose `c + cedilla` në vend të karaktereve NFC `ë/ç`. Kjo ndikonte fort te TF-IDF. Artikulli `true_232` kalonte nga P(Fake) 99.02% pa normalizim në 49.62% pas normalizimit.

U shtua Unicode NFC në preprocessing dhe në linguistic features. Në test set, tri probabilitete ndryshuan dhe një klasifikim binar u korrigjua.

### 2. Input pa informacion tekstual

Input-i vetëm me `!!!`, emoji ose karaktere të padukshme mund të merrte prediction. Tani bllokohet me mesazhin që kërkohet të paktën një shkronjë ose numër.

### 3. Paralajmërimi për tekst të shkurtër

Teksti i shkurtër mori P(Fake) 93.81%. Warning-u tani sqaron se edhe një përqindje e lartë mund të jetë më pak e besueshme kur mungon informacioni.

### 4. Lista e shprehjeve sensacionale

Shpjegimi njihte `lajm i fundit`, por jo `lajmi i fundit`, dhe nuk njihte `ekskluzive`. Të dy variantet u shtuan vetëm për explanation; prediction-i nuk ndryshoi.

## Shembujt e demonstrimit

Label-i në tabelë është label-i i corpus-it dhe nuk përfaqëson një fact-check të ri të kryer gjatë Ditës 9.

| Lloji | ID / label | Titulli | Pjesë e përmbajtjes | Vendimi | Real / Fake |
|---|---|---|---|---|---:|
| `likely_real` | `true_612` / real | A do të zgjatet leja për punëtorët nga Kosova në Gjermani? | Leja speciale për punëtorët migrantë... Sipas një projektligji... | `likely_real` | 85.13% / 14.87% |
| `likely_fake` | `fake_1939` / fake | Vjosa Osmani ia dërgon një “selam” LDK-së... | Nënkryetarja e LDK-së duket se vazhdon... | `likely_fake` | 14.87% / 85.13% |
| `uncertain` | `true_232` / real | Këshilltari i Bidenit e vlerëson lart Thaçin | Michael Carpenter, këshilltar i Joe Biden... | `uncertain` | 50.38% / 49.62% |
| False positive | `true_786` / real | Ekskluzive: Alfred Cako zbulon vaksinën... | Alfred Cako konspiracionisti shqiptar ishte mysafir... | `likely_fake` | 10.77% / 89.23% |
| False negative | `fake_1932` / fake | Flet mjekja shqiptare në Kanada... | Në një intervistë dhënë për gazetën, doktoresha... | `likely_real` | 99.99% / 0.01% |
| Gabim me siguri të lartë | `fake_702` / fake | LAJMI I FUNDIT - Thaçi: Nëse nuk arrihet marrëveshja... | Presidenti i Kosovës, Hashim Thaçi, tha në një intervistë... | `likely_real` | 99.80% / 0.20% |

Karakteristikat kryesore:

| ID | Fjalë | `!` | Uppercase | `ë/ç` | Marker-a |
|---|---:|---:|---:|---:|---|
| `true_612` | 163 | 0 | 4.89% | 8.26% | `sipas`, `studimi`, `mund të` |
| `fake_1939` | 146 | 0 | 4.20% | 7.13% | asnjë nga listat |
| `true_232` | 170 | 0 | 4.20% | 7.92% | `deklaroi` |
| `true_786` | 46 | 0 | 4.49% | 4.59% | `ekskluzive` |
| `fake_1932` | 2376 | 1 | 2.15% | 8.33% | `mund të` |
| `fake_702` | 2233 | 0 | 3.48% | 8.64% | `lajmi i fundit`, `ndoshta`, `mund të` |

Të dhënat e plota të shembujve ruhen te `reports/day9_demo_examples.csv`.

## Interpretimi i gabimeve

### Fake formal i klasifikuar real

`fake_1932` ka 2376 fjalë dhe stil të gjatë, të strukturuar e informues. Modeli e klasifikoi `likely_real` me 99.99% për Real. Kjo tregon mbështetje të fortë te ngjashmëria stilistike dhe gjatësia, jo te faktet.

### Real emocional i klasifikuar fake

`true_786` ka titullin “Ekskluzive...” dhe pretendime emocionale mbi vdekjen dhe lindjet. Edhe pse label-i i corpus-it është real, modeli dha 89.23% Fake. Ky rast tregon se një artikull real mund të raportojë pretendime sensacionale.

### Gabim shumë i sigurt

`fake_702` përmban “LAJMI I FUNDIT”, por është shumë i gjatë dhe formal. Modeli dha 99.80% Real. Edhe probability calibration nuk garanton që një prediction individual është i saktë.

## Probleme që mbeten

- Teksti shumë i shkurtër mund të marrë prediction shumë të fortë.
- Mungesa e `ë/ç` krijon bias të dukshëm; rasti sintetik pa diakritika mori 99.35% Fake.
- Tekstet fake të gjata dhe formale mund të klasifikohen fort si real.
- Tekstet real që raportojnë pretendime emocionale mund të klasifikohen fake.
- 18 gabime kishin siguri të paktën 90%.
- Listat e marker-ave janë të kufizuara dhe nuk mbulojnë çdo variant gjuhësor.
- Label-et e corpus-it nuk zëvendësojnë verifikimin aktual të fakteve.
- Modeli nuk përdor reputacion burimi, datë, autor, URL ose prova të jashtme.

## Output-et

```text
reports/day9_system_test_metrics.json
reports/day9_system_test_cases.csv
reports/day9_demo_examples.csv
reports/day9_system_testing.md
```

Komanda:

```powershell
python src\models\evaluate_app_system.py
```

## Rekomandimi për Ditën 10

Të krijohet një dataset i vogël i jashtëm, i verifikuar manualisht dhe i papërdorur në trajnim, me burime, data dhe etiketa të dokumentuara. App-i duhet testuar mbi të pa ndryshuar modelin. Rezultatet duhet të krahasohen me test set-in aktual dhe të përdoren për kapitullin e kufizimeve, etikës dhe vlefshmërisë së jashtme të diplomës.
