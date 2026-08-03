# Dita 10 - Dataseti i jashtëm

## Qëllimi

Qëllimi i Ditës 10 ishte krijimi i një dataseti të vogël në gjuhën shqipe, të
dokumentuar dhe të papërdorur në trajnim. Modeli nuk u ritrajnua, pragjet nuk u
ndryshuan dhe nuk u ekzekutua asnjë prediction mbi rastet e reja.

## Struktura

Dataseti ruhet te `data/external/external_news.csv` dhe ka këto kolona:

- `external_id`, `title`, `content` dhe `label`;
- `source`, `url` dhe `published_date`;
- `label_evidence` dhe `evidence_url`;
- `topic`, `content_origin` dhe `review_status`.

Tekstet në `title` dhe `content` janë përmbledhje manuale në shqip, jo kopje të
plota të artikujve. Prova e etiketimit është ruajtur veçmas, që teksti që do të
analizojë modeli të mos përmbajë përgënjeshtrimin ose përgjigjen e saktë.

## Përbërja

U pranuan **40 artikuj/raste**:

| Tema | Real | Fake | Gjithsej |
| --- | ---: | ---: | ---: |
| Politikë | 4 | 4 | 8 |
| Shëndetësi | 4 | 4 | 8 |
| Ekonomi | 4 | 4 | 8 |
| Sociale | 4 | 4 | 8 |
| Teknologji | 4 | 4 | 8 |
| **Gjithsej** | **20** | **20** | **40** |

Përmbajtjet kanë 32–43 fjalë, me mesatare 37.15 fjalë. Datat shtrihen nga
korriku 2024 deri në korrik 2026.

## Etiketimi

Për `real` u përdorën njoftime institucionale:

- 16 raste nga Këshilli i Ministrave i Shqipërisë;
- 2 raste nga Banka e Shqipërisë;
- 2 raste nga INSTAT.

Etiketa `real` i referohet vetëm ngjarjes, vendimit ose statistikës konkrete që
është përmbledhur. Ajo nuk do të thotë se çdo deklaratë politike brenda faqes
burimore është verifikuar në mënyrë të pavarur.

Për `fake` u përdorën vetëm raste të kontrolluara nga Krypometër me vlerësimin
e qartë `Rrenë`. Provat përfshijnë krahasim me materialin origjinal, kontroll të
dokumenteve dhe burimeve zyrtare, deklarata të personave ose institucioneve dhe
analizë të përmbajtjeve të gjeneruara me inteligjencë artificiale.

Gjatë përzgjedhjes u gjetën dhe u hoqën **3 kandidatë me etiketë të paqartë
`Me kos`**:

- video të fabrikuara të anëtarëve të LDK-së;
- pretendimi për dënimin e Sami Lushtakut;
- videoja satirike për klonimin e Donald Trump.

Këto raste nuk u përdorën për vlerësimin binar `real/fake`.

## Kontrollet e cilësisë

U ekzekutua:

```powershell
python src\data\validate_external_dataset.py
```

Rezultatet:

| Kontrolli | Rezultati |
| --- | ---: |
| Mungesa në kolonat e detyrueshme | 0 |
| Etiketa ose tema të pavlefshme | 0 |
| Raste pa miratim manual | 0 |
| Data të pavlefshme | 0 |
| URL me format të pavlefshëm | 0 |
| URL që nuk u hapën gjatë kontrollit online | 0 nga 40 |
| URL identike me URL-të e corpus-it raw | 0 |
| Përmbajtje nën 25 fjalë | 0 |
| Prova etiketimi shumë të shkurtra | 0 |
| Dublikata ID, URL, titull ose përmbajtje | 0 |
| Probleme Unicode NFC ose karaktere zëvendësuese | 0 |

Të 40 rreshtat përmbajnë të paktën një shkronjë shqipe `ë` ose `ç` dhe ruhen
në UTF-8.

## Kontrolli ndaj train set-it

Çdo titull, përmbajtje dhe tekst i bashkuar u krahasua me 3,994 artikujt e
`data/interim/articles_clean.csv`.

Gjithashtu u lexuan 3,994 skedarët `true-meta-information` dhe
`fake-meta-information`, të cilët përmbanin 7,988 URL unike të postimeve dhe
artikujve burimorë.

- tituj ose tekste identike: **0**;
- URL identike me metadata-t raw: **0**;
- raste me ngjashmëri të paktën 0.90: **0**;
- ngjashmëria më e lartë me një artikull trajnimi: **0.339944**;
- mesatarja e ngjashmërisë maksimale: **0.206203**;
- çifte brenda datasetit të jashtëm me ngjashmëri të paktën 0.90: **0**.

Për kontrollin e ngjashmërisë u përdor TF-IDF me n-gramë karakteresh vetëm si
mjet auditimi. Ky nuk është trajnim ose testim i modelit të lajmeve.

## Output-et

```text
data/external/external_news.csv
data/external/README.md
src/data/validate_external_dataset.py
tests/test_external_dataset.py
reports/day10_external_dataset_audit.json
reports/day10_external_similarity_review.csv
reports/day10_external_dataset.md
```

## Kufizimet

- Dataseti është i vogël dhe nuk përfaqëson të gjitha mediat ose dialektet
  shqiptare.
- Përmbajtjet janë përmbledhje manuale 32–43 fjalë, jo artikujt e plotë.
- Raste `real` vijnë kryesisht nga burime institucionale, ndërsa rastet `fake`
  nga pretendime në rrjete sociale të dokumentuara nga një fact-checker. Ky
  ndryshim burimi dhe stili mund të ndikojë te modeli.
- Disa raste `fake` lidhen me imazhe ose video; modeli do të shohë vetëm
  përshkrimin tekstual dhe nuk analizon median origjinale.
- Verifikimi i URL-ve tregon se faqet ishin të arritshme më 03.08.2026, por
  përmbajtja online mund të ndryshojë ose të hiqet më vonë.

Për këto arsye dataseti është **gati për një vlerësim të jashtëm pilot**, por jo
si benchmark përfundimtar i përgjithësimit të modelit.

## Dita 11

Në Ditën 11 duhet të ngarkohet modeli i ruajtur pa ritrajnim dhe pa ndryshim të
pragjeve, të bëhen prediction-et për 40 rastet dhe të llogariten accuracy,
precision, recall, F1, confusion matrix dhe shpërndarja
`likely_real/uncertain/likely_fake`. Rezultatet duhen analizuar edhe sipas temës,
etiketës dhe gjatësisë, duke theksuar veçmas ndikimin e përmbledhjeve manuale dhe
ndryshimin e burimeve.
