# Dita 5: Modeli hibrid

## Qëllimi

U ndërtua dhe u vlerësua modeli `TF-IDF + linguistic features` me Logistic Regression. Krahasimi përdor të njëjtin train dhe të njëjtin test të pastër për të katër provat.

## Kontrolli i output-eve të mëparshme

- Dataset i pastruar: 3994 artikuj.
- Train fillestar: 3195 artikuj.
- Test fillestar: 799 artikuj.
- Mbivendosje `article_id` train/test: 0.
- Mbivendosje `pair_id` train/test: 0.
- Rreshta pa linguistic features: 0.
- Mospërputhje labels: 0.
- Mospërputhje `pair_id`: 0.

U gjetën 7 artikuj në test me tekst identik me një tekst në train. Ata u përjashtuan vetëm nga vlerësimi i Ditës 5, prandaj testi i pastër ka 792 artikuj. Kjo shmang vlerësimin mbi kopje që modeli i ka parë gjatë trajnimit.

## Bashkimi i të dhënave

Teksti dhe karakteristikat gjuhësore u bashkuan vetëm me `article_id`. Pas bashkimit u kontrolluan përsëri `pair_id`, `label` dhe `label_name`; rendi i rreshtave nuk u përdor si supozim.

TF-IDF përdor `model_text`, që bashkon titullin me përmbajtjen pas normalizimit bazë të hapësirave. Shkronjat shqipe, kapitalizimi dhe pikësimi ruhen.

U përdorën 29 karakteristika numerike:

- strukturë dhe gjatësi: `word_count`, `sentence_count`, `character_count`, `avg_word_length`, `avg_sentence_length`, `title_length`, `content_length`;
- pikësim: count-et dhe ratio-t për pikëçuditëse, pikëpyetje, presje, thonjëza dhe tri pika;
- kapitalizim: `uppercase_word_count`, `uppercase_word_ratio`, `uppercase_char_ratio`, `title_excessive_uppercase`;
- diakritika: count-et për `ë`/`ç`, `diacritic_count`, `diacritic_ratio`, dhe sinjali për diakritika të mundshme që mungojnë;
- shprehje: count-et dhe ratio-t për fjalë sensacionale, tregues burimi dhe pasiguri.

`TfidfVectorizer`, imputimi dhe standardizimi u përshtatën vetëm mbi train brenda pipeline-it.

## Rezultatet

| Modeli | Accuracy | Precision | Recall | F1 | F1 fake |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF only | 0.8977 | 0.9012 | 0.8977 | 0.8975 | 0.8919 |
| Linguistic features only | 0.8270 | 0.8272 | 0.8270 | 0.8270 | 0.8237 |
| TF-IDF + linguistic features | 0.8902 | 0.8912 | 0.8902 | 0.8901 | 0.8863 |
| Hybrid without length features | 0.8914 | 0.8920 | 0.8914 | 0.8914 | 0.8883 |

Kolonat Precision, Recall dhe F1 janë mesatare të ponderuara për të dy klasat. `F1 fake` paraqitet veçmas për klasën me label `1`.

Confusion matrix përdor rendin `[[real→real, real→fake], [fake→real, fake→fake]]`:

- TF-IDF only: `[[377, 22], [59, 334]]`
- Linguistic features only: `[[335, 64], [73, 320]]`
- Hybrid: `[[366, 33], [54, 339]]`
- Hybrid pa feature-t e gjatësisë: `[[364, 35], [51, 342]]`

## Prova pa feature-t e gjatësisë

U hoqën: `word_count`, `sentence_count`, `character_count`, `avg_sentence_length`, `title_length`, `content_length`.

Accuracy ndryshoi me +0.0012 kundrejt modelit hibrid të plotë. Kjo tregon se në kombinimin aktual heqja e feature-ve direkte të gjatësisë nuk e dëmtoi rezultatin. Sinjalet e tjera gjuhësore mbetën të përdorshme, por nuk e kaluan TF-IDF baseline.

## Përfundimi i krahasimit

Modeli hibrid ndryshoi accuracy me -0.0075 kundrejt TF-IDF only. Pra linguistic features nuk e përmirësuan klasifikimin në këtë konfigurim të parë. Modeli më i mirë sipas F1 për klasën fake ishte **TF-IDF only**.

Për aplikacionin e ardhshëm, TF-IDF only është kandidati më i mirë aktual për parashikimin. Karakteristikat gjuhësore mbeten të vlefshme për shpjegimin e input-it dhe për analiza, edhe pse bashkimi i tyre nuk dha rritje metrike.

## Shembull parashikimi

- `article_id`: `true_290`
- Label real në dataset: `real`
- Parashikimi: `real`
- Probabilitet real sipas modelit: 0.7849
- Probabilitet fake sipas modelit: 0.2151
- Fjalë sensacionale: tronditëse
- Tregues burimi: sipas, raporti
- Pikëçuditëse: 0
- Numër fjalësh: 249
- Gjatësi teksti: 1469
- `diacritic_ratio`: 0.076242
- `uppercase_ratio`: 0.022071

Ky është probabilitet sipas modelit dhe karakteristikave gjuhësore. Nuk zëvendëson verifikimin faktik të lajmit.

## Kufizime

- Rezultatet vijnë nga një ndarje e vetme train/test.
- Shtatë kopje ekzakte u gjetën pas ndarjes së vjetër dhe u hoqën nga testi i vlerësimit.
- Dataseti mund të përmbajë sinjale të burimit, temës ose gjatësisë, jo vetëm sinjale të vërtetësisë.
- Listat e fjalëve sensacionale dhe source markers janë manuale dhe fillestare.
- Probabilitetet e Logistic Regression nuk janë kalibruar ende.
- Karakteristikat gjuhësore nuk bëjnë verifikim faktesh.

## Hapi i rekomanduar për Ditën 6

Të bëhet error analysis për rastet ku TF-IDF dhe modeli hibrid gabojnë, pastaj të kontrollohet kalibrimi i probabiliteteve dhe pragu për rezultatin `i pasigurt`. Kjo duhet bërë para ndërtimit të Streamlit.
