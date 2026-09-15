# Analiza gjuhësore e dataset-it

## 1. Qëllimi

Të njihej dataset-i përpara ndërtimit të modeleve dhe të kuptohej se si ndryshojnë lajmet real dhe fake në gjatësi, strukturë, pikësim, kapitalizim, shprehje sensacionale, tregues burimi, pasiguri dhe përdorim të shkronjave `ë/ç`.

## 2. Çfarë u analizua

U analizuan të gjithë 3,994 artikujt: 1,998 lajme real dhe 1,996 lajme fake. Për çdo karakteristikë u krahasua mesatarja e dy klasave dhe shpërndarja e vlerave. Marker-at gjuhësorë u paraqitën edhe për 100 fjalë, në mënyrë që ndryshimi i madh në gjatësinë e artikujve të mos krijojë krahasime të padrejta.

Analiza është përshkruese dhe nuk përdor teste statistikore. Ajo synon ta bëjë dataset-in të kuptueshëm përpara se të analizohen modelet.

## 3. Rezultatet kryesore

- Dataset-i është pothuajse plotësisht i balancuar: 50.03% real dhe 49.97% fake.
- Lajmet real janë dukshëm më të gjata: mesatarisht rreth 291 fjalë, kundrejt 132 fjalëve për lajmet fake.
- Lajmet real kanë më shumë fjali, ndërsa lajmet fake kanë tituj mesatarisht më të gjatë.
- Pas normalizimit për 100 fjalë, lajmet real përdorin më shpesh tregues burimi dhe shprehje pasigurie.
- Kur shenjat llogariten për 100 fjalë, lajmet real përdorin më shumë presje dhe thonjëza, ndërsa lajmet fake përdorin më shumë pikëçuditëse, pikëpyetje dhe tri pika.
- Lajmet fake përdorin pak më shpesh kapitalizim dhe shprehje sensacionale.
- Përdorimi i shkronjave `ë/ç` është mesatarisht më i lartë te lajmet real.
- Emoji-t janë shumë të rralla: gjenden vetëm në 26 nga 3,994 artikuj. Prej tyre, 14 janë lajme real dhe 12 fake, prandaj nuk paraqesin dallim të dobishëm ndërmjet klasave.

Tabela [linguistic_comparison.csv](linguistic_comparison.csv) ruan mesataren për lajmet real, mesataren për lajmet fake, diferencën dhe klasën ku secila karakteristikë është më e lartë. Tabela [punctuation_comparison.csv](punctuation_comparison.csv) paraqet veçmas çdo shenjë pikësimi për 100 fjalë. Tabela [emoji_summary.csv](emoji_summary.csv) dokumenton përdorimin e rrallë të emoji-ve pa i trajtuar ato si feature të modelit.

## 4. Përfundimi

Dy klasat kanë dallime të dukshme gjuhësore, veçanërisht në gjatësinë e tekstit, përdorimin e burimeve, pasigurinë, kapitalizimin dhe diakritikat. Këto dallime përshkruajnë corpus-in e përdorur dhe nuk duhet të interpretohen si rregulla universale për vërtetësinë e një lajmi. Gjatësia, në veçanti, mund të jetë bias i dataset-it.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Meqë karakteristikat gjuhësore dallojnë mes klasave, ato u testuan më pas si input i një modeli linguistic-only dhe i modeleve hybrid. Ky hap kontrolloi nëse dallimet përshkruese sillnin përmirësim real mbi TF-IDF.

## 6. Artefaktet dhe script-et përkatëse

- Script-e: `src/features/linguistic_features.py`, `src/features/build_linguistic_features.py`, `archive/experiments/features/analyze_linguistic_features.py`.
- Dataset: `data/processed/linguistic_features.csv`.
- Tabela: [feature_summary.csv](feature_summary.csv), [linguistic_comparison.csv](linguistic_comparison.csv), [punctuation_comparison.csv](punctuation_comparison.csv), [emoji_summary.csv](emoji_summary.csv).
- Figura: [shpërndarja e lajmeve real dhe fake](figures/shperndarja_lajmeve_real_fake.png), [shpërndarja e gjatësisë së lajmeve](figures/shperndarja_gjatesise_lajmeve.png), [krahasimi i gjatësisë dhe strukturës](figures/krahasimi_gjatesise_dhe_struktures.png), [krahasimi i shenjave të pikësimit](figures/krahasimi_shenjave_te_pikesimit.png), [krahasimi i sinjaleve gjuhësore](figures/krahasimi_sinjaleve_gjuhesore.png), [krahasimi i kapitalizimit dhe diakritikave](figures/krahasimi_kapitalizimit_dhe_diacritikave.png).
- Model: nuk prodhohet model në këtë fazë.
