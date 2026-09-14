# Analiza e karakteristikave gjuhësore

## 1. Qëllimi

Të matej nëse lajmet real dhe fake ndryshojnë në gjatësi, strukturë, pikësim, kapitalizim, fraza sensacionale, tregues burimi, pasiguri dhe përdorim të diakritikave.

## 2. Çfarë u provua

U nxorën karakteristika numerike për 3,994 artikuj. Mesataret sipas klasës u krahasuan dhe për veçoritë kryesore u përdorën Mann–Whitney U dhe Cohen's d për rëndësinë statistikore dhe madhësinë e efektit.

## 3. Rezultatet kryesore

[feature_summary.csv](feature_summary.csv) ruan mesataret real/fake. Figurat tregojnë shpërndarjen e gjatësisë, raportet, marker-at dhe efektet kryesore. CSV-ja historike e plotë me Mann–Whitney U dhe Cohen's d nuk ekziston aktualisht, ndaj p-vlerat individuale nuk riprodhohen në këtë përmbledhje.

## 4. Përfundimi

Karakteristikat gjuhësore përmbajnë sinjal dallues, por disa prej tyre—sidomos gjatësia—mund të pasqyrojnë bias të corpus-it dhe jo një rregull të përgjithshëm të lajmeve fake.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Meqë sinjalet ishin të matshme, u trajnuan modele linguistic-only dhe hybrid për të provuar nëse ato përmirësojnë baseline-in TF-IDF.

## 6. Artefaktet dhe script-et përkatëse

- Script-e: `src/features/linguistic_features.py`, `src/features/build_linguistic_features.py`, `archive/experiments/features/analyze_linguistic_features.py`.
- Dataset: `data/processed/linguistic_features.csv`.
- Rezultat: [feature_summary.csv](feature_summary.csv).
- Figura: [word_count_distribution.png](figures/word_count_distribution.png), [marker_count_means.png](figures/marker_count_means.png), [ratio_feature_means.png](figures/ratio_feature_means.png), [top_effect_sizes.png](figures/top_effect_sizes.png).
- Model: nuk prodhohet model në këtë fazë.
