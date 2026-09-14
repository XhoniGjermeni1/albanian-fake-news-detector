# Analiza e gjatësisë dhe domain shift-it

## 1. Qëllimi

Të kontrollohej nëse gjatësia e artikullit shpjegon gabimet dhe rënien e performancës nga test-i i brendshëm te benchmark-u i jashtëm.

## 2. Çfarë u provua

Performanca dhe probabiliteti fake u analizuan sipas grupeve të gjatësisë dhe sipas klasës. U provuan variante të kontrolluara ku artikujt u shkurtuan ose u zgjeruan dhe u krahasuan karakteristikat e domain-it të brendshëm me atë të jashtëm.

## 3. Rezultatet kryesore

U gjet lidhje negative midis gjatësisë dhe probabilitetit fake, edhe brenda klasave. Në vlerësimin final, fake mbi 250 fjalë kishin recall 0.4483, ndërsa grupi 61–120 fjalë kishte accuracy 0.9560. Tabela zyrtare gjendet te [length_metrics.csv](length_metrics.csv).

## 4. Përfundimi

Modeli nuk merr `word_count` si feature numerike, por gjatësia ndikon indirekt përmes numrit dhe shpërndarjes së n-grameve TF-IDF. Gjatësia shpjegon vetëm një pjesë të domain shift-it; burimi, periudha, tema dhe stili mbeten faktorë të rëndësishëm.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Për të kapur sinjale më të imta ortografike dhe për të ulur varësinë nga fjalët e plota, u krahasuan Word, Character dhe Word+Character TF-IDF.

## 6. Artefaktet dhe script-et përkatëse

- Entrypoint: `archive/experiments/models/analyze_length_domain_shift.py`.
- Zbatim: `archive/experiments/models/experiment_support/day12_analysis.py`.
- Dataset-e: `data/interim/train.csv`, `data/interim/test.csv`, `data/external/external_news.csv`, `data/interim/day12_external_expansions.csv`.
- Rezultat final i ripërdorur: `reports/final/length_metrics.csv`.
- Figurat historike: [domain_shift.png](figures/domain_shift.png), [external_expansion.png](figures/external_expansion.png), [internal_length_performance.png](figures/internal_length_performance.png), [internal_stability.png](figures/internal_stability.png), [probability_vs_length.png](figures/probability_vs_length.png).
