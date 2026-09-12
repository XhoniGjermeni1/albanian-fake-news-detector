# Dita 1 - Auditimi i datasetit

Burimi i datasetit: `https://github.com/rexshijaku/alb-fake-news-corpus`

Artikulli ACM: `https://dl.acm.org/doi/10.1145/3487288`

Dataseti i përpunuar për Ditën 1 ruhet në format CSV te `data/processed/articles.csv`. Ruhet edhe një kopje Parquet te `data/processed/articles.parquet` për përdorim më efikas në Python.

## Struktura e gjetur

Repository u vendos te `data/raw/alb-fake-news-corpus/`.

Në rrënjën e repository-t u gjetën:

- `.git/`
- `full_texts/`
- `readme.md`

Brenda `full_texts/` u gjetën:

| Dosja | Përshkrimi | Numri i skedarëve |
| --- | --- | ---: |
| `true/` | artikuj të vërtetë | 1998 |
| `fake/` | artikuj të pavërtetë | 1996 |
| `true-pos/` | artikuj të vërtetë me POS tags | 1998 |
| `fake-pos/` | artikuj të pavërtetë me POS tags | 1996 |
| `true-meta-information/` | metadata për artikujt e vërtetë | 1998 |
| `fake-meta-information/` | metadata për artikujt e pavërtetë | 1996 |

Shembuj path-esh:

- `data/raw/alb-fake-news-corpus/full_texts/true/1.txt`
- `data/raw/alb-fake-news-corpus/full_texts/fake/1.txt`
- `data/raw/alb-fake-news-corpus/full_texts/true-pos/1.txt`
- `data/raw/alb-fake-news-corpus/full_texts/fake-meta-information/1.txt`

## Pair IDs

Emrat e skedarëve janë numerikë dhe përdoren si `pair_id`. Çiftet nuk janë plotësisht të balancuara:

- `pair_id` vetëm te `true`: `362`, `425`, `565`, `735`
- `pair_id` vetëm te `fake`: `956`, `1632`

## Validimi bazë pas ngarkimit

- Artikuj total: 3994
- Artikuj real/true: 1998
- Artikuj fake: 1996
- Tituj që mungojnë: 0
- Përmbajtje që mungojnë: 0
- `pair_id` që mungojnë: 0
- Tekste të duplikuara: 41 rreshta në 20 grupe
- Artikuj shumë të shkurtër nën 80 karaktere: 0

Statistika të gjatësisë:

| Fusha | Minimum | Maksimum | Mesatare | Medianë |
| --- | ---: | ---: | ---: | ---: |
| `title` karaktere | 6 | 185 | 74.67 | 74.0 |
| `content` karaktere | 40 | 35862 | 1173.93 | 712.0 |
| `raw_text` karaktere | 91 | 35946 | 1249.6 | 784.5 |
| `raw_text` fjalë | 18 | 5894 | 207.73 | 132.0 |

## Probleme të mundshme

- Dataseti nuk është plotësisht i çiftëzuar sipas ID-ve numerike.
- Ka tekste të duplikuara që duhen analizuar përpara trajnimit.
- Për Ditën 1 janë përdorur vetëm dosjet `true/` dhe `fake/`; POS dhe metadata u identifikuan, por nuk u integruan ende në datasetin e përpunuar.
