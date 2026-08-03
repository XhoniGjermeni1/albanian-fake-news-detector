# Dataseti i jashtëm i Ditës 10

Skedari `external_news.csv` përmban 40 raste në gjuhën shqipe për vlerësimin e
jashtëm të modelit në një ditë të mëvonshme. Ai nuk është përdorur për trajnim
dhe Dita 10 nuk ekzekuton parashikime mbi këto raste.

## Përmbajtja

- 20 raste `real` dhe 20 raste `fake`;
- 8 raste për secilën temë: politikë, shëndetësi, ekonomi, sociale dhe
  teknologji;
- brenda çdo teme ka 4 raste `real` dhe 4 raste `fake`;
- datat shtrihen nga korriku 2024 deri në korrik 2026, pas periudhës së corpus-it
  të përdorur për trajnim.

Kolonat janë:

| Kolona | Përshkrimi |
| --- | --- |
| `external_id` | ID unike e rastit |
| `title` | Titulli që do t'i jepet modelit |
| `content` | Përmbledhje neutrale në shqip që do t'i jepet modelit |
| `label` | `real` ose `fake` |
| `source` | Institucioni ose lloji i burimit |
| `url` | Faqja nga e cila u dokumentua rasti |
| `published_date` | Data e publikimit të faqes burimore |
| `label_evidence` | Arsyeja e dokumentuar për etiketën |
| `evidence_url` | Burimi që mbështet etiketën |
| `topic` | Tema kryesore |
| `content_origin` | Mënyra si u krijua teksti i ruajtur |
| `review_status` | Statusi i kontrollit manual |

## Metoda e mbledhjes

Tekstet në `title` dhe `content` janë përmbledhje manuale dhe jo kopje të plota
të artikujve. Kjo ruan një format të njëtrajtshëm, shmang ripublikimin e tekstit
të mbrojtur dhe ndan tekstin që do të shohë modeli nga prova e etiketimit.
Kolona `content_origin` e dokumenton këtë me vlerën `manual_summary_sq`.

Rastet `real` mbështeten në njoftime institucionale nga Këshilli i Ministrave,
Banka e Shqipërisë dhe INSTAT. Etiketa mbulon ngjarjen, vendimin ose statistikën
konkrete të përmbledhur; ajo nuk vërteton deklarata subjektive politike.

Rastet `fake` mbështeten te verifikime të Krypometrit me vlerësimin e qartë
`Rrenë`. Përmbajtja përshkruan pretendimin që qarkulloi, ndërsa përgënjeshtrimi
ruhet vetëm te `label_evidence`, që të mos i japë modelit përgjigjen brenda
input-it. Tre kandidatë me vlerësimin `Me kos` u përjashtuan si etiketa jo mjaft
të qarta për vlerësim binar.

## Kontrolli

Nga rrënja e projektit ekzekuto:

```powershell
python src\data\validate_external_dataset.py
```

Kontrolli verifikon kolonat, mungesat, etiketat, datat, URL-të, gjatësinë,
dublikatat, Unicode NFC dhe ngjashmërinë me datasetin e trajnimit. URL-të
krahasohen edhe me skedarët raw `true-meta-information` dhe
`fake-meta-information`. Rezultatet ruhen te:

```text
reports/day10_external_dataset_audit.json
reports/day10_external_similarity_review.csv
```

Kufizimi kryesor është se ky është një dataset pilot me përmbledhje të shkruara
manualisht dhe jo një corpus me artikuj të plotë nga shumë redaksi. Në Ditën 11
rezultatet e modelit duhet të interpretohen si vlerësim pilot dhe jo si matje
përfundimtare e përgjithësimit në të gjitha mediat shqiptare.
