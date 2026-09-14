# Validimi i dataset-it dhe kontrolli i leakage-it

## 1. Qëllimi

Të kthehej corpus-i raw në një dataset të strukturuar dhe të kontrollohej që ndarja train/test nuk lejonte kalimin e çifteve ose kopjeve të tekstit midis grupeve të vlerësimit.

## 2. Çfarë u provua

U lexuan file-t real/fake, u ndanë titulli dhe përmbajtja, u krijuan `article_id` dhe `pair_id`, u kontrolluan kolonat, mungesat, etiketat, dublikatat dhe shpërndarjet. Teksti u normalizua në Unicode NFC dhe split-i u krijua sipas `pair_id`; në evaluation u kontrollua edhe mbivendosja e tekstit identik.

## 3. Rezultatet kryesore

- Dataset-i i përpunuar ka 3,994 artikuj.
- Split-i i ngrirë ka 3,195 artikuj train dhe 799 test.
- U identifikuan 7 artikuj test me tekst identik me train-in.
- Test set-i zyrtar pas përjashtimit ka 792 artikuj: 399 real dhe 393 fake.
- Nuk ekziston një CSV/JSON ose figurë e veçantë e këtij auditimi; shifrat verifikohen në `reports/final/metrics.json` dhe `FINAL_REPORT.md`.

## 4. Përfundimi

Dataset-i është i përdorshëm, por çdo evaluation duhet të ruajë grupet sipas `pair_id` dhe tekstit identik. Split-et ekzistuese duhet të mbeten të ngrira.

## 5. Vendimi / pse kaluam në eksperimentin tjetër

Pasi u sigurua një test set i pastër, u krijua një pikë reference minimale me DummyClassifier dhe më pas baseline-i i parë që mëson nga teksti.

## 6. Artefaktet dhe script-et përkatëse

- Script-e: `src/data/load_dataset.py`, `src/data/validate_dataset.py`, `src/data/build_dataset.py`, `src/preprocessing/clean_text.py`, `src/evaluation/data_utils.py`.
- Dataset-e: `data/processed/articles.csv`, `data/processed/articles.parquet`, `data/interim/articles_clean.csv`, `data/interim/train.csv`, `data/interim/test.csv`.
- Burime verifikimi: `reports/final/metrics.json`, `reports/final/FINAL_REPORT.md`.
- Model: nuk prodhohet model në këtë fazë.
- Figura të verifikueshme: të padisponueshme.
