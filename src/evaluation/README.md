# Evaluation Utilities

Kjo dosje përmban vetëm utility që përdoren nga më shumë se një eksperiment:

- `data_utils.py`: rindërtimi i `model_text`, heqja e train/test duplicates,
  group-safe folds dhe grupet e gjatësisë;
- `metrics.py`: metrikat binare, decision scores dhe rrumbullakimi për raporte;
- `experiment_utils.py`: SHA-256 dhe tabela të thjeshta Markdown.

Këto module nuk trajnojnë modele dhe nuk shkruajnë raporte vetë. Metodologjia
specifike mbetet në skriptin përkatës të eksperimentit.
