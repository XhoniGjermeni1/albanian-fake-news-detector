# Model Modules

## Runtime final

- `predict_final.py`: ngarkon artefaktin e ngrirë, përgatit tekstin, merr
  probabilitetet dhe ndërton kontratën publike `predict_final_news()`.
- `prediction_utils.py`: mban pragjet `0.30/0.70`, vendimin me tri nivele dhe
  linguistic explanation.

Këto janë të vetmet module të kësaj dosjeje që përdor Streamlit. Modeli i
kalibruar është brenda artefaktit `.joblib`; runtime-i nuk ritrajnon dhe nuk
rikalibron asgjë.

## Eksperimentet historike

Skriptet e tjera dokumentojnë baseline-in, linguistic/hybrid models, external
evaluation, domain shift, krahasimin TF-IDF, classifier selection, tuning,
calibration dhe ngrirjen finale. Ato ruhen në path-et ekzistuese për të mos
prishur testet dhe riprodhueshmërinë e raporteve.

Indeksi i plotë është te `experiments/README.md`. `predict.py` ruan API-të e
modeleve të vjetra; runtime-i final përdor vetëm `predict_final.py`.
