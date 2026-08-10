# Model Modules

## Runtime final

- `prediction_utils.py`: pragjet, vendimi me tri nivele dhe sinjalet gjuhësore.
- `predict_final.py`: ngarkimi dhe prediction-i i modelit final `v1.0.0`.

Këto janë modulet që përdor `app/streamlit_app.py`.

## Ngrirja e modelit

- `finalize_model.py`: verifikon kandidatin e Ditës 16 dhe krijon manifestin e
  artefaktit final pa bërë trajnim.
- `calibrate_linear_svm.py`: eksperimenti i calibration-it që prodhoi
  kandidatin final.

## Analiza historike

Modulet e tjera dokumentojnë rrugën eksperimentale: baseline Logistic
Regression, linguistic/hybrid models, external evaluation, domain shift,
Word/Character TF-IDF, classifier comparison dhe SVM tuning. Ato ruhen për
riprodhueshmëri akademike, por nuk importohen nga Streamlit.

`predict.py` ruan API-të e modeleve historike. Runtime-i final përdor
`predict_final.py`.
